import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
import time
import numpy as np
import random
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, plot_confusion_matrix
from utils.logger import create_logger
from datasets.build import build_dataloader
from datasets.blending import FixMixupBlending
from utils.config import get_config
from models import dk_clip

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/dfew7/16_16.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--gpu', default=[0, 1], type=int,help='GPU id to use.')
    parser.add_argument('--output', type=str, default="DFEWAS2")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config):
    # load train and valid dataset
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

    # load pretrained model
    model, _ = dk_clip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                          device="cpu", jit=False,
                          T=config.DATA.NUM_FRAMES,
                          droppath=config.MODEL.DROP_PATH_RATE,
                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                          use_cache=config.MODEL.FIX_TEXT,
                          logger=logger,
                          N=config.DATA.NUM_DIVIDE,
                          cfg=config
                          )
    model = model.cuda()

    # training data augmentation
    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        criterion_soft = SoftTargetCrossEntropy()
        mixup_fn = FixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                    smoothing=config.AUG.LABEL_SMOOTH,
                                    mixup_alpha=config.AUG.MIXUP,
                                    fmix_alpha=config.AUG.CUTMIX,
                                    switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
        criterion_soft = SoftTargetCrossEntropy()

    else:
        criterion = nn.CrossEntropyLoss()
        criterion_soft = SoftTargetCrossEntropy()

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                      find_unused_parameters=True)

    start_epoch, max_acc_global, max_acc_local, max_acc_fuse = 0, 0.0, 0.0, 0.0
    # retrain
    if config.TRAIN.AUTO_RESUME:
        resume_file_path = auto_resume_helper(config.OUTPUT)
        if resume_file_path:
            config.defrost()
            config.MODEL.RESUME = resume_file_path
            config.freeze()
            logger.info(f'auto resuming from {resume_file_path}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        start_epoch, max_acc_global = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    # textual prompt
    text_labels = generate_text(train_data)

    # model test
    if config.TEST.ONLY_TEST:
        acc1, acc1_local, acc1_fuse = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return
    print("Start training")
    #model train and valid
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft)

        # Global, local and fuse classification accuracy
        acc_global, acc_local, acc_fuse = validate(val_loader, text_labels, model, config)
        logger.info(
            f"Accuracy of the network on the {len(val_data)} test videos: {acc_global:.2f}% {acc_local:.2f}% {acc_fuse:.2f}%")
        is_best = acc_global > max_acc_global
        max_acc_global = max(max_acc_global, acc_global)
        max_acc_local = max(max_acc_local, acc_local)
        max_acc_fuse = max(max_acc_fuse, acc_fuse)
        logger.info(f'Max accuracy: {max_acc_global:.2f}%  {max_acc_local:.2f}% {max_acc_fuse:.2f}%')
        # save model
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_acc_global, optimizer, lr_scheduler, logger, config.OUTPUT,
                         is_best)
    # validation after training
    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc:.1f}%")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, criterion_soft):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    scaler = GradScaler()
    texts = text_labels.cuda(non_blocking=True)

    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        with autocast():
            output_global, output_local, feat = model(images, texts)
            if epoch>13:
                total_loss = criterion(output_global, label_id)
                total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
            else:
                pre_output_local = torch.sum(
                    output_local.view(config.TRAIN.BATCH_SIZE, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES), dim=1).squeeze(dim=-1)
                # total_loss = global_loss + 1.0 * (local_loss + KL_loss)
                total_loss = criterion(output_global, label_id) + 1.0 * (criterion(pre_output_local, label_id) + criterion_soft(pre_output_local.softmax(dim=-1), output_global.softmax(dim=-1)))
                total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()

    acc_global_meter, acc_local_meter, acc_fuse_meter = AverageMeter(), AverageMeter(), AverageMeter()

    probility = []
    video_pre_global = []
    video_pre_local = []
    video_pre_fuse = []
    video_label = []
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()

            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_local = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                with autocast():
                    output, output_local, feat = model(image_input, text_inputs)
                if idx < 1:
                    feature = feat
                else:
                    feature = torch.cat((feature, feat), dim=0)

                pre_output_global = output.view(b, -1)
                pre_output_local = torch.sum(output_local.view(b, config.DATA.NUM_FRAMES // config.DATA.NUM_DIVIDE, config.DATA.NUM_CLASSES),dim=1).squeeze(dim=-1)

                similarity = pre_output_global.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity

                similarity_local = pre_output_local.view(b, -1).softmax(dim=-1)
                tot_similarity_local += similarity_local.view(b, -1)

            probility.extend(tot_similarity.data.cpu().numpy().copy())
            values_global, indices_global = tot_similarity.topk(1, dim=-1)

            values_local, indices_local = tot_similarity_local.topk(1, dim=-1)

            fuse_similarity = tot_similarity + tot_similarity_local
            values_fuse, indices_fuse = fuse_similarity.topk(1, dim=-1)

            acc_global = 0
            acc_local = 0
            acc_fuse = 0
            for i in range(b):
                video_pre_global.append(indices_global[i].data.cpu().numpy().copy())
                video_pre_local.append(indices_local[i].data.cpu().numpy().copy())
                video_pre_fuse.append(indices_fuse[i].data.cpu().numpy().copy())
                video_label.append(label_id[i].data.cpu().numpy().copy())
                if indices_global[i] == label_id[i]:
                    acc_global += 1
                if indices_local[i] == label_id[i]:
                    acc_local += 1
                if indices_fuse[i] == label_id[i]:
                    acc_fuse += 1

            acc_global_meter.update(float(acc_global) / b * 100, b)
            acc_local_meter.update(float(acc_local) / b * 100, b)
            acc_fuse_meter.update(float(acc_fuse) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc_global_meter.avg:.3f}\t'
                    f'Acc@1: {acc_local_meter.avg:.3f}\t'
                    f'Acc@1: {acc_fuse_meter.avg:.3f}\t'
                )
    # confusion matrix
    cf = confusion_matrix(video_label, video_pre_global)
    np.set_printoptions(precision=4)
    normalized_cm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100

    cls_cnt = normalized_cm.sum(axis=1)
    cls_hit = np.diag(normalized_cm)
    # print(cf)
    cls_acc = cls_hit / cls_cnt
    cls_acc = np.around(cls_acc, 4)
    cm = np.array(normalized_cm)
    # save_path = 'DK-CLIP/results'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # labels_name = ['hap', 'sad', 'neu', 'ang', 'sur', 'dis', 'fea']
    # plot_confusion_matrix(cm, labels_name, 'DKCLIP', cls_acc)
    #
    # #t-SNE
    # col = ['orange', 'purple', 'g', 'r', 'darkblue', 'chocolate', 'c']
    # x_embed = TSNE(n_components=2, perplexity=100, n_iter=10000).fit_transform(feature.data.cpu())
    # label = np.array(video_label)
    # plt.figure(figsize=(6, 6))
    # for i in range(7):
    #     idxs = np.where(label == i)[0]
    #     plt.scatter(x_embed[idxs, 0], x_embed[idxs, 1], color=col[i], s=6, label=labels_name[i])
    # plt.legend(loc='upper left')
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.savefig(os.path.join(save_path, 'DKCLIP_TSNE.jpg'), format='jpg')
    # plt.show()

    print(cls_acc)
    upper = np.mean(np.max(cf, axis=1) / cls_cnt)
    print('upper bound: {}'.format(upper))

    print('-----Evaluation is finished------')
    print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

    cf_local = confusion_matrix(video_label, video_pre_local).astype(float)
    cls_cnt_local = cf_local.sum(axis=1)
    cls_hit_local = np.diag(cf_local)
    # print(cf)
    cls_acc_local = cls_hit_local / cls_cnt_local
    cls_acc_local = np.around(cls_acc_local, 4)
    print(cls_acc_local)
    upper = np.mean(np.max(cf_local, axis=1) / cls_cnt_local)
    print('upper bound: {}'.format(upper))

    print('-----Evaluation is finished------')
    print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc_local) * 100))

    cf_fuse = confusion_matrix(video_label, video_pre_fuse).astype(float)
    cls_cnt_fuse = cf_fuse.sum(axis=1)
    cls_hit_fuse = np.diag(cf_fuse)
    # print(cf)
    cls_acc_fuse = cls_hit_fuse / cls_cnt_fuse
    cls_acc_fuse = np.around(cls_acc_fuse, 4)
    print(cls_acc_fuse)
    upper = np.mean(np.max(cf_fuse, axis=1) / cls_cnt_fuse)
    print('upper bound: {}'.format(upper))

    print('-----Evaluation is finished------')
    print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc_fuse) * 100))

    acc_global_meter.sync()
    acc_local_meter.sync()
    acc_fuse_meter.sync()
    logger.info(f' * Acc@1 {acc_global_meter.avg:.3f} Acc_loca@1 {acc_local_meter.avg:.3f}  Acc_fuse@1 {acc_fuse_meter.avg:.3f}')

    return acc_global_meter.avg, acc_local_meter.avg, acc_fuse_meter.avg


if __name__ == '__main__':
    args, config = parse_option()

    # 初始化分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])  # 使用环境变量中的 LOCAL_RANK
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        local_rank = args.local_rank  # 如果未设置环境变量，则使用命令行参数

    # 设置当前 GPU 设备
    torch.cuda.set_device(local_rank)

    # 初始化分布式进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 确保所有进程同步
    dist.barrier()

    # 设置随机种子
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 创建输出目录
    output_dir = Path(config.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = create_logger(output_dir=output_dir, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"Working directory: {output_dir}")

    # 保存配置文件（仅主进程执行）
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, output_dir)

    # 启动主训练逻辑
    main(config)