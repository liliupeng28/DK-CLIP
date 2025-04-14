import numpy as np
import torch.distributed as dist
import torch
import clip
import os
import matplotlib.pyplot as plt
from torch import nn
import itertools

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME):
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.
    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

# encode textual prompts
def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])
    return classes

# supervised contrastive loss
class UnifiedContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(UnifiedContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1)
        sum_pos = (y_true * torch.exp(-y_pred)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

def plot_confusion_matrix(cm, labels_name, title, acc):
    save_path = 'DK-CLIP/results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')
    # plt.title(title)
    num_class = np.array(range(len(labels_name)))
    plt.xticks(num_class, labels_name, rotation=45, fontsize=13)
    plt.yticks(num_class, labels_name, fontsize=13)
    plt.ylabel('Truth Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, title + ".jpg"), format='jpg')
    plt.close()