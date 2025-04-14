from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta
import numpy as np

import math
import random
from scipy.stats import beta


def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)

    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power

    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space

    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha

    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.

    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda

    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """

    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1-mask)
    return x1+x2, index, lam


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes, smoothing=0.):
        self.num_classes = num_classes
        self.off_value = smoothing / self.num_classes
        self.on_value = 1. - smoothing + self.off_value

    @abstractmethod
    def do_blending(self, imgs, label, **kwargs):
        pass

    def __call__(self, imgs, label, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probablity distribution over classes) are float tensors
        with the shape of (B, 1, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (torch.Tensor): Hard labels, integer tensor with the shape
                of (B, 1) and all elements are in range [0, num_classes).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            mixed_label (torch.Tensor): Blended soft labels, float tensor with
                the shape of (B, 1, num_classes) and all elements are in range
                [0, 1].
        """
        one_hot_label = one_hot(label, num_classes=self.num_classes, on_value=self.on_value, off_value=self.off_value, device=label.device)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        return mixed_imgs, mixed_label


class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2, smoothing=0.):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.
    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2, smoothing=0.):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label


class LabelSmoothing(BaseMiniBatchBlending):
    def do_blending(self, imgs, label, **kwargs):
        return imgs, label


class CutmixMixupBlending(BaseMiniBatchBlending):
    def __init__(self, num_classes=400, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]
        return imgs, label

    def do_mixup(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with MViT style. Cutmix for half for mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob :
            return self.do_cutmix(imgs, label)
        else:
            return self.do_mixup(imgs, label)


class FixMixupBlending(BaseMiniBatchBlending):
    def __init__(self, num_classes=400, smoothing=0.1, mixup_alpha=.8, fmix_alpha=1, switch_prob=0.5, decay_power=3, max_soft=0.0, size=(224, 224), reformulate=False):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(fmix_alpha, fmix_alpha)
        self.switch_prob = switch_prob
        self.decay_power = decay_power
        self.max_soft = max_soft
        self.size = size
        self.reformulate = reformulate
        self.alpha = fmix_alpha

    def do_fmix(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)

        rand_index = torch.randperm(imgs.size(0)).to(imgs.device)
        mask = torch.from_numpy(mask).float().to(imgs.device)
        x1 = mask * imgs
        x2 = (1 - mask) * imgs[rand_index]

        label = lam * label + (1 - lam) * label[rand_index, :]
        return x1+x2, label

    def do_video_mixup(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample()
        batch_size = imgs.size(0)
        T_size = imgs.size(1)
        rand_index = torch.randperm(batch_size)

        rand_T = torch.randperm(T_size)
        num_T = int(lam*T_size)
        rand_T_index_index = rand_T[:num_T]

        for i in range(batch_size):
            imgs[i, rand_T_index_index] = imgs[rand_index[i], rand_T_index_index]
        label = (1 - lam) * label + lam * label[rand_index, :]

        return imgs, label

    def do_mixup(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with MViT style. Cutmix for half for mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_fmix(imgs, label)
        else:
            return self.do_mixup(imgs, label)