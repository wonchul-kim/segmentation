import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(T.Compose):
    def __call__(self, image, target, weight=None):
        if weight is None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        for t in self.transforms:
            image, target, weight = t(image, target, weight)
        return image, target, weight 
    
class Compose_(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, (size, size))
        target = F.resize(target, (size, size), interpolation=Image.NEAREST)

        return image, target

class RandomResize_(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, (size, size))

        return image

class RandomHorizontalFlip(object):
    def __init__(self, hflip_prob):
        self.hflip_prob = hflip_prob

    def __call__(self, image, target, weight=None):
        p = random.random()
        if p < self.hflip_prob:
            image = F.hflip(image)
            target = F.hflip(target)

        if weight is None:
            return image, target
        elif p > self.hflip_prob:
            weight = F.hflip(weight)

        return image, target, weight 

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.hflip_prob})'

class RandomVerticalFlip(object):
    def __init__(self, vflip_prob):
        self.vflip_prob = vflip_prob

    def __call__(self, image, target, weight=None):
        p = random.random()
        if p < self.vflip_prob:
            image = F.vflip(image)
            target = F.vflip(target)

        if weight is None:
            return image, target
        elif p > self.vflip_prob:
            weight = F.vflip(weight)

        return image, target, weight 

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.vflip_prob})'

class RandomRotate(object):
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob

    def __call__(self, image, target):
        p = random.random()
        if p < self.rotate_prob:
            deg = random.uniform(0, 180)
            image = F.rotate(image, deg, Image.BILINEAR)
            target = F.rotate(target, deg, Image.BILINEAR)

        return image, target



class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target, weight=None):
        if weight is None:
            image = F.to_tensor(image)
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
            # target = F.to_tensor(target)
            return image, target
        weight = weight.view(1, *weight.shape)
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # target = F.to_tensor(target)

        return image, target, weight

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor_(object):
    def __call__(self, image):
        image = F.to_tensor(image)

        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Normalize_(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class DoubleElasticTransform:
    """Based on implimentation on
    https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    def __init__(self, prob=0.2, alpha=250, sigma=10, seed=None, randinit=True):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
        self.randinit = randinit
        


    def __call__(self, image, mask, weight=None):
        if random.random() < self.prob:
            if self.randinit:
                seed = random.randint(1, 100)
                self.random_state = np.random.RandomState(seed)
                self.alpha = random.uniform(100, 300)
                self.sigma = random.uniform(10, 15)

            dim = image.shape
            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )

            image = image.numpy()
            mask = mask.numpy()
            x, y = np.meshgrid(np.arange(dim[1]), np.arange(dim[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            for ch in range(len(image)):
                image[ch, :, :] = map_coordinates(image[ch, :, :], indices, order=1).reshape(dim[1:])
            mask = map_coordinates(mask, indices, order=1).reshape(dim[1:])
            image, mask = torch.Tensor(image), torch.Tensor(mask)
            if weight is None:
                return image, mask
            weight = weight.view(*dim[1:]).numpy()
            weight = map_coordinates(weight, indices, order=1)
            weight = weight.reshape(dim)
            weight = torch.Tensor(weight)

        return (image, mask) if weight is None else (image, mask, weight)

class GaussianNoise:
    """Apply Gaussian noise to tensor."""

    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        noise = 0
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
