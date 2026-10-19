import os
from torchvision import datasets as tvd
from data.texts import CIFAR100_CLASSES, IMAGENET100_WNIDS

class _ImageNet100Folder(tvd.ImageFolder):

    def find_classes(self, directory):
        present = {d.name for d in os.scandir(directory) if d.is_dir()}
        missing = [w for w in IMAGENET100_WNIDS if w not in present]
        if missing:
            raise FileNotFoundError(f'{len(missing)} ImageNet-100 class folders missing under {directory}, e.g. {missing[:3]}')
        classes = sorted(IMAGENET100_WNIDS)
        return (classes, {c: i for (i, c) in enumerate(classes)})

def load_images(name, data_root, imagenet_val):
    if name == 'cifar100':
        if data_root is None:
            raise ValueError('CIFAR-100 representations are missing: pass --data_root to generate them')
        ds = tvd.CIFAR100(root=data_root, train=False, download=True)
        if [c.replace('_', ' ') for c in ds.classes] != CIFAR100_CLASSES:
            raise RuntimeError('CIFAR-100 class order differs from class_texts.CIFAR100_CLASSES')
        return ds
    if name == 'imagenet100':
        if imagenet_val is None:
            raise ValueError('ImageNet-100 representations are missing: pass --imagenet_val to generate them')
        return _ImageNet100Folder(imagenet_val)
    raise ValueError(name)
