"""
PyTorch dataloaders to load the weird dataformat I'm using for my datasets. In
order to conserve memory, my datasets are pickled lists of PNG encoded images.
Reading these datasets involves using the Python Imaging Library (PIL) to
decode the PNG binary strings into PIL images.

The CombinedPickledPNGs dataloader will concatenate multiple datasets together.
This is useful is you want to train on for example both imagenet and word
datasets.
"""
import os.path as op
import pandas as pd
from glob import glob

from torch.utils.data import IterableDataset
import numpy as np
import webdataset as wds


class WebDataset(IterableDataset):
    """Reads datasets in webdataset form

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        labels ('int' | 'vector'): What kind of labels to use. Either a integer
            class label or a distributed (word2vec) vector.
        label_offset (int): offset for 'int' style labels
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, labels='int', label_offset=0):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.label_offset = label_offset
        self.labels = labels
        base_fname = 'train' if train else 'test'
        self.meta = pd.read_csv(op.join(root, f'{base_fname}.csv'), index_col=0)
        self.vectors = np.atleast_2d(np.loadtxt(op.join(root, 'vectors.csv'), delimiter=',', skiprows=1, usecols=np.arange(1, 301), encoding='utf8', dtype=np.float32, comments=None))
        self.classes = self.meta.groupby('label').agg('first')['text']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.dataset = iter(wds.WebDataset(glob(f'{root}/{base_fname}/*.tar')).shuffle(1000).decode('pil').to_tuple('png', 'cls'))

    def __next__(self):
        img, label = next(self.dataset)
        if self.labels == 'int':
            label += self.label_offset
        elif self.labels == 'vector':
            label = self.vectors[label]
        else:
            raise ValueError(f'Invalid label type {self.label}, needs to be either "int" or "vector"')

        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __iter__(self):
        """
        Yields:
            tuple: (image, target) where target is index of the target class.
        """
        return self

    def __len__(self):
        return len(self.meta)


class Combined(IterableDataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        gcd = np.gcd.reduce([len(ds) for ds in self.datasets])
        n_times = [len(ds) // gcd for ds in self.datasets]
        while True:
            for source, n in zip(sources, n_times):
                try:
                    for _ in range(n):
                        yield next(source)
                except StopIteration:
                    return
                    
    def __len__(self):
        return sum([len(ds) for ds in self.datasets])
