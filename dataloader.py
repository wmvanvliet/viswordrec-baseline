"""
PyTorch dataloader for the TFRecord format.
"""
from io import BytesIO
import os.path as op
import struct
from glob import glob

import pandas as pd
import example_pb2
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import IterableDataset
import webdataset as wds
import numpy as np


class TFRecord(VisionDataset):
    """Reads datasets in the form of tfrecord

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        label_offset (int): offset for 'int' style labels
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 label_offset=0):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self.label_offset = label_offset
        base_fname = 'train' if train else 'test'
        self.file = open(op.join(root, f'{base_fname}.tfrecord'), 'rb')
        self.file_index = np.loadtxt(op.join(root, f'{base_fname}.index'), dtype=np.int64)[:, 0]
        self.meta = pd.read_csv(op.join(root, f'{base_fname}.csv'), index_col=0)
        self.classes = self.meta.groupby('label').agg('first')['text']
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.file.seek(self.file_index[index])

        length_bytes = bytearray(8)
        crc_bytes = bytearray(4)
        datum_bytes = bytearray(1024 * 1024)

        if self.file.readinto(length_bytes) != 8:
            raise RuntimeError("Failed to read the record size.")
        if self.file.readinto(crc_bytes) != 4:
            raise RuntimeError("Failed to read the start token.")
        length, = struct.unpack("<Q", length_bytes)
        if length > len(datum_bytes):
            datum_bytes = datum_bytes.zfill(int(length * 1.5))
        datum_bytes_view = memoryview(datum_bytes)[:length]
        if self.file.readinto(datum_bytes_view) != length:
            raise RuntimeError("Failed to read the record.")
        if self.file.readinto(crc_bytes) != 4:
            raise RuntimeError("Failed to read the end token.")

        example = example_pb2.Example()
        example.ParseFromString(datum_bytes_view)

        features = {}
        for key in ['image/encoded', 'image/class/label']:
            field = example.features.feature[key].ListFields()[0]
            inferred_typename, value = field[0].name, field[1].value

            # Decode raw bytes into respective data types
            if inferred_typename == "bytes_list":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif inferred_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif inferred_typename == "int64_list":
                value = np.array(value, dtype=np.int32)
            features[key] = value

        img = Image.open(BytesIO(features['image/encoded'])).convert('RGB')
        target = int(features['image/class/label'][0]) + self.label_offset

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.file_index)


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
        self.dataset = iter(wds.WebDataset(glob(f'{root}/{base_fname}/*.tar'), shardshuffle=True).shuffle(1000).decode('pil').to_tuple('jpeg', 'cls'))
        if labels not in ['int', 'vector']:
            raise ValueError(f'Invalid label type {self.label}, needs to be either "int" or "vector"')

    def __next__(self):
        img, target = next(self.dataset)

        if self.labels == 'vector':
            vec = self.vectors[target]

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)

        if self.labels == 'int':
            return img, target
        elif self.labels == 'vector':
            return img, (target, vec)

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
