"""
PyTorch dataloader for the TFRecord format.
"""
from io import BytesIO
import os.path as op
import struct

import pandas as pd
import example_pb2
from PIL import Image
from torchvision.datasets import VisionDataset
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
