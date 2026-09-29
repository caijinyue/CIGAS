# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Optional, Callable, Tuple, Any
from torch import Tensor
from torch.utils.data import Dataset

class ImageFolderWithFilename(datasets.ImageFolder):

    def __init__(self, root: str, prefix: Optional[str]=None, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, loader: Callable[[str], Any]=datasets.folder.default_loader, is_valid_file: Optional[Callable[[str], bool]]=None):
        """
        Args:
            root: 数据集根目录
            prefix: 要保留的文件名前缀，只保留以此前缀开头的文件
            transform: 图像预处理函数
            target_transform: 标签转换函数
            loader: 图像加载函数
            is_valid_file: 自定义文件验证函数
        """
        self.prefix = prefix
        if prefix is not None:
            is_valid_file = self._combine_validators(is_valid_file)
        super().__init__(root=root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)

    def _combine_validators(self, original_validator: Optional[Callable]) -> Callable:
        """组合原始验证器和前缀验证器"""

        def validator(filepath: str) -> bool:
            if original_validator and (not original_validator(filepath)):
                return False
            filename = os.path.basename(filepath)
            return filename.split('-')[0] == self.prefix
        return validator

    def __getitem__(self, index: int) -> Tuple[Tensor, int, str]:
        """获取图像、标签和文件名"""
        img, label = super().__getitem__(index)
        filename = os.path.basename(self.samples[index][0])
        return (img, label, filename)

def build_dataset(is_train, args, prefix=None):
    transform = build_transform(is_train, args)
    if 'SMDG' in is_train:
        root = os.path.join(args.data_path, is_train, 'test')
    elif 'REFUGE2' in is_train:
        root = os.path.join(args.data_path, is_train, 'test')
    else:
        root = os.path.join(args.data_path, is_train)
    print('dataset root:', root)
    dataset = ImageFolderWithFilename(root, transform=transform, prefix=prefix)
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train == 'train':
        transform = create_transform(input_size=args.input_size, is_training=True, color_jitter=args.color_jitter, auto_augment=args.aa, interpolation='bicubic', re_prob=args.reprob, re_mode=args.remode, re_count=args.recount, mean=mean, std=std)
        return transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def _normalize_col_name(col):
    """Normalize column name: replace non-breaking spaces, collapse multiple spaces, strip."""
    import re
    col = col.replace('\xa0', ' ')
    col = re.sub('\\s+', ' ', col)
    return col.strip()

def build_multi_label_dataset_gon_ngon(mode, args, prefix=None):
    """
    Build multi-label dataset by merging GON + NGON CSV/xlsx files.
    Reads from label_multilabel_GON-NGON/ directory.
    Normalizes column names and aligns to 12 unified label columns.
    """
    transform = build_transform(mode == 'train', args)
    label_dir = os.path.join(args.data_path, 'label_multilabel_GON-NGON')
    mode_file_map = {'train': ('train_GON_RY_20260105', 'train_NGON_RY_20260211'), 'val': ('val_GON_RY_20260105', 'val_NGON_RY20260211'), 'test': ('test_GON_RY_20260105', 'test_NGON_RY20260211'), 'SMDG': ('SMDG_GON_RY_20260105', 'external_NGON_RY20260211'), 'REFUGE2': ('REFUGE2_GON_RY_20260105', 'REFUGE2_NGON_RY20260211')}
    if mode not in mode_file_map:
        raise ValueError(f'Unknown mode: {mode}')
    gon_stem, ngon_stem = mode_file_map[mode]

    def _read_label_file(stem):
        for ext in ['.csv', '.xlsx', '.xls']:
            fpath = os.path.join(label_dir, stem + ext)
            if os.path.exists(fpath):
                if ext == '.csv':
                    df = pd.read_csv(fpath)
                else:
                    df = pd.read_excel(fpath)
                print(f'  Loaded {fpath} ({len(df)} rows)')
                return df
        raise FileNotFoundError(f'No label file found for: {stem} in {label_dir}')
    print(f'[GON-NGON] Loading mode={mode}')
    df_gon = _read_label_file(gon_stem)
    df_ngon = _read_label_file(ngon_stem)
    df_gon.columns = [_normalize_col_name(c) for c in df_gon.columns]
    df_ngon.columns = [_normalize_col_name(c) for c in df_ngon.columns]
    unified_label_cols = ['enlarged vertical cup-to-disc ratio', 'glaucomatous disc cupping', 'laminar-dot sign within the cup', 'ISNT rule violation', 'neuro-retinal rim thinning/notching', 'beta-zone peripapillary atrophy', 'optic disc pallor relative to surrounding retina', 'nasalization of central retinal vessels', 'glaucomatous disc haemorrhages', 'vessel bayonetting at disc margin', 'wedge-shaped RNFL defect', 'macula & background retinal lesion']
    for col in unified_label_cols:
        if col not in df_gon.columns:
            df_gon[col] = 0
    for col in unified_label_cols:
        if col not in df_ngon.columns:
            df_ngon[col] = 0
    df_gon = df_gon[['image'] + unified_label_cols].copy()
    df_ngon = df_ngon[['image'] + unified_label_cols].copy()
    df_merged = pd.concat([df_gon, df_ngon], ignore_index=True)
    print(f'[GON-NGON] Merged {mode}: GON={len(df_gon)}, NGON={len(df_ngon)}, Total={len(df_merged)}')
    img_dir_map = {'train': 'train', 'val': 'val', 'test': 'test', 'SMDG': 'external', 'REFUGE2': 'REFUGE2'}
    img_dir = os.path.join(args.data_path, img_dir_map[mode])
    dataset = MultiLabelCSVDataset(csv_path=None, img_dir=img_dir, transform=transform, label_mode='multi', df=df_merged)
    import json
    from pathlib import Path
    expected = json.loads((Path(__file__).resolve().parents[2] / 'configs' / 'labels.json').read_text())
    if dataset.label_columns != expected or dataset.num_classes != args.nb_classes:
        raise ValueError('Expected exactly the 12 ordered labels in configs/labels.json; check the annotation schema.')
    if len(dataset.img_basenames) != len(set(dataset.img_basenames)):
        raise ValueError('Duplicate image basenames make historical multilabel matching ambiguous.')
    if dataset.df['image'].map(lambda x: os.path.basename(str(x).strip())).duplicated().any():
        raise ValueError('Duplicate annotation basenames are not allowed.')
    return dataset
import glob

class MultiLabelCSVDataset(Dataset):

    def __init__(self, csv_path=None, img_dir=None, transform=None, label_mode='multi', img_col='image', supported_exts=('.jpg', '.jpeg', '.png', '.bmp'), df=None):
        """
        Args:
            csv_path (str, optional): 对于 multi-label，csv 或 xlsx 文件路径；对于 single-label，可设为 None
            img_dir (str): 图像根目录（必须提供，用于扫描所有图像）
            transform (callable, optional): 图像变换
            label_mode (str): 'multi' 或 'single'，决定标签模式
            img_col (str): 表中图像文件名列名，默认 'image'（仅 multi）
            supported_exts (tuple): 支持的图像文件扩展名
            df (pd.DataFrame, optional): Pre-built DataFrame (used by gon_ngon loader)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.label_mode = label_mode
        self.supported_exts = supported_exts
        if img_dir is None:
            raise ValueError('img_dir is required for both single-label and multi-label modes')
        if self.label_mode == 'multi':
            if df is not None:
                self.df = df
            elif csv_path is not None:
                ext = os.path.splitext(csv_path)[1].lower()
                if ext == '.csv':
                    self.df = pd.read_csv(csv_path)
                elif ext in ['.xlsx', '.xls']:
                    self.df = pd.read_excel(csv_path)
                else:
                    raise ValueError(f'Unsupported file format: {ext}')
            else:
                raise ValueError('Either csv_path or df is required for multi-label mode')
            self.img_col = img_col
            exclude_cols = {img_col, 'filename', 'path', 'img', 'id', 'index'}
            self.label_columns = [col for col in self.df.columns if col not in exclude_cols]
            self.num_classes = len(self.label_columns)
            if self.num_classes == 0:
                raise ValueError(f'No label columns found! Check your columns: {self.df.columns.tolist()}')
            self.img_to_labels = {}
            for _, row in self.df.iterrows():
                filename = os.path.basename(str(row[self.img_col]).strip())
                labels = row[self.label_columns].values.astype(np.float32)
                self.img_to_labels[filename] = labels
            print(f'[Multi-label] Loaded {len(self.img_to_labels)} positive samples from CSV (using basename matching)')
        elif self.label_mode == 'single':
            self.classes = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.num_classes = len(self.classes)
            print(f'[Single-label] Found {self.num_classes} classes: {self.classes}')
        else:
            raise ValueError(f'Unsupported label_mode: {label_mode}')
        pattern = os.path.join(img_dir, '**', '*')
        all_files = glob.glob(pattern, recursive=True)
        self.img_files = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(self.supported_exts)]
        self.img_files.sort()
        self.img_basenames = [os.path.basename(f) for f in self.img_files]
        print(f'Total images found: {len(self.img_files)}')
        if self.label_mode == 'multi':
            matched_count = sum((basename in self.img_to_labels for basename in self.img_basenames))
            print(f'[Multi-label] Successfully matched positive samples: {matched_count}')
            print(f'[Multi-label] Negative samples (no annotation in CSV): {len(self.img_files) - matched_count}')
        if len(self.img_files) == 0:
            raise ValueError(f'No images found in {img_dir} with extensions {supported_exts}')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        basename = self.img_basenames[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.label_mode == 'multi':
            labels_np = self.img_to_labels.get(basename, np.zeros(self.num_classes, dtype=np.float32))
            labels = torch.tensor(labels_np, dtype=torch.float32)
        else:
            rel_path = os.path.relpath(img_path, self.img_dir)
            cls_name = os.path.dirname(rel_path)
            label_idx = self.class_to_idx.get(cls_name, 0)
            labels = torch.tensor(label_idx, dtype=torch.long)
        filename = basename
        return (image, labels, filename)
