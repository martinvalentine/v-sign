import os
import cv2
import pdb
import glob
import time
import torch
import warnings

# Ignore future warnings from libraries
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch.utils.data as data
from vsign.utils import video_augmentation

global kernel_sizes  # will store per-layer kernel/stride config

class BaseFeeder(data.Dataset):
    def __init__(
        self,
        prefix,
        gloss_dict,
        dataset='VSL_V0',
        mode="train",
        use_transform=True,
        frame_interval=1,
        image_scale=1.0,
        kernel_size=1,
        input_size=256
    ):
        # Initialize basic attributes
        # Assert that the mode is valid
        self.mode = mode.lower()
        assert self.mode in ["train", "test", "dev"], f"Invalid mode: {self.mode}"

        # Assert that use_transform is a boolean
        self.use_transform = use_transform
        assert isinstance(use_transform, bool), f"use_transform must be bool, got {type(use_transform)}"

        self.prefix = prefix
        self.dict = gloss_dict       # mapping from gloss to index
        self.dataset = dataset
        self.input_size = input_size

        # Set up global kernel_sizes used in collate_fn
        global kernel_sizes
        kernel_sizes = kernel_size

        # Frame sampling & resize params (not used by read_features)
        self.frame_interval = frame_interval
        self.image_scale = image_scale

        # Construct feature prefix for video files
        self.feat_prefix = f"{prefix}/256x256px/{dataset}/{mode}"

        # Load metadata: list of dicts with file paths and labels
        self.inputs_list = np.load(
            f"./data/processed/{dataset}/{mode}_info.npy",
            allow_pickle=True
        ).item()
        print(f"{mode} set, {len(self)} samples loaded")

        # Build transform function (Compose of several video transforms)
        self.data_aug = self.transform()
        print(f"{self.mode.capitalize()} transform loaded with frame interval = {self.frame_interval}, scale = {self.image_scale}")

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        # Fetch a single sample by index and apply normalization
        video, label, info = self.read_video(idx)
        video, label = self.normalize(video, label)
        print(f"Fetching item #{idx}: video len = {len(video)}, label = {label}")
        return video, torch.LongTensor(label), info

    def read_video(self, index):
        # Load frames from disk based on dataset version
        fi = self.inputs_list[index]
        # Build folder path for frames
        if 'VSL_V0' in self.dataset:
            img_folder = os.path.join(self.prefix, fi['folder'])

        # Gather sorted frame file paths, sample by frame_interval
        all_imgs = sorted(glob.glob(os.path.join(img_folder)))
        start = int(torch.randint(0, self.frame_interval, [1]))
        img_list = all_imgs[start::self.frame_interval]

        # Build label list: map each gloss token to its index
        label_list = []
        for gloss in fi['label'].split():
            if gloss in self.dict:
                label_list.append(self.dict[gloss][0])

        # Read & convert frames to RGB numpy arrays
        frames = [
            cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            for p in img_list
        ]
        print(f"Loaded {len(frames)} frames from {img_folder}")
        return frames, label_list, fi

    def read_features(self, index):
        # Load pre-extracted feature .npy files
        fi = self.inputs_list[index]
        data = np.load(
            f"./features/{self.mode}/{fi['fileid']}_features.npy",
            allow_pickle=True
        ).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        # Apply augmentation pipeline and mapping 8-bit RGB frames (0…255) into the range [–1, +1]
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1.0
        print(f"Video tensor shape after normalization: {video.shape}, min = {video.min()}, max = {video.max()}")
        return video, label

    def transform(self):
        if self.mode == "train" and self.use_transform:
            print("Apply training transform with augmentation")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        elif self.mode in ["test", "dev"] and self.use_transform:
            print("Apply testing/dev transform with augmentation")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])
        else:
            if self.mode == "train":
                print("Apply training transform without augmentation")
            else:
                print("Apply testing/dev transform without augmentation")
            return video_augmentation.Compose([
                video_augmentation.Resize(self.input_size),
                video_augmentation.ToTensor(),
            ])

    @staticmethod
    def collate_fn(batch):
        # 1) Sort samples by descending length for pack_padded_sequence
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        videos, labels, infos = zip(*batch)

        # 2) Compute padding parameters from global kernel_sizes
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes
        for ks in kernel_sizes:
            typ, size = ks[0], int(ks[1])
            if typ == 'K':  # convolution kernel
                left_pad = left_pad * last_stride + (size - 1) // 2
            elif typ == 'P':  # pooling stride
                last_stride = size
                total_stride *= size

        # 3a) If input is frames: pad both sides to align for conv/pool
        if videos[0].ndim > 2:
            max_len = len(videos[0])
            # compute downsampled lengths per sample
            video_length = torch.LongTensor([
                int(np.ceil(len(v) / total_stride)) * total_stride + 2 * left_pad
                for v in videos
            ])
            # compute how many frames to pad at end
            right_pad = (int(np.ceil(max_len / total_stride)) * total_stride
                         - max_len + left_pad)
            total_len = max_len + left_pad + right_pad

            padded = []
            for v in videos:
                # prepend first frame left_pad times, append last frame right_pad times
                front = v[0].unsqueeze(0).expand(left_pad, -1, -1, -1)
                back = v[-1].unsqueeze(0).expand(
                    total_len - len(v) - left_pad,
                    -1, -1, -1
                )
                padded.append(torch.cat([front, v, back], dim=0))
            video_tensor = torch.stack(padded)
        else:
            # 3b) If input is feature vectors: pad only at end
            max_len = len(videos[0])
            video_length = torch.LongTensor([len(v) for v in videos])
            padded = []
            for v in videos:
                back = v[-1].unsqueeze(0).expand(max_len - len(v), -1)
                padded.append(torch.cat([v, back], dim=0))
            video_tensor = torch.stack(padded).permute(0, 2, 1)

        # 4) Prepare labels for CTC: flatten and record lengths
        label_length = torch.LongTensor([len(l) for l in labels])
        if label_length.max() == 0:
            return video_tensor, video_length, [], [], infos

        flat_labels = torch.LongTensor([idx for lab in labels for idx in lab])

        print(f"Batch padded: {video_tensor.shape}, label lens = {label_length}")

        return video_tensor, video_length, flat_labels, label_length, infos

    # Time logging helpers
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        elapsed = time.time() - self.cur_time
        self.record_time()
        return elapsed

if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
