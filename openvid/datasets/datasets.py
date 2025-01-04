import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchcodec.decoders import SimpleVideoDecoder

from . import video_transforms
from .utils import center_crop_arr

import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import ipdb


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
        torchcodec=False,
    ):
        video_samples = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        for vid in csv_list[1:]:  # no csv head
            vid_name = vid[0]
            vid_path = os.path.join(root, vid_name)
            vid_caption = vid[1]
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])
        self.samples = video_samples

        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval

        if torchcodec:
            print("Using `torchcodec` to read videos")
            temporal_sample = video_transforms.TemporalSlice(
                n_frames=num_frames,
                frame_interval=frame_interval,
            )
        else:
            print("Using `torchvision.io` to read videos")
            temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        
        self.temporal_sample = temporal_sample
        self.root = root
        self.torchcodec = torchcodec

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                if self.torchcodec:
                    decoder = SimpleVideoDecoder(
                        source=path,
                        dimension_order="NCHW",
                    )
                    total_frames = decoder.metadata.num_frames
                    
                    video = decoder[self.temporal_sample(total_frames)]
                else:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                    total_frames = len(vframes)
                    # Sampling video frames
                    start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)

                    # print(f"start_frame_ind: {start_frame_ind}; end_frame_ind: {end_frame_ind}; total_frames:{total_frames}")
                    if not (end_frame_ind - start_frame_ind >= self.num_frames):
                        print(f"{path} with index {index} has not enough frames.")
                        return
                    frame_indice = list(np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int))
                    video = vframes[frame_indice]
            else:
                return
            
            if len(video) != self.num_frames:
                print(f"got {len(video)} frames instead of {self.num_frames}.")
                print(f"Tensor shape: {video.shape}")
                return
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # print(f"Frames tenzor size: {video.shape}")
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    # def __getitem__(self, index):
    #     for _ in range(10):
    #         try:
    #             return self.getitem(index)
    #         except Exception as e:
    #             print(e)
    #             index = np.random.randint(len(self))
    #     raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_path = ''
    root=''
    dataset = DatasetFromCSV(
        data_path,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
        root=root,
    )
    sampler = DistributedSampler(
    dataset,
    num_replicas=1,
    rank=0,
    shuffle=True,
    seed=1
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    for video_data in loader:
        print(video_data)