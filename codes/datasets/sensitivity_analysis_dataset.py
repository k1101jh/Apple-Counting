import os
import glob
import pandas
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Boxes:
    def __init__(self, boxes, conf, cls, track_id):
        self.xyxy = boxes
        self.conf = conf
        self.cls = cls
        self.id = track_id
        self.size = len(self.xyxy)

    def sampling(self, sample_ratio):
        num_samples = self.size * sample_ratio

        random_idxes = sorted(random.sample(range(self.size), num_samples))

        boxes = []
        conf = []
        cls = []
        track_id = []
        for random_idx in random_idxes:
            boxes.append(self.xyxy[random_idx])
            conf.append(self.conf[random_idx])
            cls.append(self.cls[random_idx])
            track_id.append(self.id[random_idx])

        return Boxes(torch.stack(boxes, dim=0), torch.tensor(conf), torch.tensor(cls), torch.tensor(track_id))

        # boxes, conf, cls, track_id = zip(
        #     *random.sample(list(zip(self.xyxy, self.conf, self.cls, self.id)), num_samples)
        # )

        # return Boxes(torch.stack(boxes, dim=0), torch.tensor(conf), torch.tensor(cls), torch.tensor(track_id))


class SensitivityAnalysisDataset(Dataset):
    def __init__(self, dataset_path, vid_dir, detection_rate=1, fps=30):
        self.width = 1080
        self.height = 1920
        self.detection_rate = detection_rate
        self.fps = fps
        self.vid_dir = vid_dir

        base_fps = 30
        interval = base_fps // fps
        # fps에 따라서 이미지와 gt 건너뛰는 코드 작성하기

        self.img_filelist = glob.glob(os.path.join(dataset_path, vid_dir, "img1/*"))
        self.img_filelist.sort()

        self.gt_filepath = os.path.join(os.path.join(dataset_path, vid_dir, "gt/gt.txt"))

        with open(self.gt_filepath, "r") as f:
            gt_lines = f.readlines()

        self.frame_datas = []
        frame = "1"
        boxes_xyxy = []
        track_ids = []
        for line in gt_lines:
            line = line.strip()
            vals = line.split(",")

            if vals[0] != frame:
                bbox_data = Boxes(
                    boxes=torch.stack(boxes_xyxy, dim=0),
                    conf=torch.ones(len(boxes_xyxy)),
                    cls=torch.zeros(len(boxes_xyxy)),
                    track_id=torch.tensor(track_ids),
                )
                self.frame_datas.append(bbox_data)
                boxes_xyxy = []
                track_ids = []
                frame = vals[0]

            track_ids.append(int(vals[1]))
            box = [float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])]
            box_xyxy = torch.tensor([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            boxes_xyxy.append(box_xyxy)

        bbox_data = Boxes(
            boxes=torch.stack(boxes_xyxy, dim=0),
            conf=torch.ones(len(boxes_xyxy)),
            cls=torch.zeros(len(boxes_xyxy)),
            track_id=torch.tensor(track_ids),
        )
        self.frame_datas.append(bbox_data)

        self.num_images = len(self.img_filelist)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        image = ImageOps.exif_transpose(image)
        bbox_data = self.frame_datas[idx]
        sampled_bbox_data = bbox_data.sampling(self.detection_rate)

        return [image, bbox_data, sampled_bbox_data, self.img_filelist[idx]]
