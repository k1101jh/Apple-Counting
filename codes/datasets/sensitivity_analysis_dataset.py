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

    def __len__(self):
        return self.size

    def sampling(self, detection_thres):
        scores = np.random.uniform(0, 1, self.size)

        boxes = []
        conf = []
        cls = []
        track_id = []

        num_detections = 0
        for idx, score in enumerate(scores):
            if score <= detection_thres:
                boxes.append(self.xyxy[idx])
                conf.append(self.conf[idx])
                cls.append(self.cls[idx])
                track_id.append(self.id[idx])
                num_detections += 1

        if num_detections > 0:
            return Boxes(torch.stack(boxes, dim=0), torch.tensor(conf), torch.tensor(cls), torch.tensor(track_id))
        else:
            return Boxes([], [], [], [])


class SensitivityAnalysisDataset(Dataset):
    def __init__(self, dataset_path, vid_dir, detection_thres=1):
        self.width = 1080
        self.height = 1920
        self.detection_thres = detection_thres
        self.vid_dir = vid_dir

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
        sampled_bbox_data = bbox_data.sampling(self.detection_thres)

        return [image, bbox_data, sampled_bbox_data, self.img_filelist[idx]]
