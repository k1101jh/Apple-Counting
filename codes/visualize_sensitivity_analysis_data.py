import os
import json
import cv2
import numpy as np
from tqdm import tqdm

from datasets.sensitivity_analysis_dataset import SensitivityAnalysisDataset
from utils import draw_bbox


if __name__ == "__main__":
    sensitivity_analysis_data_path = r"D:\DeepLearning\Dataset\Apple\SensitivityAnalysis"
    vid_dir = "1"
    output_dir = f"runs/visualze_data/SensitivityAnalysis_efficientdet/{vid_dir}"
    os.makedirs(output_dir, exist_ok=True)

    detection_rate = 1
    plot_bbox_conf_thres = 0.35
    resize_ratio = 1
    thickness = 2

    dataset = SensitivityAnalysisDataset(sensitivity_analysis_data_path, vid_dir, detection_thres=detection_rate)
    for frame, bbox_data, sampled_bbox_data, filename in tqdm(dataset, desc="frame", position=0, leave=True):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        for box_xyxy, conf in zip(bbox_data.xyxy, bbox_data.conf):
            if conf >= plot_bbox_conf_thres:
                draw_bbox(frame, box_xyxy, resize_ratio, (0, 0, 255), thickness)

        # frame = cv2.cvtColor()
        # cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), frame)
