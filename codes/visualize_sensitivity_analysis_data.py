import os
import json
import cv2
import numpy as np
import logging
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate

from datasets.sensitivity_analysis_dataset import SensitivityAnalysisDataset
from utils import draw_bbox


OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


@hydra.main(version_base=None, config_path="../configs", config_name="sensitivity_analysis")
def visualize_sensitivity_analysis_data(cfg: DictConfig):
    config_yaml = OmegaConf.to_yaml(cfg)
    print(config_yaml)

    vid_dir = "1"
    output_dir = cfg.visualize_data.result_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset = SensitivityAnalysisDataset(
        cfg.dataset_path,
        vid_dir,
        detection_thres=cfg.visualize_data.detection_rate,
    )

    # 시각화 결과 크기 조정
    # resized_height == NULL이면 크기 변경 없음
    resized_height = cfg.visualize_data.resized_height
    resize_ratio = 1.0
    if resized_height:
        resize_ratio = resized_height / dataset.height

    thickness = cfg.visualize_data.thickness

    for frame, bbox_data, sampled_bbox_data, filename in tqdm(dataset, desc="frame", position=0, leave=True):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        for box_xyxy, conf in zip(bbox_data.xyxy, bbox_data.conf):
            if conf >= cfg.visualize_data.plot_bbox_conf_thres:
                draw_bbox(frame, box_xyxy, resize_ratio, (0, 0, 255), thickness)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), frame)


if __name__ == "__main__":
    visualize_sensitivity_analysis_data()
