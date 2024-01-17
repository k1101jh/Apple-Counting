import os
import cv2
import numpy as np
import torch
import argparse
import glob
import json
import shutil
import yaml
import logging
import hydra
from time import time
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate

from ultralytics import YOLO
from ultralytics_clone.trackers.byte_tracker import BYTETracker
from ultralytics_clone.trackers.my_tracker import MyTracker
from ultralytics_clone.trackers.bot_sort import BOTSORT
from counter import Counter
from utils import compute_color_for_labels
from utils import display_count
from utils import draw_trajectory
from utils import draw_bbox
from utils import xywh_to_xyxy


OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def counting(
    model,
    tracker,
    video_path,
    result_path,
    count_thres,
    resized_height=1000,
    plot_lost_tracker=False,
    plot_bbox_conf_thres=0.05,
    show=False,
    save=False,
):
    time_list = []
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_filename = os.path.splitext(os.path.basename(video_path))[0]
    counter = Counter(
        tracker=tracker,
        width=width,
        height=height,
        resized_height=resized_height,
        count_thres=count_thres,
        result_path=result_path,
        save_name=vid_filename,
        plot_lost_tracker=plot_lost_tracker,
        plot_bbox_conf_thres=plot_bbox_conf_thres,
    )

    if save:
        os.makedirs(result_path, exist_ok=True)

        vid_save_path = os.path.join(result_path, os.path.basename(video_path))
        vid_writer = cv2.VideoWriter(
            vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (counter.resized_width, resized_height)
        )

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        start_time = time()
        success, frame = cap.read()

        if success:
            results = model.predict(frame, conf=0.05, iou=0.6)
            # results[0].boxes = tracker.update(results[0].boxes.cpu())

            resized_frame, _, _ = counter.update(frame, results[0].boxes.cpu())

            # Display the annotated frame
            if show:
                cv2.imshow("YOLOv8 Tracking", resized_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if save:
                vid_writer.write(resized_frame)
        else:
            # Break the loop if the end of the video is reached
            break
        end_time = time()
        time_list.append(end_time - start_time)

    # 프레임 처리 시간 및 FPS 출력
    print(f"average time: {np.mean(time_list)}")
    print(f"min time: {np.min(time_list)}")
    print(f"max time: {np.max(time_list)}")
    print(f"mid time: {np.median(time_list)}")
    print(f"FPS: {int(1./np.mean(time_list))}")

    num_tracks = counter.get_num_tracks()
    num_apples = counter.get_num_counted_tracks()

    print(f"Num Trackers: {num_tracks}")
    print(f"Num Apples: {num_apples}")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    if save:
        counter.save()

    return num_tracks, num_apples


@hydra.main(version_base=None, config_path="../configs", config_name="counting")
def count_all_vids(cfg: DictConfig):
    config_yaml = OmegaConf.to_yaml(cfg)
    print(config_yaml)

    tracker_class_name = cfg.tracker._target_[cfg.tracker._target_.rfind(".") + 1 :]
    result_dir = os.path.join(cfg.result_dir, tracker_class_name)
    if cfg.task_name:
        result_dir = os.path.join(result_dir, cfg.task_name)
    vid_file_list = glob.glob(cfg.vid_file_glob_path)

    counting_results = {}

    # Load the YOLOv8 model
    model = YOLO(cfg.model_path)

    for vid_file_path in tqdm(vid_file_list, desc="vids", position=0, leave=True):
        tracker = instantiate(cfg.tracker)

        num_tracks, num_apples = counting(
            model=model,
            tracker=tracker,
            video_path=vid_file_path,
            result_path=result_dir,
            count_thres=cfg.count_thres,
            resized_height=cfg.resized_height,
            plot_lost_tracker=True,
            show=False,
            save=True,
        )
        counting_results[os.path.basename(vid_file_path)] = {"num_tracks": num_tracks, "num_apples": num_apples}

    with open(os.path.join(result_dir, "counting_result.json"), "a") as f:
        json.dump(counting_results, f, indent=4)

    with open(os.path.join(result_dir, "configs.yaml"), "w") as f:
        yaml.dump(config_yaml, f)


if __name__ == "__main__":
    count_all_vids()
