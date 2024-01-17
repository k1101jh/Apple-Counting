import os
import cv2
import numpy as np
import torch
import argparse
import csv
import glob
import json
import random
import logging
import hydra
import motmetrics as mm
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate
from collections import defaultdict

from ultralytics import YOLO
from ultralytics_clone.trackers.byte_tracker import BYTETracker
from ultralytics_clone.trackers.my_tracker import MyTracker
from ultralytics_clone.trackers.bot_sort import BOTSORT
from datasets.sensitivity_analysis_dataset import SensitivityAnalysisDataset
from counter import Counter
from utils import draw_trajectory
from utils import draw_bbox
from utils import compute_color_for_labels
from utils import display_count
from utils import xywh_to_xyxy
from utils import xyxy_to_xywh
from utils import xyxy_to_cxywh
from utils import xywh_to_cxywh
from utils import tlbr_to_xywh
from utils import tlbr_to_cxywh

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def sensitivity_analysis(
    tracker,
    dataset,
    interval,
    result_path,
    count_thres,
    fps,
    resized_height=1000,
    plot_lost_tracker=False,
    plot_bbox_conf_thres=0.05,
    show=False,
    save=False,
):
    # fps = dataset.fps
    width = dataset.width
    height = dataset.height

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    resize_ratio = resized_height / height
    resized_width = round(width * resize_ratio)

    counter = Counter(
        tracker=tracker,
        width=width,
        height=height,
        resized_height=resized_height,
        count_thres=count_thres,
        result_path=result_path,
        save_name=dataset.vid_dir,
        plot_lost_tracker=plot_lost_tracker,
        plot_bbox_conf_thres=plot_bbox_conf_thres,
    )

    if save:
        # video 저장
        vid_filename = dataset.vid_dir + ".mp4"
        os.makedirs(result_path, exist_ok=True)
        vid_save_path = os.path.join(result_path, vid_filename)
        vid_writer = cv2.VideoWriter(
            vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (resized_width, resized_height)
        )

        # Track 저장
        tracks_result_path = os.path.join(result_path, dataset.vid_dir + "_tracks_result.txt")
        tracks_result_file = open(tracks_result_path, "w", newline="")
        tracks_result_writer = csv.writer(tracks_result_file)

        # Counted Tracks 저장
        counted_tracks_result_path = os.path.join(result_path, dataset.vid_dir + "_counted_tracks_result.txt")
        counted_tracks_result_file = open(counted_tracks_result_path, "w", newline="")
        counted_tracks_result_writer = csv.writer(counted_tracks_result_file)

    # Loop through the video frames
    frame_id = 0

    for frame, bbox_data, sampled_bbox_data, _ in tqdm(dataset, desc="frame", position=0, leave=True):
        # fps에 따라서 건너뛸지 결정
        if frame_id % interval != 0:
            frame_id += 1
            continue

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        resized_frame, tracks_data, activated_track_ids = counter.update(frame, sampled_bbox_data)

        ## Metric 계산
        gt_xywh_points = [xyxy_to_xywh(bbox) for bbox in bbox_data.xyxy]

        # 추정값의 xywh 구하기
        # gt의 track id와 tracker의 track id 매칭하기
        hypothesis_xywh_points = []
        for track_id in activated_track_ids:
            detection_xyxy = tracks_data[track_id]["boxes"][-1]["detection_xyxy"]
            box_xywh = xyxy_to_xywh(detection_xyxy)
            hypothesis_xywh_points.append(box_xywh)

            # Tracker 결과 저장
            # frame_id, tracker_id, x, y, w, h, conf, x, y, z
            # x, y, z는 -1
            if save:
                # gt.txt에는 frame_id가 1번부터 시작
                # 이 코드에서는 0부터 시작하므로 1 더하기
                tracks_result_writer.writerow([counter.frame_id, int(track_id)] + box_xywh + [1, -1, -1, -1])

        dists = mm.distances.iou_matrix(gt_xywh_points, hypothesis_xywh_points, max_iou=0.5)
        # frameid = acc.update(bbox_data.id, activated_track_ids, dists)

        # Display the annotated frame
        if show:
            cv2.imshow("Tracking", resized_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if save:
            vid_writer.write(resized_frame)

        frame_id += 1

    num_tracks = counter.get_num_tracks()
    num_apples = counter.get_num_counted_tracks()

    print(f"Num Trackers: {num_tracks}")
    print(f"Num Apples: {num_apples}")

    # Release the video capture object and close the display window
    cv2.destroyAllWindows()

    if save:
        counter.save()

        # counted tracks만 저장
        counted_tracks_result = defaultdict(lambda: [])

        for track_id in counter.counted_track_ids:
            for boxes in tracks_data[track_id]["boxes"]:
                detection_xyxy = boxes["detection_xyxy"]
                if detection_xyxy != None:
                    counted_tracks_result[boxes["frame_id"]].append(
                        {"track_id": track_id, "box_xywh": xyxy_to_xywh(detection_xyxy)}
                    )

        for frame_id in range(1, counter.frame_id + 1):
            for track_data in counted_tracks_result[frame_id]:
                track_id = int(track_data["track_id"])
                box_xywh = track_data["box_xywh"]
                counted_tracks_result_writer.writerow([frame_id, track_id] + box_xywh + [1, -1, -1, -1])

        tracks_result_file.close()
        counted_tracks_result_file.close()

    # Metric 계산
    # mh = mm.metrics.create()
    # summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="full")

    # strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    # print(strsummary)
    # summary.to_json(os.path.join(result_path, dataset.vid_dir + "_metrics.json"))

    return num_tracks, num_apples


@hydra.main(version_base=None, config_path="../configs", config_name="sensitivity_analysis")
def sensitivity_analysis_all_vids(cfg: DictConfig):
    config_yaml = OmegaConf.to_yaml(cfg)
    print(config_yaml)

    tracker_class_name = cfg.tracker._target_[cfg.tracker._target_.rfind(".") + 1 :]

    for fps in cfg.fps_list:
        interval = 30 // fps
        for detection_rate in cfg.detection_rate_list:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

            result_dir = os.path.join(
                cfg.result_dir,
                tracker_class_name,
                f"high_{cfg.tracker.args.track_high_thresh}_low_{cfg.tracker.args.track_low_thresh}_new_{cfg.tracker.args.new_track_thresh}_dr_{detection_rate}_fps_{fps}",
            )
            os.makedirs(result_dir, exist_ok=True)
            counting_results = {}

            vid_dirs = os.listdir(cfg.dataset_path)
            for vid_dir in vid_dirs:
                tracker = instantiate(cfg.tracker, frame_rate=fps)
                dataset = SensitivityAnalysisDataset(cfg.dataset_path, vid_dir, detection_thres=detection_rate)

                num_tracks, num_apples = sensitivity_analysis(
                    tracker=tracker,
                    dataset=dataset,
                    interval=interval,
                    result_path=result_dir,
                    count_thres=cfg.count_thres,
                    fps=fps,
                    resized_height=cfg.resized_height,
                    plot_lost_tracker=True,
                    plot_bbox_conf_thres=cfg.tracker.args.track_low_thresh,
                    show=False,
                    save=True,
                )

                counting_results[vid_dir] = {"num_tracks": num_tracks, "num_apples": num_apples}

                with open(os.path.join(result_dir, "counting_result.json"), "a") as f:
                    json.dump(counting_results, f, indent=4)


if __name__ == "__main__":
    sensitivity_analysis_all_vids()
