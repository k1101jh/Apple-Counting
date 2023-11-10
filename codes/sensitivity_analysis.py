import os
import cv2
import numpy as np
import torch
import argparse
import csv
import glob
import json
import random
import motmetrics as mm
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

from ultralytics import YOLO
from ultralytics_clone.trackers.byte_tracker import BYTETracker
from ultralytics_clone.trackers.my_tracker import MyTracker
from ultralytics_clone.trackers.bot_sort import BOTSORT
from datasets.sensitivity_analysis_dataset import SensitivityAnalysisDataset
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


def sensitivity_analysis(
    tracker,
    dataset,
    result_path,
    count_thres,
    resized_height=1000,
    plot_lost_tracker=False,
    display=False,
    save=False,
    save_counted_tracks_only=True,
):
    fps = dataset.fps
    width = dataset.width
    height = dataset.height

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    resize_ratio = resized_height / height
    resized_width = round(width * resize_ratio)

    count_thres_width = resized_width * count_thres

    # Store the track history
    resized_tracks_history = defaultdict(lambda: [])
    tracks_data = defaultdict(lambda: {"start_frame": 0, "detections": [], "history": []})
    counted_track_ids = set()

    if save:
        # video 저장
        vid_filename = dataset.vid_dir + ".mp4"
        os.makedirs(result_path, exist_ok=True)
        vid_save_path = os.path.join(result_path, vid_filename)
        vid_writer = cv2.VideoWriter(
            vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (resized_width, resized_height)
        )

        # Tracker 결과 저장
        tracker_result_path = os.path.join(result_path, dataset.vid_dir + "_tracker_result.txt")
        tracker_result_file = open(tracker_result_path, "w", newline="")
        tracker_result_writer = csv.writer(tracker_result_file)

    # Loop through the video frames
    frame_id = 0

    for frame, bbox_data, sampled_bbox_data, _ in tqdm(dataset, desc="frame", position=0, leave=True):
        # Run YOLOv8 tracking on the frame, persisting tracks between frames

        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        resized_frame = cv2.resize(frame, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

        # Get the boxes and track IDs
        boxes = []
        boxes_xyxy = []
        track_ids = []
        gt_id_track_id_dict = {}

        # Draw detected boxes
        for box in sampled_bbox_data.xyxy:
            draw_bbox(resized_frame, box, resize_ratio, (0, 0, 255))

        if sampled_bbox_data.size > 0:
            tracks, gt_id_track_id_dict = tracker.update(sampled_bbox_data)

            for track in tracks:
                boxes.append(track[:4].tolist())
                boxes_xyxy.append(track[:4].tolist())
                track_ids.append(track[4])

        if plot_lost_tracker:
            for lost_strack in tracker.lost_stracks:
                boxes.append(lost_strack.tlbr.tolist())
                boxes_xyxy.append(None)
                track_ids.append(lost_strack.track_id)

        # Plot the tracks
        for box, box_xyxy, track_id in zip(boxes, boxes_xyxy, track_ids):
            cxywh = tlbr_to_cxywh(box)
            cx, cy, w, h = cxywh
            track_id = int(track_id)

            # 원본 트랙 저장
            tracks_data[track_id]["history"].append((float(cx), float(cy)))

            # resize된 트랙 저장
            track = resized_tracks_history[track_id]
            track.append((float(cx * resize_ratio), float(cy * resize_ratio)))  # x, y center point

            if len(track) == 1:
                tracks_data[track_id]["start_frame"] = frame_id

            color = compute_color_for_labels(track_id)

            # box 저장
            tracks_data[track_id]["detections"].append(
                {
                    "frame_id": frame_id,
                    "box_xywh": tlbr_to_xywh(box),
                    "box_xyxy": box_xyxy,
                }
            )

            # Draw the box
            # print(box_xyxy)
            if box_xyxy:
                draw_bbox(resized_frame, box_xyxy, resize_ratio, color)

        # activated track 표시
        for tracked_track in [x for x in tracker.tracked_stracks if x.is_activated]:
            track_id = tracked_track.track_id
            # Draw the tracking lines
            track_history = resized_tracks_history[track_id]
            # track의 길이로 count할지 결정
            if track_id not in counted_track_ids:
                if abs(track_history[-1][0] - track_history[0][0]) > count_thres_width:
                    counted_track_ids.add(track_id)

            draw_trajectory(resized_frame, track_history, track_id)

        # lost track 표시
        for tracked_track in tracker.lost_stracks:
            track_id = tracked_track.track_id
            # track이 화면을 벗어나면 건너뛰기
            if tracked_track.mean[0] > width or tracked_track.mean[0] < 0:
                continue
            # Draw the tracking lines
            track_history = resized_tracks_history[track_id]
            draw_trajectory(resized_frame, track_history, track_id)

        ## Metric 계산
        gt_xywh_points = [xyxy_to_xywh(bbox) for bbox in bbox_data.xyxy]

        # 추정값의 xywh 구하기
        # gt의 track id와 tracker의 track id 매칭하기
        hypothesis_xywh_points = []
        for gt_id in sampled_bbox_data.id:
            gt_id = int(gt_id)
            # new track은 dict에 없을 것
            if gt_id in gt_id_track_id_dict:
                box_xywh = tracks_data[gt_id_track_id_dict[gt_id]]["detections"][-1]["box_xywh"]
                hypothesis_xywh_points.append(box_xywh)

                # Tracker 결과 저장
                # frame_id, tracker_id, x, y, w, h, conf, x, y, z
                # x, y, z는 -1
                if save:
                    box_xywh_str_list = [
                        f"{box_xywh[0]:.2f}",
                        f"{box_xywh[1]:.2f}",
                        f"{box_xywh[2]:.2f}",
                        f"{box_xywh[3]:.2f}",
                    ]
                    # gt.txt에는 frame_id가 1번부터 시작
                    # 이 코드에서는 0부터 시작
                    tracker_result_writer.writerow([frame_id + 1, gt_id] + box_xywh_str_list + [1, -1, -1, -1])

        dists = mm.distances.iou_matrix(gt_xywh_points, hypothesis_xywh_points)

        frameid = acc.update(bbox_data.id, list(gt_id_track_id_dict.keys()), dists)
        # print(acc.mot_events.loc[frameid])

        # Display count
        display_count(resized_frame, f"Num Trackers: {len(resized_tracks_history)}", (10, 50))
        display_count(resized_frame, f"Num Apples: {len(counted_track_ids)}", (10, 100))

        # Display the annotated frame
        if display:
            cv2.imshow("YOLOv8 Tracking", resized_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if save:
            vid_writer.write(resized_frame)

        frame_id += 1

    print(f"Num Trackers: {len(resized_tracks_history)}")
    print(f"Num Apples: {len(counted_track_ids)}")

    # Release the video capture object and close the display window
    cv2.destroyAllWindows()

    if save:
        txt_save_path = os.path.join(result_path, dataset.vid_dir + ".txt")
        with open(txt_save_path, "w") as f:
            f.write(f"Num Trackers: {len(resized_tracks_history)}\n")
            f.write(f"Num Apples: {len(counted_track_ids)}")

        # count한 궤적 정보 저장
        if save_counted_tracks_only:
            counted_tracks_history = defaultdict(lambda: [])
            start_frames = [[] for x in range(frame_id)]
            detections = [[] for x in range(frame_id)]
            for track_id in counted_track_ids:
                counted_tracks_history[track_id] = tracks_data[track_id]["history"]
                start_frames[tracks_data[track_id]["start_frame"]].append(track_id)
                for detection in tracks_data[track_id]["detections"]:
                    detections[detection["frame_id"]].append({"track_id": track_id, "box_xyxy": detection["box_xyxy"]})

            dict_to_save = {
                "history": counted_tracks_history,
                "start_frames": start_frames,
                "detections": detections,
            }

            with open(os.path.join(result_path, dataset.vid_dir + "_counted_tracks_info.json"), "w") as f:
                json.dump(dict_to_save, f, indent=4)

        tracker_result_file.close()

    # Metric 계산
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="full")

    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)
    summary.to_csv(os.path.join(result_path, os.path.splitext(vid_filename)[0] + "_summary.csv"))

    return len(resized_tracks_history), len(counted_track_ids)


def parse_opt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # counting(**vars(opt))

    random.seed(2023)
    resized_height = 1000

    sensitivity_analysis_data_path = r"D:\DeepLearning\Dataset\Apple\SensitivityAnalysis"
    tracker_name = "ByteTrack"
    count_thres = 1 / 4
    detection_rate = 1
    fps = 30

    tracker_config_pathes = {
        "ByteTrack": "configs/bytetrack.yaml",
        "BotSORT": "configs/bot_sort.yaml",
        "MyTracker": "configs/my_tracker.yaml",
    }

    tracker_constructors = {
        "ByteTrack": BYTETracker,
        "BotSORT": BOTSORT,
        "MyTracker": MyTracker,
    }

    tracker_config_path = tracker_config_pathes[tracker_name]
    tracker_config = OmegaConf.load(tracker_config_path)

    result_path = f"runs/sensitivity_analysis/{tracker_name}_dr_{detection_rate}_fps_{fps}"
    vid_dirs = os.listdir(sensitivity_analysis_data_path)
    counting_results = {}

    for vid_dir in vid_dirs:
        tracker = tracker_constructors[tracker_name](tracker_config)

        dataset = SensitivityAnalysisDataset(
            sensitivity_analysis_data_path, vid_dir, detection_rate=detection_rate, fps=fps
        )

        num_tracks, num_apples = sensitivity_analysis(
            tracker=tracker,
            dataset=dataset,
            result_path=result_path,
            count_thres=count_thres,
            resized_height=resized_height,
            plot_lost_tracker=True,
            display=False,
            save=True,
            save_counted_tracks_only=True,
        )
        counting_results[vid_dir] = {"num_tracks": num_tracks, "num_apples": num_apples}

        with open(os.path.join(result_path, "counting_result.json"), "a") as f:
            json.dump(counting_results, f, indent=4)

        break
