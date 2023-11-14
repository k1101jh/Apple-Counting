import os
import cv2
import numpy as np
import torch
import argparse
import glob
import json
import shutil
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

from ultralytics import YOLO
from utils import compute_color_for_labels
from utils import display_count
from utils import draw_trajectory
from utils import draw_bbox
from utils import xywh_to_xyxy
from utils import tlbr_to_cxywh


class Counter:
    def __init__(
        self,
        tracker,
        width,
        height,
        resized_height,
        count_thres,
        result_path,
        save_name,
        plot_lost_tracker=True,
    ):
        self.tracker = tracker
        self.width = width
        self.resized_height = resized_height
        self.result_path = result_path
        self.save_name = save_name
        self.plot_lost_tracker = plot_lost_tracker

        self.resize_ratio = resized_height / height
        self.resized_width = round(width * self.resize_ratio)

        self.count_thres_width = width * count_thres

        # Store the track history
        self.resized_tracks_history = defaultdict(lambda: [])
        self.tracks_data = defaultdict(lambda: {"start_frame": 0, "boxes": [], "history": []})
        self.counted_track_ids = set()

        self.frame_id = 0

    def update(self, frame, bbox):
        resized_frame = cv2.resize(
            frame, dsize=(self.resized_width, self.resized_height), interpolation=cv2.INTER_CUBIC
        )

        # Get the boxes and track IDs
        boxes_xyxy = []
        detections_xyxy = []
        activated_track_ids = []
        lost_track_ids = []

        # Draw detected boxes
        for box_xyxy in bbox.xyxy:
            draw_bbox(resized_frame, box_xyxy, self.resize_ratio, (0, 0, 255))

        if len(bbox) > 0:
            # botsort가 아니면 두 번째 인자인 이미지는 의미 없음
            tracks = self.tracker.update(bbox)

            for track in tracks:
                boxes_xyxy.append(track[:4].tolist())
                detections_xyxy.append(track[:4].tolist())
                activated_track_ids.append(track[4])

        if self.plot_lost_tracker:
            for lost_strack in self.tracker.lost_stracks:
                boxes_xyxy.append(lost_strack.tlbr.tolist())
                detections_xyxy.append(None)
                lost_track_ids.append(lost_strack.track_id)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # Plot the tracks
        for box_xyxy, detection_xyxy, track_id in zip(
            boxes_xyxy, detections_xyxy, activated_track_ids + lost_track_ids
        ):
            cxywh = tlbr_to_cxywh(box_xyxy)
            cx, cy, w, h = cxywh

            # 원본 트랙 저장
            self.tracks_data[track_id]["history"].append((float(cx), float(cy)))

            # resize된 트랙 저장
            track = self.resized_tracks_history[track_id]
            track.append((float(cx * self.resize_ratio), float(cy * self.resize_ratio)))  # x, y center point

            if len(track) == 1:
                self.tracks_data[track_id]["start_frame"] = self.frame_id

            # box 저장
            # detection_xyxy는 실제 detection 결과가 아니라 track의 결과!!
            self.tracks_data[track_id]["boxes"].append(
                {
                    "frame_id": self.frame_id,
                    "box_xyxy": box_xyxy,
                    "detection_xyxy": detection_xyxy,
                }
            )

            # Draw the box
            color = compute_color_for_labels(track_id)
            if detection_xyxy:
                draw_bbox(resized_frame, detection_xyxy, self.resize_ratio, color)

        # activated track 표시
        for tracked_track in [x for x in self.tracker.tracked_stracks if x.is_activated]:
            track_id = tracked_track.track_id
            # Draw the tracking lines
            track_history = self.tracks_data[track_id]["history"]
            # track의 길이로 count할지 결정
            if track_id not in self.counted_track_ids:
                if abs(track_history[-1][0] - track_history[0][0]) > self.count_thres_width:
                    self.counted_track_ids.add(track_id)

            draw_trajectory(resized_frame, self.resized_tracks_history[track_id], track_id)

        # lost track 표시
        for tracked_track in self.tracker.lost_stracks:
            track_id = tracked_track.track_id
            # track이 화면을 벗어나면 건너뛰기
            if tracked_track.mean[0] > self.width or tracked_track.mean[0] < 0:
                continue
            # Draw the tracking lines
            track_history = self.resized_tracks_history[track_id]
            draw_trajectory(resized_frame, track_history, track_id)

        # Display count
        display_count(resized_frame, f"Num Trackers: {len(self.resized_tracks_history)}", (10, 50))
        display_count(resized_frame, f"Num Apples: {len(self.counted_track_ids)}", (10, 100))

        self.frame_id += 1

        return resized_frame, self.tracks_data, activated_track_ids

    def get_num_tracks(self):
        return len(self.resized_tracks_history)

    def get_num_counted_tracks(self):
        return len(self.counted_track_ids)

    def save(self):
        # 계수 결과 저장
        txt_save_path = os.path.join(self.result_path, self.save_name + ".txt")
        with open(txt_save_path, "w") as f:
            f.write(f"Num Trackers: {len(self.resized_tracks_history)}\n")
            f.write(f"Num Apples: {len(self.counted_track_ids)}")

        # count한 궤적 정보 저장
        counted_tracks_history = defaultdict(lambda: [])
        start_frames = [[] for x in range(self.frame_id)]
        detections = [[] for x in range(self.frame_id)]
        for track_id in self.counted_track_ids:
            counted_tracks_history[track_id] = self.tracks_data[track_id]["history"]
            start_frames[self.tracks_data[track_id]["start_frame"]].append(track_id)
            for boxes in self.tracks_data[track_id]["boxes"]:
                detections[boxes["frame_id"]].append({"track_id": track_id, "detection_xyxy": boxes["detection_xyxy"]})

        dict_to_save = {
            "history": counted_tracks_history,
            "start_frames": start_frames,
            "detections": detections,
        }

        with open(os.path.join(self.result_path, self.save_name + "_counted_tracks_info.json"), "w") as f:
            json.dump(dict_to_save, f, indent=4)
