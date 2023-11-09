import os
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm

from utils import compute_color_for_labels
from utils import draw_trajectory
from utils import draw_bbox
from utils import display_count


def visualize_counted_tracks(vid_path, tracks_info_path, result_path, count_thres, resized_height):
    capture = cv2.VideoCapture(vid_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_ratio = resized_height / h
    resized_width = round(w * resize_ratio)
    count_thres_width = resized_width * count_thres

    vid_writer = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (resized_width, resized_height))

    with open(tracks_info_path, "r") as f:
        track_info = json.load(f)

    counted_tracks_history = track_info["history"]
    start_frames = track_info["start_frames"]
    detections = track_info["detections"]
    track_history_frame_idx = {}

    print(f"result_path: {result_path}")

    exist_track_ids = []
    counted_track_ids = set()
    frame_id = 0
    while capture.isOpened():
        print(f"processing frame num: {frame_id}", end="\r")

        ret, frame = capture.read()
        if ret:
            resized_frame = cv2.resize(frame, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

            for started_track_id in start_frames[frame_id]:
                exist_track_ids.append(str(started_track_id))
                track_history_frame_idx[str(started_track_id)] = 1

            for box_info in detections[frame_id]:
                track_id = box_info["track_id"]
                box_xyxy = box_info["box_xyxy"]

                # counted track에 해당하는 bbox만 표시
                color = compute_color_for_labels(track_id)

                # Draw the box
                if box_xyxy:
                    draw_bbox(resized_frame, box_xyxy, resize_ratio, color)

            # activated track 표시
            terminated_track_list_idxes = []
            for i, exist_track_id in enumerate(exist_track_ids):
                if exist_track_id in counted_tracks_history:
                    frame_idx = track_history_frame_idx[exist_track_id]

                    # frame 번호가 track 길이보다 긴지 확인하기
                    if frame_idx > len(counted_tracks_history[exist_track_id]):
                        terminated_track_list_idxes.append(i)
                        continue

                    # track resize
                    counted_tracks_history[exist_track_id][frame_idx - 1] = (
                        float(counted_tracks_history[exist_track_id][frame_idx - 1][0] * resize_ratio),
                        float(counted_tracks_history[exist_track_id][frame_idx - 1][1] * resize_ratio),
                    )

                    # 현재 프레임까지만 history 표시
                    track_history = counted_tracks_history[exist_track_id][:frame_idx]

                    # track이 화면 벗어나면 지우기
                    if track_history[-1][0] > resized_width or track_history[-1][0] < 0:
                        terminated_track_list_idxes.append(i)
                        continue

                    # track count하기
                    if exist_track_id not in counted_track_ids:
                        if abs(track_history[-1][0] - track_history[0][0]) > count_thres_width:
                            counted_track_ids.add(exist_track_id)

                    draw_trajectory(resized_frame, track_history, int(exist_track_id))

                    track_history_frame_idx[exist_track_id] += 1
                else:
                    terminated_track_list_idxes.append(i)

            num_deleted = 0
            for track_idx in terminated_track_list_idxes:
                del exist_track_ids[track_idx - num_deleted]
                num_deleted += 1

            display_count(resized_frame, f"Num Apples: {len(counted_track_ids)}", (10, 100))

            vid_writer.write(resized_frame)
            frame_id += 1
        else:
            break

    print("\nvid saved!")

    capture.release()


if __name__ == "__main__":
    source_dir = "runs/tracking/MyTracker/RDA_640"

    source_vid_dir = r"D:/DeepLearning/Dataset/RDA apple data/*"

    filenames = [
        "230726-Cam1-Line07-L.mp4",
        "230726-Cam1-Line10-L.mp4",
        "230726-Cam1-Line11-L.mp4",
        "230726-Cam1-Line14-L.mp4",
        "230726-Cam1-Line15-L.mp4",
        "230726-Cam1-Line18-L.mp4",
        "230816-Cam1-Line07-L.mp4",
        "230816-Cam1-Line10-L.mp4",
        "230816-Cam1-Line11-L.mp4",
        "230816-Cam1-Line14-L.mp4",
        "230816-Cam1-Line15-L.mp4",
        "230816-Cam1-Line18-L.mp4",
        "231006-Cam1-Line07-L.mp4",
        "231006-Cam1-Line11-L.mp4",
        "231006-Cam1-Line15-L.mp4",
    ]

    count_thres = 1 / 4
    resized_height = 1000

    for filename in filenames:
        source_vid_path = glob.glob(source_vid_dir + "/*/" + filename)[0]
        track_info_path = os.path.join(source_dir, os.path.splitext(filename)[0] + "_counted_tracks_info.json")

        result_path = os.path.join(source_dir, os.path.splitext(filename)[0] + "_counted_tracks.mp4")

        visualize_counted_tracks(source_vid_path, track_info_path, result_path, count_thres, resized_height)
