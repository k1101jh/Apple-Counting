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
from ultralytics_clone.trackers.byte_tracker import BYTETracker
from ultralytics_clone.trackers.my_tracker import MyTracker
from ultralytics_clone.trackers.bot_sort import BOTSORT
from utils import compute_color_for_labels
from utils import display_count
from utils import draw_trajectory
from utils import draw_bbox
from utils import xywh_to_xyxy


def counting(
    model_path,
    tracker,
    video_path,
    result_path,
    count_thres,
    resized_height=1000,
    plot_lost_tracker=False,
    display=False,
    save=False,
    save_counted_tracks_only=True,
):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_ratio = resized_height / height
    resized_width = round(width * resize_ratio)

    count_thres_width = resized_width * count_thres

    # Store the track history
    resized_tracks_history = defaultdict(lambda: [])
    tracks_data = defaultdict(lambda: {"start_frame": 0, "detections": [], "history": []})
    counted_track_ids = set()

    if save:
        vid_filename = os.path.split(video_path)[1]
        os.makedirs(result_path, exist_ok=True)
        vid_save_path = os.path.join(result_path, vid_filename)
        vid_writer = cv2.VideoWriter(
            vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (resized_width, resized_height)
        )

    frame_id = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames

            # results = model.track(frame, persist=True, tracker="configs/bytetrack.yaml", conf=0.1, iou=0.4)
            results = model.predict(frame, conf=0.05, iou=0.7)
            # results[0].boxes = tracker.update(results[0].boxes.cpu())
            det = results[0].boxes.cpu().numpy()

            resized_frame = cv2.resize(frame, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

            # Get the boxes and track IDs
            boxes = []
            boxes_xyxy = []
            track_ids = []

            # Draw detected boxes
            for box in results[0].boxes.xyxy.cpu():
                draw_bbox(resized_frame, box, resize_ratio, (0, 0, 255))

            if len(det) > 0:
                # botsort가 아니면 두 번째 인자인 이미지는 의미 없음
                tracks, _ = tracker.update(results[0].boxes.cpu(), model.predictor.batch[1])
                if len(tracks) > 0:
                    idx = tracks[:, -1].astype(int)
                    results[0] = results[0][idx]
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu().tolist()
                    boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

            if plot_lost_tracker:
                for lost_strack in tracker.lost_stracks:
                    boxes.append(lost_strack.mean[:4])
                    boxes_xyxy.append(None)
                    track_ids.append(lost_strack.track_id)

            # Visualize the results on the frame
            # annotated_frame = results[0].plot()

            # Plot the tracks
            for box, box_xyxy, track_id in zip(boxes, boxes_xyxy, track_ids):
                x, y, w, h = box

                # 원본 트랙 저장
                tracks_data[track_id]["history"].append((float(x), float(y)))

                # resize된 트랙 저장
                track = resized_tracks_history[track_id]
                track.append((float(x * resize_ratio), float(y * resize_ratio)))  # x, y center point

                if len(track) == 1:
                    tracks_data[track_id]["start_frame"] = frame_id

                color = compute_color_for_labels(track_id)

                # box 저장
                tracks_data[track_id]["detections"].append(
                    {
                        "frame_id": frame_id,
                        "box_xyxy": box_xyxy,
                    }
                )

                # Draw the box
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
        else:
            # Break the loop if the end of the video is reached
            break

    print(f"Num Trackers: {len(resized_tracks_history)}")
    print(f"Num Apples: {len(counted_track_ids)}")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    if save:
        txt_save_path = os.path.join(result_path, os.path.splitext(vid_filename)[0] + ".txt")
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

            with open(
                os.path.join(result_path, os.path.splitext(vid_filename)[0] + "_counted_tracks_info.json"), "w"
            ) as f:
                json.dump(dict_to_save, f, indent=4)

    return len(resized_tracks_history), len(counted_track_ids)


def parse_opt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # counting(**vars(opt))

    task = "tracking"
    tracker_name = "MyTracker"
    count_thres = 1 / 4
    resized_height = 1000

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

    # tracker_config = OmegaConf.load(tracker_config_path)
    # tracker = tracker_constructors[tracker_name](tracker_config)

    # counting(
    #     model_path=r"./detection_checkpoints/yolov8m_RDA_800/weights/best.pt",
    #     tracker=tracker,
    #     # video_path=r"D:\DeepLearning\Dataset\RDA apple data\2023-08-16\7\230816-Cam1-Line07-L.mp4",
    #     video_path=r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\7\231006-Cam1-Line07-L.mp4",
    #     result_path=r"runs/RDA apple data_1000",
    #     count_thres=count_thres,
    #     resized_height=resized_height,
    #     plot_lost_tracker=True,
    #     display=True,
    #     save=False,
    # )

    # vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-07-26\*\*R.mp4")
    # vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-08-16\*\*R.mp4")
    # vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\*\*R.mp4")
    vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\*\*\*[LR].mp4")

    model_path = "detection_checkpoints/yolov8m_RDA_640/weights/best.pt"
    result_path = f"runs/{task}/{tracker_name}/RDA_640_iou_0.7"
    counting_results = {}
    tracker_config = OmegaConf.load(tracker_config_path)

    for vid_file_path in tqdm(vid_file_list, desc="vids", position=0, leave=True):
        tracker = tracker_constructors[tracker_name](tracker_config)

        num_tracks, num_apples = counting(
            model_path=model_path,
            tracker=tracker,
            video_path=vid_file_path,
            result_path=result_path,
            count_thres=count_thres,
            resized_height=resized_height,
            plot_lost_tracker=True,
            display=False,
            save=True,
            save_counted_tracks_only=True,
        )
        counting_results[os.path.basename(vid_file_path)] = {"num_tracks": num_tracks, "num_apples": num_apples}

    with open(os.path.join(result_path, "counting_result.json"), "a") as f:
        json.dump(counting_results, f, indent=4)

    shutil.copy(tracker_config_path, os.path.join(result_path, os.path.basename(tracker_config_path)))
