import os
import cv2
import numpy as np
import torch
import argparse
import glob
import json
from omegaconf import OmegaConf
from collections import defaultdict

from ultralytics import YOLO
from ultralytics_clone.trackers.byte_tracker import BYTETracker

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def compute_color_for_labels(label):
    color = [int(int(p * (label**2 - label + 1)) % 255) for p in palette]
    color[1] = color[1] % 100
    return tuple(color)


def counting(model_path, tracker_config_path, video_path, result_path, count_thres, display=False, save=False):
    resized_height = 1000
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])
    counted_track = {}

    tracker_config = OmegaConf.load(tracker_config_path)
    tracker = BYTETracker(tracker_config)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_ratio = resized_height / height
    resized_width = round(width * resize_ratio)

    count_thres_width = resized_width * count_thres

    if save:
        vid_filename = os.path.split(video_path)[1]
        os.makedirs(result_path, exist_ok=True)
        vid_save_path = os.path.join(result_path, vid_filename)
        vid_writer = cv2.VideoWriter(
            vid_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (resized_width, resized_height)
        )

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames

            # results = model.track(frame, persist=True, tracker="configs/bytetrack.yaml", conf=0.1, iou=0.4)
            results = model.predict(frame, conf=0.05, iou=0.8)
            # results[0].boxes = tracker.update(results[0].boxes.cpu())
            det = results[0].boxes.cpu().numpy()

            resized_frame = cv2.resize(frame, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

            # Get the boxes and track IDs
            boxes = []
            boxes_xyxy = []
            track_ids = []

            # Draw detected boxes
            for box in results[0].boxes.xyxy.cpu():
                x1, y1, x2, y2 = box
                top_left_x = float(x1 * resize_ratio)
                top_left_y = float(y1 * resize_ratio)
                bottom_right_x = float(x2 * resize_ratio)
                bottom_right_y = float(y2 * resize_ratio)
                cv2.rectangle(
                    resized_frame,
                    [round(top_left_x), round(top_left_y)],
                    [round(bottom_right_x), round(bottom_right_y)],
                    (0, 0, 255),
                    2,
                )

            if len(det) > 0:
                tracks = tracker.update(results[0].boxes.cpu(), model.predictor.batch[1])
                if len(tracks) > 0:
                    idx = tracks[:, -1].astype(int)
                    results[0] = results[0][idx]
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    boxes_xyxy = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            # annotated_frame = results[0].plot()

            # Plot the tracks
            for box, box_xyxy, track_id in zip(boxes, boxes_xyxy, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x * resize_ratio), float(y * resize_ratio)))  # x, y center point
                # if len(track) > 30:  # retain 90 tracks for 90 frames
                #     track.pop(0)

                color = compute_color_for_labels(track_id)

                # Draw the box
                x1, y1, x2, y2 = box_xyxy
                top_left_x = float(x1 * resize_ratio)
                top_left_y = float(y1 * resize_ratio)
                bottom_right_x = float(x2 * resize_ratio)
                bottom_right_y = float(y2 * resize_ratio)
                cv2.rectangle(
                    resized_frame,
                    [round(top_left_x), round(top_left_y)],
                    [round(bottom_right_x), round(bottom_right_y)],
                    color,
                    2,
                )

            for tracked_track in [x for x in tracker.tracked_stracks if x.is_activated]:
                track_id = tracked_track.track_id
                # Draw the tracking lines
                track = track_history[track_id]
                # track의 길이로 count할지 결정
                if track_id not in counted_track:
                    if abs(track[-1][0] - track[0][0]) > count_thres_width:
                        counted_track[track_id] = True
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    resized_frame,
                    [points],
                    isClosed=False,
                    color=compute_color_for_labels(track_id),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            for tracked_track in tracker.lost_stracks:
                track_id = tracked_track.track_id
                # Draw the tracking lines
                track = track_history[track_id]
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    resized_frame,
                    [points],
                    isClosed=False,
                    color=compute_color_for_labels(track_id),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            # Display count
            cv2.putText(
                resized_frame,
                f"Num Trackers: {len(track_history)}",
                (10, 50),
                2,
                1,
                (15, 15, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                resized_frame,
                f"Num Apples: {len(counted_track)}",
                (10, 100),
                2,
                1,
                (15, 15, 240),
                2,
                cv2.LINE_AA,
            )

            # Display the annotated frame
            if display:
                cv2.imshow("YOLOv8 Tracking", resized_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if save:
                vid_writer.write(resized_frame)
        else:
            # Break the loop if the end of the video is reached
            break

    print(f"Num Trackers: {len(track_history)}")
    print(f"Num Apples: {len(counted_track)}")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    if save:
        txt_save_path = os.path.join(result_path, os.path.splitext(vid_filename)[0] + ".txt")
        with open(txt_save_path, "w") as f:
            f.write(f"Num Trackers: {len(track_history)}\n")
            f.write(f"Num Apples: {len(counted_track)}")

    return len(track_history), len(counted_track)


def parse_opt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # counting(**vars(opt))

    bytetrack_config_path = "configs/bytetrack.yaml"
    bot_sort_config_path = "configs/bot_sort.yaml"
    my_tracker_config_path = "configs/my_tracker.yaml"
    count_thres = 1 / 3
    # counting(
    #     model_path=r"./detection_checkpoints/yolov8m_1000/weights/best.pt",
    #     tracker_config_path=bytetrack_config_path,
    #     # video_path=r"D:\DeepLearning\Dataset\RDA apple data\2023-08-16\7\230816-Cam1-Line07-L.mp4",
    #     video_path=r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\7\231006-Cam1-Line07-L.mp4",
    #     result_path=r"runs/RDA apple data_1000",
    #     count_thres=count_thres,
    #     display=True,
    #     save=True,
    # )

    vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\*\*[LR].mp4")
    model_path = r"./detection_checkpoints/yolov8m_GFB_WSU2019_KFuji_800/weights/best.pt"
    result_path = r"runs/GFB_WSU2019_KFuji_800_hyperparam"
    counting_results = {}

    for vid_file_path in vid_file_list:
        num_tracks, num_apples = counting(
            model_path=model_path,
            tracker_config_path=bytetrack_config_path,
            video_path=vid_file_path,
            result_path=result_path,
            count_thres=count_thres,
            display=False,
            save=True,
        )
        counting_results[os.path.basename(vid_file_path)] = {"num_tracks": num_tracks, "num_apples": num_apples}

        with open(os.path.join(result_path, "counting_result.json"), "w") as f:
            json.dump(counting_results, f, indent=4)
