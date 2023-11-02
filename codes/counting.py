import os
import cv2
import numpy as np
import torch
import argparse
import glob
import json
import shutil
from omegaconf import OmegaConf
from collections import defaultdict

from ultralytics import YOLO
from ultralytics_clone.trackers.byte_tracker import BYTETracker
from ultralytics_clone.trackers.my_tracker import MyTracker
from ultralytics_clone.trackers.bot_sort import BOTSORT

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    xyxy = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
    return xyxy


def compute_color_for_labels(label):
    color = [int(int(p * (label**2 - label + 1)) % 255) for p in palette]
    color[1] = color[1] % 100
    return tuple(color)


def display_count(image, text, pos):
    cv2.putText(
        image,
        text,
        pos,
        2,
        1,
        (15, 15, 240),
        2,
        cv2.LINE_AA,
    )


def draw_trajectory(image, track_history, track_id):
    points = np.hstack(track_history).astype(np.int32).reshape((-1, 1, 2))
    color = compute_color_for_labels(track_id)
    text_pos = [int(track_history[-1][0]), int(track_history[-1][1])]
    cv2.putText(
        image,
        f"{track_id}",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.polylines(
        image,
        [points],
        isClosed=False,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def counting(
    model_path,
    tracker,
    video_path,
    result_path,
    count_thres,
    plot_lost_tracker=False,
    display=False,
    save=False,
    save_counted_tracks_only=True,
):
    resized_height = 1000
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
    tracks_history = defaultdict(lambda: [])
    counted_track_ids = set()

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
                track = tracks_history[track_id]
                track.append((float(x * resize_ratio), float(y * resize_ratio)))  # x, y center point
                # if len(track) > 30:  # retain 90 tracks for 90 frames
                #     track.pop(0)

                color = compute_color_for_labels(track_id)

                # Draw the box
                if box_xyxy:
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

            # activated track 표시
            for tracked_track in [x for x in tracker.tracked_stracks if x.is_activated]:
                track_id = tracked_track.track_id
                # Draw the tracking lines
                track_history = tracks_history[track_id]
                # track의 길이로 count할지 결정
                if track_id not in counted_track_ids:
                    if abs(track_history[-1][0] - track_history[0][0]) > count_thres_width:
                        counted_track_ids.add(track_id)

                draw_trajectory(resized_frame, track_history, track_id)

            # lost track 표시
            for tracked_track in tracker.lost_stracks:
                track_id = tracked_track.track_id
                # track이 화면을 벗어나면 건너뛰기
                if tracked_track.mean[0] > width:
                    continue
                # Draw the tracking lines
                track_history = tracks_history[track_id]
                draw_trajectory(resized_frame, track_history, track_id)

            # Display count
            display_count(resized_frame, f"Num Trackers: {len(tracks_history)}", (10, 50))
            display_count(resized_frame, f"Num Apples: {len(counted_track_ids)}", (10, 100))

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

    print(f"Num Trackers: {len(tracks_history)}")
    print(f"Num Apples: {len(counted_track_ids)}")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    if save:
        txt_save_path = os.path.join(result_path, os.path.splitext(vid_filename)[0] + ".txt")
        with open(txt_save_path, "w") as f:
            f.write(f"Num Trackers: {len(tracks_history)}\n")
            f.write(f"Num Apples: {len(counted_track_ids)}")

        # count한 궤적 정보 저장
        if save_counted_tracks_only:
            counted_tracks_history = defaultdict(lambda: [])
            for track_id in counted_track_ids:
                counted_tracks_history[track_id] = tracks_history[track_id]

            with open(
                os.path.join(result_path, os.path.splitext(vid_filename)[0] + "_counted_tracks_history.json"), "w"
            ) as f:
                json.dump(counted_tracks_history, f, indent=4)

    return len(tracks_history), len(counted_track_ids)


def parse_opt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # counting(**vars(opt))

    tracker_name = "MyTracker"
    count_thres = 1 / 4

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
    #     plot_lost_tracker=True,
    #     display=True,
    #     save=False,
    # )

    vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\*\*L.mp4")
    vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\*\231006-Cam1-Line07-L.mp4")

    model_path = r"./detection_checkpoints/yolov8m_RDA_800/weights/best.pt"
    result_path = r"runs/tracking/MyTracker/RDA_800_vx_vy"
    counting_results = {}
    tracker_config = OmegaConf.load(tracker_config_path)

    for vid_file_path in vid_file_list:
        tracker = tracker_constructors[tracker_name](tracker_config)

        num_tracks, num_apples = counting(
            model_path=model_path,
            tracker=tracker,
            video_path=vid_file_path,
            result_path=result_path,
            count_thres=count_thres,
            plot_lost_tracker=True,
            display=False,
            save=True,
            save_counted_tracks_only=True,
        )
        counting_results[os.path.basename(vid_file_path)] = {"num_tracks": num_tracks, "num_apples": num_apples}

    with open(os.path.join(result_path, "counting_result.json"), "w") as f:
        json.dump(counting_results, f, indent=4)

    shutil.copy(tracker_config_path, os.path.join(result_path, os.path.basename(tracker_config_path)))
