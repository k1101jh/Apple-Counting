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
from counter import Counter
from utils import compute_color_for_labels
from utils import display_count
from utils import draw_trajectory
from utils import draw_bbox
from utils import xywh_to_xyxy


def counting(
    model,
    tracker,
    video_path,
    result_path,
    count_thres,
    resized_height=1000,
    plot_lost_tracker=False,
    show=False,
    save=False,
):
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


def parse_opt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # counting(**vars(opt))

    task = "tracking"
    tracker_name = "ByteTrack"
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
    # vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\2023-10-06\*\*L.mp4")
    vid_file_list = glob.glob(r"D:\DeepLearning\Dataset\RDA apple data\*\*\*[LR].mp4")

    model_path = "detection_checkpoints/yolov8m_RDA_800/weights/best.pt"
    result_path = f"runs/{task}/{tracker_name}/RDA_800_final"
    counting_results = {}
    tracker_config = OmegaConf.load(tracker_config_path)

    # Load the YOLOv8 model
    model = YOLO(model_path)

    for vid_file_path in tqdm(vid_file_list, desc="vids", position=0, leave=True):
        tracker = tracker_constructors[tracker_name](tracker_config)

        num_tracks, num_apples = counting(
            model=model,
            tracker=tracker,
            video_path=vid_file_path,
            result_path=result_path,
            count_thres=count_thres,
            resized_height=resized_height,
            plot_lost_tracker=True,
            show=False,
            save=True,
        )
        counting_results[os.path.basename(vid_file_path)] = {"num_tracks": num_tracks, "num_apples": num_apples}

    with open(os.path.join(result_path, "counting_result.json"), "a") as f:
        json.dump(counting_results, f, indent=4)

    shutil.copy(tracker_config_path, os.path.join(result_path, os.path.basename(tracker_config_path)))
