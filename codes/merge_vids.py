import os
import cv2
import numpy as np
from tqdm import tqdm


def merge_vids(vid_paths: list[str], captions: list[str], result_path):
    # 영상이 두 개 이하면 합치는 의미 없음
    assert len(vid_paths) >= 2, "영상이 두 개 이상 필요합니다."
    os.makedirs(os.path.split(result_path)[0], exist_ok=True)

    captures = []
    for vid_path in vid_paths:
        captures.append(cv2.VideoCapture(vid_path))

    num_vids = len(captures)

    fps = captures[0].get(cv2.CAP_PROP_FPS)
    w = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    for capture in captures[1:]:
        assert (
            fps == capture.get(cv2.CAP_PROP_FPS)
            and w == int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            and h == int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ), "fps, w 또는 h가 다른 영상이 있습니다."

    vid_writer = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * num_vids, h))

    print(f"result_path: {result_path}")

    frame_cnt = 0
    while all([capture.isOpened() for capture in captures]):
        frame_cnt += 1
        print(f"processing frame num: {frame_cnt}", end="\r")

        ret_and_frames = [capture.read() for capture in captures]
        rets = [ret_and_frame[0] for ret_and_frame in ret_and_frames]
        frames = [ret_and_frame[1] for ret_and_frame in ret_and_frames]

        if not all(rets):
            break

        concated_img = np.concatenate(frames, axis=1)

        # 영상 사이 구분선 넣기
        for i in range(1, num_vids):
            cv2.line(concated_img, (w * i, 0), (w * i, h), (0, 0, 0), 1, cv2.LINE_AA)

        for i, caption in enumerate(captions):
            cv2.putText(
                concated_img,
                caption,
                (10 + w * i, 150),
                2,
                1.2,
                (15, 51, 255),
                2,
                cv2.LINE_AA,
            )

        vid_writer.write(concated_img)

    print("\nvid saved!")

    for capture in captures:
        capture.release()


if __name__ == "__main__":
    result_dir = "merged_vids/counting/ByteTrack_MyTracker"
    os.makedirs(result_dir, exist_ok=True)

    source_vids = [
        "runs/tracking/ByteTrack/RDA_800",
        "runs/tracking/MyTracker/RDA_800_vx",
    ]

    captions = ["ByteTrack", "Proposed Tracker"]

    filenames = [
        "231006-Cam1-Line07-L.mp4",
        "231006-Cam1-Line11-L.mp4",
        "231006-Cam1-Line15-L.mp4",
        "231006-Cam1-Line07-L_counted_tracks.mp4",
        "231006-Cam1-Line11-L_counted_tracks.mp4",
        "231006-Cam1-Line15-L_counted_tracks.mp4",
    ]

    for filename in tqdm(filenames, desc="vid_num", position=0, leave=True):
        source_vid_paths = [os.path.join(source_vid, filename) for source_vid in source_vids]
        result_path = os.path.join(result_dir, filename)

        merge_vids(source_vid_paths, captions, result_path)

    ## 원본 counting + counted track 영상

    # result_dir = r"merged_vids\counting\MyTracker_RDA_800_vx"

    # source_vid_paths = [
    #     r"runs\tracking\MyTracker\RDA_800_vx\230816-Cam1-Line07-L.mp4",
    #     r"runs\tracking\MyTracker\RDA_800_vx\230816-Cam1-Line10-L.mp4",
    #     r"runs\tracking\MyTracker\RDA_800_vx\230816-Cam1-Line11-L.mp4",
    #     r"runs\tracking\MyTracker\RDA_800_vx\230816-Cam1-Line14-L.mp4",
    #     r"runs\tracking\MyTracker\RDA_800_vx\230816-Cam1-Line15-L.mp4",
    #     r"runs\tracking\MyTracker\RDA_800_vx\230816-Cam1-Line18-L.mp4",
    # ]

    # captions = ["all tracks", "counted tracks"]

    # for source_vid_path in source_vid_paths:
    #     counted_track_vid_path = os.path.splitext(source_vid_path)[0] + "_counted_tracks.mp4"
    #     result_path = os.path.join(result_dir, os.path.split(source_vid_path)[1])

    #     merge_vids([source_vid_path, counted_track_vid_path], captions, result_path)
