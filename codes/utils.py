import cv2
import numpy as np


palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


def xyxy_to_cxywh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return [(x1 + x2) / 2, (y1 + y2) / 2, w, h]


def xywh_to_cxywh(xywh):
    x, y, w, h = xywh
    return [x + w / 2, y + h / 2, w, h]


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def tlbr_to_cxywh(tlbr):
    tlx, tly, brx, bry = tlbr
    return [(tlx + brx) / 2, (tly + bry) / 2, brx - tlx, tly - bry]


def tlbr_to_xywh(tlbr):
    ## yolov8에는 tlbr이 [min_x, min_y, max_x, max_y]
    ## bry가 왜 try보다 큰 값인지..??
    tlx, tly, brx, bry = tlbr
    return [tlx, tly, brx - tlx, bry - tly]


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
        (255, 255, 255),
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


def draw_bbox(image, box_xyxy, resize_ratio, color):
    x1, y1, x2, y2 = box_xyxy
    top_left_x = float(x1 * resize_ratio)
    top_left_y = float(y1 * resize_ratio)
    bottom_right_x = float(x2 * resize_ratio)
    bottom_right_y = float(y2 * resize_ratio)
    cv2.rectangle(
        image,
        [round(top_left_x), round(top_left_y)],
        [round(bottom_right_x), round(bottom_right_y)],
        color,
        2,
    )
