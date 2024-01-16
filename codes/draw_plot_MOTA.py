import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc("font", family=font)


if __name__ == "__main__":
    fig_save_path = "runs/plots"
    os.makedirs(fig_save_path, exist_ok=True)

    yolo_MOTA_dict = {"ByteTrack": [7.3, 22.3, 31.9, 39.3], "제안 방법": [23.6, 37.9, 39.3, 39.9]}
    faster_rcnn_MOTA_dict = {"ByteTrack": [10.1, 24.9, 30.4, 34.6], "제안 방법": [27.2, 34.5, 34.6, 33.2]}
    efficientdet_MOTA_dict = {"ByteTrack": [10.9, 26.5, 33.8, 39.5], "제안 방법": [27.0, 38.6, 39.9, 39.9]}

    x = [1, 2, 3, 4]
    fps_list = [5, 10, 15, 30]
    plt.style.use("default")
    plt.xticks(x, fps_list)
    plt.ylim(0, 50)
    plt.xlabel("FPS")
    plt.ylabel("MOTA(%)")
    plt.tick_params(axis="both", direction="in")

    plt.plot(x, yolo_MOTA_dict["ByteTrack"], "+-", color="red", label="ByteTrack YOLOv8")
    plt.plot(x, faster_rcnn_MOTA_dict["ByteTrack"], "+-", color="green", label="ByteTrack Faster R-CNN")
    plt.plot(x, efficientdet_MOTA_dict["ByteTrack"], "+-", color="blue", label="ByteTrack EfficientDet")

    plt.plot(x, yolo_MOTA_dict["제안 방법"], "x--", color="red", label="제안 방법 YOLOv8")
    plt.plot(x, faster_rcnn_MOTA_dict["제안 방법"], "x--", color="green", label="제안 방법 Faster R-CNN")
    plt.plot(x, efficientdet_MOTA_dict["제안 방법"], "x--", color="blue", label="제안 방법 EfficientDet")

    plt.legend(loc="lower right", prop=font)

    # plt.show()
    plt.savefig(os.path.join(fig_save_path, "MOTA 비교.png"))
    plt.close()
