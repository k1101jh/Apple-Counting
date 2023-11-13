## https://github.com/cheind/py-motmetrics

# import required packages
import os
import glob
import logging
import motmetrics as mm
import numpy as np
from collections import OrderedDict


def motMetricsEnhancedCalculator(gtSource, tSource):
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=",")

    # load tracking output
    t = np.loadtxt(tSource, delimiter=",")

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype("int").tolist(), t_dets[:, 0].astype("int").tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="full")
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info("Comparing %s...", k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5))
            names.append(k)
        else:
            logging.warning("No ground truth for %s, skipping.", k)

    return accs, names


def motMetrics(gtfiles, tsfiles):
    # gtfiles = glob.glob(os.path.join(gt_dir_path, "*.txt"))
    # tsfiles = glob.glob(os.path.join(source_dir_path, "*.txt"))

    logging.info("Found %d groundtruths and %d test files.", len(gtfiles), len(tsfiles))
    logging.info("Available LAP solvers %s", str(mm.lap.available_solvers))
    logging.info("Default LAP solver '%s'", mm.lap.default_solver)
    logging.info("Loading files.")

    gt = OrderedDict([(os.path.basename(f)[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.basename(f)[0], mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    metrics = list(mm.metrics.motchallenge_metrics)
    # if args.exclude_id:
    #     metrics = [x for x in metrics if not x.startswith("id")]

    logging.info("Running metrics")

    # if args.id_solver:
    #     mm.lap.default_solver = args.id_solver
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info("Completed")


if __name__ == "__main__":
    # vid_dir = "1"
    # gt_source_path = rf"D:\DeepLearning\Dataset\Apple\SensitivityAnalysis\{vid_dir}\gt\gt.txt"
    # t_source_path = rf"D:\DeepLearning\Experiment\Multi Object Tracking\Apple-Counting\runs\sensitivity_analysis\ByteTrack_dr_1_fps_30\{vid_dir}_tracker_result.txt"
    # motMetricsEnhancedCalculator(gt_source_path, t_source_path)

    # 모든 파일 성능 확인

    tracker_name = "ByteTrack"
    detection_rate = "1"
    fps = 30

    gt_files = glob.glob(r"D:\DeepLearning\Experiment\Multi Object Tracking\TrackEval\data\gt\*.txt")
    ts_files = glob.glob(
        rf"D:\DeepLearning\Experiment\Multi Object Tracking\Apple-Counting\runs\sensitivity_analysis\{tracker_name}_dr_{detection_rate}_fps_{fps}\*_tracks_result.txt"
    )
    motMetrics(gt_files, ts_files)
