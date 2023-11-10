# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import numpy as np

from .basetrack import BaseTrack, TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYWH


class MyTrack(STrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-angle-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.
        tlbr_to_tlwh(tlbr): Convert tlbr bounding box to tlwh format.
        tlwh_to_tlbr(tlwh): Convert tlwh bounding box to tlbr format.
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, gt_id=None):
        """Initialize new STrack instance."""
        super().__init__(tlwh, score, cls, gt_id)
        self.start_x, self.start_y, _, _ = self.tlwh_to_xywh(tlwh[:-1])
        self.coefficient_matrix = []
        self.num_matched = 0

    def predict(self):
        """Predicts mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for given stracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        #     self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.num_matched += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def calc_coefficient_matrix(self, good_stracks):
        self.coefficient_matrix = {}
        for good_strack in good_stracks:
            # x, y, w, h, vx, vy, vw, vh
            # vx와 vy만 필요
            if (good_strack.mean[4] == 0) or (good_strack.mean[5] == 0):
                continue

            coefficient_matrix = [self.mean[4] / good_strack.mean[4], self.mean[5] / good_strack.mean[5]]
            self.coefficient_matrix[good_strack.track_id] = coefficient_matrix

    def update_with_coefficient_matrix(self, good_stracks):
        # coeffs = []
        vx_vys = []

        for good_strack in good_stracks:
            if good_strack.track_id in self.coefficient_matrix:
                coefficient_matrix = self.coefficient_matrix[good_strack.track_id]
                vx_vys.append(
                    [good_strack.mean[4] * coefficient_matrix[0], good_strack.mean[5] * coefficient_matrix[1]]
                )
                # coeffs.append(coefficient_matrix)

        if len(vx_vys) > 0:
            vx_vy = np.mean(vx_vys, axis=0)

            self.mean[4] = vx_vy[0]  # vx
            # self.mean[5] = vx_vy[1]  # vy

            # self.predict()

        # if len(vx_vys) >= 1:
        #     print()
        #     print("id:", self.track_id)
        #     print("pos:", self.mean[0], self.mean[1])
        #     print("mean:", self.mean[4], self.mean[5])
        #     print("coefficient_mat:")
        #     for coeff in coeffs:
        #         print(coeff)
        #     print("vx, vy")
        #     for vx_vy in vx_vys:
        #         print(vx_vy)

    def force_update(self, good_stracks, coeff):
        vx_vys = []

        for good_strack in good_stracks:
            vx_vys.append([good_strack.mean[4], good_strack.mean[5]])

        if len(vx_vys) > 0:
            vx_vy = np.mean(vx_vys, axis=0)

            self.mean[4] = vx_vy[0] * coeff
            self.mean[5] = vx_vy[1] * coeff

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xywh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        return self.tlwh_to_xywh(self._tlwh)

    def convert_coords(self, tlwh):
        """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class MyTracker:
    """
    적용 목록
    - Kalman filter 사용 시 종횡비 대신 w, h 사용
    - activate thresh를 설정하여 track이 생성되고 이 thresh 이전에 다시 매칭되면 activate로 전환


    적용할 목록
    - IoU - Re-ID Fusion - get_dist 함수 바꾸기
    - Observation-centric Re-Update stage - 필요할까?
    - good tracks로 새로운 track 움직임 추정

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (object): Kalman Filter object.

    Methods:
        update(results, img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IOU.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.good_stracks = []  # type: list[STrack]
        # self.unconfirmed_stracks = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xyxy
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls
        gt_ids = results.id
        gt_id_track_id_dict = {}

        remain_inds = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        if len(scores) == 1:
            remain_inds = [remain_inds]
            inds_low = [inds_low]
            inds_high = [inds_high]
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        gt_ids_first = None
        gt_ids_second = None
        if gt_ids != None:
            gt_ids_first = gt_ids[remain_inds]
            gt_ids_second = gt_ids[inds_second]

        detections = self.init_track(dets, scores_keep, cls_keep, gt_ids_first, img)
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)

        # new track predict하는 코드 추가
        self.multi_predict(unconfirmed)

        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
                gt_id_track_id_dict[det.gt_id] = track.track_id
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                gt_id_track_id_dict[det.gt_id] = track.track_id
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        detections_second = self.init_track(dets_second, scores_second, cls_second, gt_ids_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
                gt_id_track_id_dict[det.gt_id] = track.track_id
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                gt_id_track_id_dict[det.gt_id] = track.track_id

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Good track 찾아내기
        # 가장 긴 track을 최대 n개 찾아내기
        stracks_lens = {}
        good_stracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                if track.num_matched > self.args.good_track_thresh:
                    stracks_lens[track] = track.num_matched
                # track_x_len = abs(track.start_x - track.mean[0])
                # if track_x_len > self.vid_width * self.args.good_track_thresh:
                #     stracks_lens[track] = track_x_len
        stracks_lens = sorted(stracks_lens.items(), key=lambda item: item[1], reverse=True)
        for strack, _ in stracks_lens:
            good_stracks.append(strack)
            if len(good_stracks) == self.args.num_good_stracks:
                break

        print(f"num good stracks: {len(good_stracks)}")

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
            if detections[idet].gt_id not in gt_id_track_id_dict:
                gt_id_track_id_dict[detections[idet].gt_id] = track.track_id
        for it in u_unconfirmed:
            track = unconfirmed[it]
            # activate threshold 안에 다시 검출된 경우에 activate로 전환
            if self.frame_id - track.start_frame >= self.args.activate_thresh:
                track.mark_removed()
                removed_stracks.append(track)
            # else:
            # self.tracked_stracks.append(track)
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            # 이 코드가 실제로 activate 시키지는 않음(frame_id가 1인 경우만 activate)
            track.activate(self.kalman_filter, self.frame_id)

            # good strack의 정보로 track 강제 업데이트
            track.force_update(good_stracks, self.args.new_track_update_coeff)

            activated_stracks.append(track)
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
            # good track으로 lost strack 업데이트
            else:
                track.update_with_coefficient_matrix(good_stracks)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        # good track으로 coefficient matrix 계산하기
        for track in self.tracked_stracks:
            if track.is_activated:
                track.calc_coefficient_matrix(good_stracks)

        # self.good_stracks.extend(good_stracks)
        return (
            np.asarray(
                [x.tlbr.tolist() + [x.track_id, x.score, x.cls, x.idx] for x in self.tracked_stracks if x.is_activated],
                dtype=np.float32,
            ),
            gt_id_track_id_dict,
        )

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, gt_ids, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        if gt_ids == None:
            return [MyTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections
        else:
            return (
                [MyTrack(xyxy, s, c, gt_id.item()) for (xyxy, s, c, gt_id) in zip(dets, scores, cls, gt_ids)]
                if len(dets)
                else []
            )  # detections

    # def get_dists(self, tracks, detections):
    #     """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""
    #     dists = matching.iou_distance(tracks, detections)
    #     dists_mask = dists > self.proximity_thresh

    #     # TODO: mot20
    #     # if not self.args.mot20:
    #     dists = matching.fuse_score(dists, detections)

    #     if self.args.with_reid and self.encoder is not None:
    #         emb_dists = matching.embedding_distance(tracks, detections) / 2.0
    #         emb_dists[emb_dists > self.appearance_thresh] = 1.0
    #         emb_dists[dists_mask] = 1.0
    #         dists = np.minimum(dists, emb_dists)
    #     return dists

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IOU and fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Returns the predicted tracks using the YOLOv8 network."""
        MyTrack.multi_predict(tracks)

    def reset_id(self):
        """Resets the ID counter of STrack."""
        MyTrack.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combine two lists of stracks into a single one."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate stracks with non-maximum IOU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
