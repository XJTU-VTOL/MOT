import numpy as np
from .tracklet import STrack, TrackState
from .utils import *
from .kalman_filter import KalmanFilter
import Logger

class Tracker:
    """
    Parameter:
        1. 追踪缓存

    """
    def __init__(self):
        # 存储所有轨迹
        self.tracked_tracks = []  # 追踪的轨迹
        self.lost_tracks = []  # 跟丢轨迹
        self.removed_tracks = []  # 放弃轨迹

        # 状态变量
        self.frame_id = 0  # 当前帧
        self.kalman_filter = KalmanFilter()
        self.init_state = False

        # 参数
        # TODO confidence threshold for a new track
        self.det_thresh = 0.7
        # TODO lost time to remove a track
        self.max_time_lost = 10
        self.embed_match_threshold = 0.7
        self.iou_match_threshold = 0.6

    def reset(self):
        self.frame_id = 0

    def update(self, preds: np.ndarray):
        """
        更新当前帧的检测结果

        注意：
            1. 当前的所有检测框恢复到原图片对应尺寸

        :param preds:
            （N， 6 + D）
            x1, y1, x2, y2, conf, cls, embedding
        :return:
            (M, 4)
            x1, y1, x2, y2, cls, id 所有追踪到的物体
        """
        if self.frame_id == 0:
            # 第一帧，记录所有物体
            self.frame_id = 1
            for i in range(preds.shape[0]):
                tlbrs = preds[i, :5]
                f = preds[i, 6:]
                track = STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30)
                track.activate(self.kalman_filter, self.frame_id)
                self.tracked_tracks.append(track)
            ids = np.array([track.track_id for track in self.tracked_tracks if track.is_activated])
            frame_id = np.array([self.frame_id for track in self.tracked_tracks if track.is_activated])
            tracked_results = preds[:, [0, 1, 2, 3, 5]]
            tracked_results = np.concatenate([frame_id[:, np.newaxis], tracked_results, ids[:, np.newaxis]], axis=1)

            return tracked_results

        # TODO 没有用到 cls 信息
        if len(preds) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(preds[:, :5], preds[:, 6:])]
        else:
            # TODO How to Deal with 0 detection?
            detections = []

        unconfirmed = []  # 未匹配的轨迹
        tracked_tracks = []  # type: list[STrack]
        refined_tracks = []
        activated_tracks = []
        lost_tracks = []
        removed_tracks = []

        # 收集所有激活的轨迹
        # TODO: 应该是 TrackState 是 Tracked 而不是 是否激活
        for track in self.tracked_tracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                # TODO Debug 时使用
                unconfirmed.append(track)
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_tracks.append(track)

        ''' Step 1: First association, with embedding '''
        track_pool = joint_tracks(tracked_tracks, self.lost_tracks)
        if len(tracked_tracks) > 0 and len(detections) > 0:
            STrack.multi_predict(track_pool, self.kalman_filter)  # 预测更新
            dists = embedding_distance(track_pool, detections)  # 由卡尔曼滤波 Gate 阈值化距离（概率）
            dists = fuse_motion(self.kalman_filter, dists, track_pool, detections)
            matches, u_track, u_detection = linear_assignment(dists, thresh=self.embed_match_threshold)
            # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

            for itracked, idet in matches:
                # itracked is the id of the track and idet is the detection
                track = track_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    # If the track is active, add the detection to the track
                    track.update(detections[idet], self.frame_id)
                    activated_tracks.append(track)
                else:
                    # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks.append(track)


            # TODO Logger
            if Logger.logger is not None:
                Logger.logger.logger.debug("Match {} trackes after embedding with threshold {}".format(len(matches), self.embed_match_threshold))
        else:
            matches, u_track, u_detection = [], [], []

        ''' Step 2: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if track_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(track_pool[i])
        if len(r_tracked_stracks) > 0 and len(detections) > 0:
            dists = iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = linear_assignment(dists, thresh=self.iou_match_threshold)
            # matches is the list of detections which matched with corresponding tracks by IOU distance method
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks.append(track)

            # TODO Logger
            if Logger.logger is not None:
                Logger.logger.logger.debug(
                    "Match {} trackes after IoU with threshold {}".format(len(matches), self.iou_match_threshold))
        else:
            matches, u_track, u_detection = [], [], []

        # 收集所有经过 IoU 匹配后未配上的轨迹
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        if len(unconfirmed) > 0 and len(detections) > 0:
            dists = iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_tracks.append(unconfirmed[itracked])
        else:
            matches, u_unconfirmed, u_detection = [], [], []

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)

        """ Step 3: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refined_tracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_tracks)

        # get scores of lost tracks
        output_tracks = [track.tlbr for track in self.tracked_tracks if track.is_activated]
        ids = np.array([track.track_id for track in self.tracked_tracks if track.is_activated])
        cls = np.array([0 for track in self.tracked_tracks if track.is_activated])
        frame_id = np.array([self.frame_id for track in self.tracked_tracks if track.is_activated])
        if len(output_tracks) > 0:
            output_tracks = np.stack(output_tracks, axis=0)
            output_tracks = np.concatenate([frame_id[:, np.newaxis], output_tracks, cls[:, np.newaxis], ids[:, np.newaxis]], axis=1)
        self.frame_id += 1

        # TODO 设置 Log 等级
        if Logger.logger is not None:
            Logger.logger.logger.info('===========Frame {}=========='.format(self.frame_id))
            Logger.logger.logger.info('Activated: {}'.format([track.track_id for track in activated_tracks]))
            Logger.logger.logger.info('Refind: {}'.format([track.track_id for track in refined_tracks]))
            Logger.logger.logger.info('Lost: {}'.format([track.track_id for track in lost_tracks]))
            Logger.logger.logger.info('Removed: {}'.format([track.track_id for track in removed_tracks]))
            # print('Final {} s'.format(t5-t4))
        return output_tracks

