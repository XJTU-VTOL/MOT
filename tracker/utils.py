import numpy as np
from scipy.spatial.distance import cdist
from tracker.kalman_filter import chi2inv95
import lap
from numba import jit

def joint_tracks(tlista, tlistb):
    """
    合并两个轨迹集合，剔除相同轨迹

    :param tlista:
    :param tlistb:
    :return:
    """
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


def sub_tracks(tlista, tlistb):
    """
    从 tlista 中删除 tlistb 中的元素

    :param tlista:
    :param tlistb:
    :return:
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def embedding_distance(tracks, detections, metric='cosine'):
    """
    TODO 加速？

    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features)) # Nomalized features

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    Kalman Filter Gate Distance 可以剔除与预测较远的物体。

    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_) * gating_distance
    return cost_matrix


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


@jit(cache=True, nopython=True)
def ious(box1, box2, IoU):
    M = box1.shape[0]
    N = box2.shape[0]

    for b1_ind in range(M):
        for b2_ind in range(N):
            b1 = box1[b1_ind]
            b2 = box2[b2_ind]

            lx = min(b1[2], b2[2])
            mx = max(b1[0], b2[0])

            ly = min(b1[3], b2[3])
            my = max(b1[1], b2[1])

            I = max(lx - mx, 0) * max(ly - my, 0)
            U = (b1[2] - b1[0]) * (b1[3] - b1[1]) + \
                (b2[2] - b2[0]) * (b2[3] - b2[1])

            IoU[b1_ind, b2_ind] = I / U


def IoU_wrapper(box1, box2):
    IoU = np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)
    ious(box1, box2, IoU)
    return IoU


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = np.stack([track.tlbr for track in atracks], axis=0)
        btlbrs = np.stack([track.tlbr







                           for track in btracks], axis=0)

    _ious = IoU_wrapper(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
