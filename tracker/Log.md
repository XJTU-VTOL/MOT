几点说明

1. Kalman Filter 中 Gating 解释

Gate 判断 Measurement 与 Prediction 的差距，并根据 chi2 数据判断概率。如果差距过大，那么概率小，所以在函数 `fuse_motion` 会过大偏差的距离设置为 `inf`。

> 在 FairMOT 中: ` We also use
Kalman Filter [34] to predict the locations of the tracklets in
the current frame. If it is too far from the linked detection, we
set the corresponding cost to infinity which effectively prevents
from linking the detections with large motion.` 与此功能相同。