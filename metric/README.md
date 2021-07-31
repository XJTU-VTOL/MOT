* `DetectionMetric`

1. `num_cls` (int) 类别的数量。

2. `IoUs` (float), IoU 阈值，判断检测框与真实框是否匹配。

3. `conf_threshold` (float) Confidence 阈值，小于此阈值的框忽略。

* `APCurve`

1. `num_cls` (int) 类别的数量。

2. `IoUs` (np.ndarray), IoU 阈值， AP 中一系列的 IoU 阈值。

3. `conf_threshold` (float) Confidence 阈值，小于此阈值的框忽略。

* `TrackMetric`

1. `num_cls` (int) 类别的数量。

2. `IoUs` (float), IoU 阈值，判断检测框与真实框是否匹配。

3. `conf_threshold` (float) Confidence 阈值，小于此阈值的框忽略。
