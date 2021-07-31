import unittest
import numpy as np
from tracker.utils import ious
import torch
from torchvision.ops import box_iou
from tracker.tracker import Tracker
from Logger import Logger

class TrackerTest(unittest.TestCase):

    def test_iou(self):
        b1 = np.abs(np.random.random((4, 4)) * np.random.randint(1, 10))
        b1[:, 2:] += b1[:, :2]
        b2 = np.abs(np.random.random((4, 4)) * np.random.randint(1, 10))
        b2[:, 2:] += b2[:, :2]
        IoU = np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float)
        ious(b1, b2, IoU)

        b1_tensor = torch.from_numpy(b1)
        b2_tensor = torch.from_numpy(b2)
        IoU_tensor = box_iou(b1_tensor, b2_tensor)

        T_IoU = IoU_tensor.numpy()
        self.assertLess(np.max(np.abs(IoU - T_IoU)), 1e-3)

    def test_tracker(self):
        logger = Logger("log.log", level='debug')
        T = Tracker()

        det1 = np.array([
            [1., 1., 5., 5., 0.7, 3],
            [3., 2., 11., 12, 0.6, 2]
        ])

        det1_feature = np.random.random((2, 256))
        det1 = np.concatenate([det1, det1_feature], axis=1)

        T.update(det1)

        det1 = np.array([
            [1.2, 0.8, 5.1, 5.3, 0.7, 3],
            [3.1, 2.2, 9.6, 9.7, 0.6, 2]
        ])

        det1_feature = np.random.random((2, 256))
        det1 = np.concatenate([det1, det1_feature], axis=1)
        T.update(det1)

if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=2)


