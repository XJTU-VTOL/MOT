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

if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=1)


