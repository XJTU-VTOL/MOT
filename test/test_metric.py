import unittest
import numpy as np
from tracker.utils import ious
import torch
from torchvision.ops import box_iou
from metric import DetectionMetric, TrackMetric
from Logger import Logger
from pprint import pprint
from metric.util import find_id

def generate_data():
    xy = torch.abs(torch.randn(100, 2))
    wh = torch.abs(torch.randn(100, 2))
    x2y2 = xy + wh
    conf = torch.zeros((100, 1))
    conf.uniform_(0., 1.0)

    cls = np.random.randint(0, 10, (100,))
    cls = torch.from_numpy(cls).unsqueeze(1)

    data = torch.cat([xy, x2y2, conf, cls], dim=1)
    return data

class TrackerTest(unittest.TestCase):

    def test_detection(self):
        det1 = torch.tensor([
            [3., 3., 7., 7., 0.7, 0],
            [5., 1., 12., 5., 0.8, 0],
            [1., 4., 5., 9., 0.9, 0],
            [17., 2., 22., 7., 0.7, 1],
            [16, 5., 21., 10., 0.8, 1]
        ])

        target1 = torch.tensor([
            [2., 2., 6., 6., 0],
            [15., 1., 20., 6., 1],
            [12., 9., 17., 14., 1]
        ])

        # Random Generation
        det2 = generate_data()
        target2 = generate_data()[:, [0, 1, 2, 3, 5]]

        Metric = DetectionMetric(num_cls=10, IoUs=0.3)
        Metric.update(det1, target1)
        Metric.update(det2, target2)
        pprint(Metric.compute())

    def test_id_match_kernel(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6]).long().cuda()
        b = torch.tensor([2, 3, 1, 4]).long().cuda()
        find_id(a, b)

    def test_tracker(self):
        """
        两帧与上面相同

        :return:
        """
        det1 = torch.tensor([
            [0, 3., 3., 7., 7., 0, 1],
            [0, 5., 1., 12., 5., 0, 2],
            [0, 1., 4., 5., 9.,  0, 3],
            [0, 17., 2., 22., 7., 1, 4],
            [0, 16, 5., 21., 10., 1, 5]
        ]).cuda()

        target1 = torch.tensor([
            [0, 2., 2., 6., 6., 0, 1],
            [0, 15., 1., 20., 6., 1, 2],
            [0, 12., 9., 17., 14., 1, 3]
        ]).cuda()

        det2 = torch.tensor([
            [1, 3., 3., 7., 7., 0, 1],
            [1, 5., 1., 12., 5., 0, 2],
            [1, 1., 4., 5., 9., 0, 3],
            [1, 17., 2., 22., 7., 1, 4],
            [1, 16, 5., 21., 10., 1, 5]
        ]).cuda()

        target2 = torch.tensor([
            [1, 2., 2., 6., 6., 0, 1],
            [1, 15., 1., 20., 6., 1, 2],
            [1, 12., 9., 17., 14., 1, 3]
        ]).cuda()

        Metric = TrackMetric(2, 0.3)
        Metric.update(det1, target1)
        Metric.update(det2, target2)
        pprint(Metric.compute())

if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=2)


