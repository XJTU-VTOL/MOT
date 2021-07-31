import unittest
from model.BaseModule import YoloBackbone
import torch
from torchvision.transforms import transforms as T
from model.model_util import parse_model_cfg
from model.BaseModule import YoloHead
import torch.nn as nn
import json
from model.model_util import extract_net_and_yolo_param

class module_test(unittest.TestCase):
    def test_backbone(self, return_result=False):
        cfg_file = "../model_cfg/yolov3.cfg"
        backbone = YoloBackbone(cfg_file).cuda().eval()

        data = torch.randn((1, 3, 608, 1088)).cuda()
        features = backbone(data)
        for f in features:
            print("Output in :", f.shape)

        if return_result:
            return features

    def test_yolo(self):
        net, yolo_def = extract_net_and_yolo_param("../model_cfg/yolov3.cfg")
        yolo_head = YoloHead(net, yolo_def)
        yolo_head.cuda()
        yolo_head.eval()
        targets = torch.tensor([
            [0, 1, 0, 1, 1, 1, 1],
        ]).cuda()

        features = self.test_backbone(True)
        pred = yolo_head(features)
        pass

if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=2)
