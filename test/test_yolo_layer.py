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
    def test_yolo_train(self):
        net, yolo_def = extract_net_and_yolo_param("../model_cfg/yolov3.cfg")
        yolo_head = YoloHead(net, yolo_def)
        yolo_head.cuda()
        yolo_head.train()
        targets = torch.tensor([
            [0, 1, 0, 1, 1, 1, 1],
        ]).cuda()

        #features = self.test_backbone(True)
        f1=torch.rand([1, 536, 19, 34]).cuda()
        f2=torch.rand([1, 536, 38, 68]).cuda()
        f3=torch.rand([1, 536, 76, 136]).cuda()
        features=[f1,f2,f3]
        loss = yolo_head(features,targets)
        print(loss)
    def test_yolo_pred(self):
        net, yolo_def = extract_net_and_yolo_param("../model_cfg/yolov3.cfg")
        yolo_head = YoloHead(net, yolo_def)
        yolo_head.cuda()
        yolo_head.eval()
        f1 = torch.rand([1, 536, 19, 34]).cuda()
        f2 = torch.rand([1, 536, 38, 68]).cuda()
        f3 = torch.rand([1, 536, 76, 136]).cuda()
        features = [f1, f2, f3]
        pred = yolo_head(features)
        print(pred)


if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=2)
