import unittest
from model.BaseModule import YoloBackbone
import torch
from torchvision.transforms import transforms as T
from model.model_util import parse_model_cfg
from model.BaseModule import YoloHead
import torch.nn as nn
import json
from model.model_util import extract_net_and_yolo_param
from dataset.dataset import *
'''
TODO:backbone don't need to train? no losses?
'''
class module_test(unittest.TestCase):

    def test_backbone(self, return_result=False):
        pass
        cfg_file = "../model_cfg/yolov3.cfg"
        backbone = YoloBackbone(cfg_file).cuda().train()
        trainset_paths = {"MIX": "E://MOT_lyh//MOT_lyh//MOT//dataset//test.train"}
        dataset_root = "E://multi-object//MOT_lyh//MOT"

        transforms = T.Compose([T.ToTensor()])
        # Get dataloader
        dataset = JointDataset(dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,

                                                 num_workers=1, pin_memory=True, drop_last=True, collate_fn=collate_fn)

        '''
        JointDataset returns imgs, labels, img_path, (h, w)
        labels :[class] [identity] [x_center] [y_center] [width] [height]
        '''
        for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
                #print(targets)
                # imgs, targets, _, _, targets_len=j

                data = imgs.cuda()
                results = backbone(data)
                for f in results:
                        print("results in :", f.shape)

                if return_result:
                    return results,targets
                break

    def test_yolo(self):
        net, yolo_def = extract_net_and_yolo_param("../model_cfg/yolov3.cfg")
        yolo_head = YoloHead(net, yolo_def)
        yolo_head.cuda()
        yolo_head.train()
        '''
        targets = torch.tensor([
            [0, 1, 0, 1, 1, 1, 1],
        ]).cuda()
        '''
        features,targets = self.test_backbone(True)
        losses = yolo_head(features,targets)
        print(losses)

if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.test_yolo(verbosity=2)
