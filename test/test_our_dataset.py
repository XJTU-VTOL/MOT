import argparse
import json
import time
from time import gmtime, strftime
import unittest
from torchvision.transforms import transforms as T
import torch
from dataset.dataset import *
class Test_RealDataset(unittest.TestCase):
    def test_D(self):
        trainset_paths ={"MIX":"E://MOT_lyh//MOT_lyh//MOT//dataset//test.train"}
        dataset_root = "D://multi-object//MOT_lyh//MOT"


        transforms = T.Compose([T.ToTensor()])
        # Get dataloader
        dataset = JointDataset(dataset_root, trainset_paths, (1088,608), augment=True, transforms=transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,

                                              num_workers=1, pin_memory=True, drop_last=True, collate_fn=collate_fn)

        '''
        JointDataset returns imgs, labels, img_path, (h, w)
        labels :[class] [identity] [x_center] [y_center] [width] [height]
        '''
        for i,  (imgs, targets, _, _, targets_len) in enumerate(dataloader):

            print(targets)
            #imgs, targets, _, _, targets_len=j


if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=2)
