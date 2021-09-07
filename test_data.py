import pytorch_lightning as pl
import yaml
import argparse
from model import model_dict
import torch
from dataset.dataset import *
def main(img_path):
    img_size = (1088, 608)
    height=img_size[0]
    width=img_size[1]

    model=torch.load("E:\\雷博书\\MOT\\lightning_logs\\version_11\\checkpoints\\epoch=19-step=1219.ckpt")
    img = cv2.imread(str(img_path))
    img, ratio, padw, padh = letterbox(img, height=height, width=width)
    img = np.ascontiguousarray(img[:, :, ::-1])


if __name__=='__main__':
    main()


