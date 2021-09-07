
import torch
from torch import nn
import pytorch_lightning as pl
import yaml
import argparse
from model import model_dict
from model.yolo import  YoloTrainModel
def test_yolo_train():
    arg = argparse.ArgumentParser()
    arg.add_argument("--cfg", type=str, default="train_cfg/yolo.yaml")
    arg = pl.Trainer.add_argparse_args(arg)
    opt = arg.parse_args()

    f = open(opt.cfg, 'r', encoding='utf-8')
    cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    PATH="E:\\雷博书\\MOT\\lightning_logs\\version_254\\checkpoints\\epoch=19-step=32799.ckpt"
    new_model =model_dict[cfg_dict["name"]].load_from_checkpoint(checkpoint_path=PATH,opt=opt,config=cfg_dict["cfg"])
    trainer = pl.Trainer(gpus=1)
    test_dataloader=new_model.val_dataloader()
    trainer.test( new_model, test_dataloaders=test_dataloader)

if __name__=='__main__':
    test_yolo_train()
