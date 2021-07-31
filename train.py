import pytorch_lightning as pl
import yaml
import argparse
from model import model_dict

def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--cfg", type=str, default="train_cfg/yolo.yaml")
    arg = pl.Trainer.add_argparse_args(arg)
    opt = arg.parse_args()

    f = open(opt.cfg, 'r', encoding='utf-8')
    cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = model_dict[cfg_dict["name"]](opt, cfg_dict["cfg"])

    trainer = pl.Trainer.from_argparse_args(opt)
    trainer.fit(model)


if __name__=='__main__':
    main()
