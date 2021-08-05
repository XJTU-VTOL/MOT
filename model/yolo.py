import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from .BaseModule import YoloBackbone, YoloHead
from tracker.tracker import Tracker
from .model_util import parse_model_cfg, extract_net_and_yolo_param
from dataset.util import collate_fn
from dataset import ToyDataset
from dataset.dataset import JointDataset
from metric import TrackMetric
from typing import List, Union, Dict
from torch.utils.data import DataLoader
from util import xywh2xyxy

class YoloTrainModel(pl.LightningModule):
    def __init__(self, opt, config):
        super(YoloTrainModel, self).__init__()
        # raed hyper parameters from [net]
        self.hyperparameters, yolo_defs = extract_net_and_yolo_param(config["model_cfg"])
        self.config = config
        self.opt = opt

        # Build Backbone
        self.backbone = YoloBackbone(config["model_cfg"])

        self.head = YoloHead(self.hyperparameters, yolo_defs)
        self.tracker = Tracker()
        self.track_metric = TrackMetric()

        # 从 .cfg 文件获取信息
        self.config['dataset']['image_size'] = (int(self.hyperparameters['height']), int(self.hyperparameters['width']))

    def training_step(self, batch, batch_idx):
        images, targets, paths, sizes = batch
        features = self.backbone(images)
        loss, loss_item = self.head(features, targets)

        for key, val in loss_item.items():
            self.log(key, val, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.tracker.reset()

    def validation_step(self, batch, batch_idx):
        images, targets, paths, sizes = batch
        features = self.backbone(images)
        preds = self.head(features)
        for batch_id, frame in enumerate(preds):
            if frame is not None:
                frame = frame.cpu().numpy()
                tracked_result = self.tracker.update(frame)
                tracked_result = torch.tensor(tracked_result).to(targets.device).float()
            else:
                tracked_result = torch.ones((0, 7), device=targets.device)

            # 选取对应的 targets
            batch_target = targets[targets[:, 0] == batch_id]
            batch_target[:, 3:] = xywh2xyxy(batch_target[:, 3:])
            batch_target = batch_target[:, [3, 4, 5, 6, 1, 2]]
            frame_id = torch.ones((len(batch_target), 1), device=batch_target.device) * self.tracker.frame_id
            batch_target = torch.cat([frame_id, batch_target], dim=1)
            self.track_metric.update(tracked_result, batch_target)

    def validation_epoch_end(self, outputs):
        self.tracker.reset()
        Accuracy, Recall = self.track_metric.compute()

        for name, val in Accuracy.items():
            self.log("acc_"+str(name), val, on_epoch=True, on_step=False)

        for name, val in Recall.items():
            self.log("recall_"+str(name), val, on_epoch=True, on_step=False)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        dataset_config = self.config['dataset']
        dataset = JointDataset(dataset_config["root"], dataset_config["path"])
        dataloader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=collate_fn)

        return dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataset_config = self.config['dataset']
        dataset = JointDataset(dataset_config["root"], dataset_config["path"])
        dataloader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=collate_fn)

        return dataloader

    def configure_optimizers(self):
        lr = self.config.get("lr", 1e-3)
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, self.parameters()), lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(0.5 * self.opt.max_epochs), int(0.75 * self.opt.max_epochs)],
                                                         gamma=0.1)

        return [optimizer], [scheduler]