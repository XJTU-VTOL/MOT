import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from torchvision.ops import box_iou
from .util import find_id


class DetectionMetric:
    """
    Author: 雷博书
    """
    def __init__(self, num_cls:int = 17, IoUs:float = 0.1):
        """
        :param num_cls:
            所有类别数量
        :param IoUs:
            IoU 阈值
        :param conf_threshold:
            置信度阈值
        """
        self.iou = IoUs
        self.classes = {}

        for i in range(num_cls):
            self.classes[i] = np.zeros((3, ))  # TP, FN, FP

    def update(self, pred, target):
        """
        Update the metric object in one step

        :param pred (torch.Tensor) (N, 6):
            (x1, y1, x2, y2, cls)
        :param target (torch.Tensor) (M, 5):
            (x1, y1, x2, y2, cls)
        :return:
        """
        # ''' Step1 Filter by conf threshold '''
        # conf = pred[:, 4]
        # pred = pred[conf > self.conf_theshold]

        ''' Step 2 Calculate IoU '''
        if(len(pred)==0):
            return
        print("pred",pred)
        cls_id = torch.unique(target[:, 4])
        for cls in cls_id:
            cls = int(cls.item())
            print("cls",cls)
            #print("pred",pred)
            pred_cls = pred[pred[:, 4] == cls]
            target_cls = target[target[:, 4] == cls]

            if len(pred_cls) == 0 or len(pred_cls[0])==1:
                print("0 pred_cls")
                continue

            print("pred_cls", pred_cls)
            pred_cls_tlbr = pred_cls[:, :4]  # (x1, y1, x2, y2)
            target_cls_tlbr = target_cls[:, :4]
            print("pred_cls_tlbr",pred_cls_tlbr)
            print("target_cls_tlbr",target_cls_tlbr)
            IoU = box_iou(target_cls_tlbr, pred_cls_tlbr)
            print("IOU",IoU)
            IoU_numpy = IoU.cpu().numpy()
            try:
                row_ind, col_ind = linear_sum_assignment(IoU_numpy, maximize=True)
                select_IoU = IoU[row_ind, col_ind]

                TP = np.sum(np.where(select_IoU.cpu() > self.iou, 1, 0))
                FN = target_cls.shape[0] - TP
                FP = pred_cls.shape[0] - TP
                print("detection_metric: ", "TP ", TP, "FN ", FN, "FP  ", FP)
                self.classes[cls] += np.array([TP, FN, FP])

            except:
                print("IoU_numpy",IoU_numpy)

    def compute(self):
        """
        计算各个类的准确度和召回率

        :return:
            Acc: Dict 类别：准确率
            Recall: Dict 类别：召回率
        """
        Accuracy = {}
        Recall = {}

        for key, stat in self.classes.items():
            TP, FN, FP = stat


            try:
                Accuracy[key] = TP / (TP + FP)
            except ZeroDivisionError:
                Accuracy[key] = 0.

            try:
                Recall[key] = TP / (TP + FN)
            except ZeroDivisionError:
                Recall[key] = 0.

        return Accuracy, Recall


class APCurve:
    """
    绘制检测的 AP 曲线。

    """
    def __init__(self, num_cls:int = 10, IoUs:np.ndarray = None):
        """
        :param num_cls:
            类别数
        :param IoUs:
            选取采用的 IoU 阈值
        """
        self.ious = np.linspace(0.7, 0.95, 10) if IoUs is None else IoUs
        self.metrics = []
        for iou in self.ious:
            self.metrics.append(DetectionMetric(num_cls, iou))

    def update(self, pred, target):
        """
        使用同 DetectionMetric 类
        :param pred:
        :param target:
        :return:
        """
        for m in self.metrics:
            m.update(pred, target)

    def cls_ap(self, cls_id = 0):
        """
        得到对应类的 AP 曲线

        :param cls_id:
            类别
        :return:
            acc: List[float] 准确度
            recall: List[recall] 召回率
        """
        acc = []
        recall = []
        for m in self.metrics:
            all_acc, all_recall = m.compute()
            acc.append(all_acc[cls_id])
            recall.append(all_recall[cls_id])

        return acc, recall

    def mean_cls_ap(self):
        """
        对所有类别求平均

        :return:
        """
        mean_acc = []
        mean_recall = []

        for m in self.metrics:
            all_acc, all_recall = m.compute()
            all_acc_array = []
            all_recall_array = []
            for a, r in zip(all_acc.values(), all_recall.values()):
                all_acc_array.append(a)
                all_recall_array.append(r)

            mean_acc.append(np.mean(np.array(all_acc_array)))
            mean_recall.append(np.mean(np.array(all_recall_array)))

        return mean_acc, mean_recall

class TrackMetric:
    """
    追踪评价指标

    """
    def __init__(self, num_cls: int = 10, IoUs: float = 0.5):
        self.ious = IoUs
        self.pred_classes = {}
        self.target_classes = {}

        self.classes = {}
        for i in range(num_cls):
            self.classes[i] = np.zeros((3, ))  # TP, FN, FP

        self.frame_id = 0

    def update(self, pred, target):
        """
        Update the metric object in one step

        :param pred (torch.Tensor) (N, 7):
            (frame_id, x1, y1, x2, y2, cls, id)
        :param target (torch.Tensor) (M, 7):
            (frame_id, x1, y1, x2, y2, cls, id)
        :return:
        """
        self.frame_id += 1

        ''' Step1 Filter by conf threshold '''
        # conf = pred[:, 5]
        # pred = pred[conf > self.conf_theshold]

        ''' Step 2 Record All Boxes '''
        cls_id = torch.unique(target[:, 5])
        for cls in cls_id:
            cls = int(cls.item())

            if len(pred.shape) == 2:
                pred_cls = pred[pred[:, 5] == cls]
            else:
                pred_cls = torch.ones((0, 7), device=pred.device, dtype=pred.dtype)
            track_hist = self.pred_classes.get(cls, [])
            track_hist.append(pred_cls)
            self.pred_classes[cls] = track_hist

            if len(target) == 2:
                target_cls = target[target[:, 5] == cls]
            else:
                target_cls = torch.ones((0, 7), device=target.device, dtype=target.dtype)
            track_hist = self.target_classes.get(cls, [])
            track_hist.append(target_cls)
            self.target_classes[cls] = track_hist

    def compute(self):
        """
        Compute Accuracy and Recall in this video sequence and record TP FP FN

        Usage:
        # read sequence 1
        metric.compute()

        # read sequence 2
        # This will report stats in both sequence 1 and 2
        metric.compute()

        :return:
        """
        Accuracy = {}
        Recall = {}

        for cls in self.pred_classes.keys():
            pred_track = self.pred_classes[cls]  # List[torch.Tensor]
            target_track = self.target_classes[cls]

            pred_track = torch.cat(pred_track, dim=0)
            all_pred_id = pred_track[:, 6].int()
            pred_id = torch.unique(all_pred_id, sorted=True)
            num_pred_id = len(pred_id)

            target_track = torch.cat(target_track, dim=0)
            all_target_id = target_track[:, 6].int()
            target_id = torch.unique(all_target_id, sorted=True)
            num_target_id = len(target_id)

            frame_pred_stat = []
            frame_target_stat = []
            frame_IoU_stat = []

            pred_padding_num = 0
            target_padding_num = 0
            for f in range(1, self.frame_id+1):  # from frame 1 to self.frame_id
                pred_boxes = torch.zeros((num_pred_id, 4), device=pred_track.device, dtype=pred_track.dtype)
                target_boxes = torch.zeros((num_target_id, 4), device=pred_track.device, dtype=pred_track.dtype)

                frame_idx = pred_track[:, 0].int()
                cur_frame_pred = pred_track[frame_idx == f]  # select current frame
                cur_frame_pred_id = cur_frame_pred[:, 6].int()  # select idx
                if len(cur_frame_pred_id) > 0:
                    match = find_id(pred_id, cur_frame_pred_id)
                    pred_padding_num += (num_pred_id - len(match))
                    pred_boxes[match] = cur_frame_pred[:, 1:5]
                frame_pred_stat.append(pred_boxes)

                frame_idx = target_track[:, 0].int()
                cur_frame_target = target_track[frame_idx == f]  # select current frame
                cur_frame_target_id = cur_frame_target[:, 6].int()  # select idx
                if len(cur_frame_target_id) > 0:
                    match = find_id(target_id, cur_frame_target_id)
                    target_padding_num += (num_target_id - len(match))
                    target_boxes[match] = cur_frame_target[:, 1:5]
                frame_pred_stat.append(target_boxes)

                IoU = box_iou(target_boxes, pred_boxes) # (M, N)
                frame_IoU_stat.append(IoU)

            IoU_all = torch.stack(frame_IoU_stat, dim=0)
            IoU = torch.mean(IoU_all, dim=0)

            IoU_numpy = IoU.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(IoU_numpy, maximize=True)
            matched_iou = IoU_all[:, row_ind, col_ind].permute(1, 0)  # (M, B)

            matches, frame = matched_iou.shape
            TP = torch.where(matched_iou > self.ious, 1, 0).sum().item()
            FP = num_pred_id * frame - TP - pred_padding_num
            FN = num_target_id * frame - TP - target_padding_num

            self.classes[cls] += np.array([TP, FN, FP])

            TP = self.classes[cls][0]
            FN = self.classes[cls][1]
            FP = self.classes[cls][2]

            Accuracy[cls] = TP / (TP + FP)
            Recall[cls] = TP / (TP + FN)

        self.target_classes.clear()
        self.pred_classes.clear()

        return Accuracy, Recall
