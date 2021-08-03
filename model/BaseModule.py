import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_util import *
import math
import numpy as np
import Logger
from typing import Union, List, Tuple
from torchvision.ops import nms
from util import xywh2xyxy
import pynvml
pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

batch_norm=nn.BatchNorm2d

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                after_bn = batch_norm(filters)
                modules.add_module('batch_norm_%d' % i, after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1.
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight)
                nn.init.zeros_(after_bn.bias)
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            # put the head you want
            modules.add_module('yolo_%d' % i, EmptyLayer())

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YoloBackbone(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_dict):
        """
        cfg_dict: yolo configuration file (str)
        nid number of id
        test_emb test situation
        """
        super(YoloBackbone, self).__init__()
        if isinstance(cfg_dict, str):
            cfg_dict = parse_model_cfg(cfg_dict)  #move model config to dict
        self.module_defs = cfg_dict
        self.hyperparams, self.module_list = create_modules(self.module_defs)  # create modules

    def forward(self, x):
        """
        forward:
        targets : the thing you need to train
        targets_len : equal to the length of targets
        """
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module(x)
                output.append(x)
            layer_outputs.append(x)
        return output #分列表输出tensor，实现FPN结构


#   output
#   xywh,confidence, embeddings(which are used in the JDE tracker.update to update tracklets)
#   "In the original code in the MOT-matser, the various prediction heads are embedded in the darknet"

def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, nID, nE):
        """
        构建一个 yolo layer

        :param anchors:
            predifined anchors for this yolo layer
        :param nC:
            number of classes
        :param nID:
            number of IDs
        :param nE:
            dimension of embeddings
        """
        super(YOLOLayer, self).__init__()
        # self.layer = yolo_layer
        nA = len(anchors)
        if not isinstance(anchors, torch.Tensor):
            self.anchors = torch.tensor(anchors).float()
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.nID = nID  # number of identities
        self.nGh, self.nGw = 0, 0 # the height and width of the grid
        self.img_size = 0
        self.emb_dim = nE
        self.shift = [1, 3, 5]
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.classLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))  # -2.3
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1) if self.nID > 1 else 1

    def forward(self, p_cat, img_size, targets=None, classifier=None,class_classifier=None,test_emb=False):
        """

        :param p_cat:
        :param img_size:
        :param targets: (N, 7)
            (batch_id, cls, id, x, y, w, h)
        :param classifier:
        :param test_emb:
        :return:
        """
        local_device = p_cat.device

        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        self.nB, self.nGh, self.nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.anchors.device != p_cat.device:
            # move all the parameters to the same device
            self.anchors = self.anchors.to(p_cat.device)

        if self.img_size != img_size:
            # build yolo grids here
            self.create_grids(img_size)
            self.img_size = img_size

        p = p.view(self.nB, self.nA, self.nC + 5, self.nGh, self.nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        Logger.logger.logger.debug("p.size {}".format(p.shape))
        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        # Training
        #target class

        if targets is not None:
            if test_emb:
                tconf, tbox, tids = self.build_targets_max(targets)
            else:
                tconf, tbox, tids,tcls = self.build_targets_thres(targets)
                Logger.logger.logger.debug("tconf.size {}".format(tconf.shape))
            tconf, tbox, tids,tcls = tconf, tbox, tids,tcls
            mask = tconf > 0

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()

            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                lbox, lconf = torch.tensor(0., device=local_device), torch.tensor(0., device=local_device)
            lconf = self.SoftmaxLoss(p_conf, tconf)

            lid = torch.tensor(0., device=local_device).squeeze()
            lclass = torch.tensor(0., device=local_device).squeeze()
            emb_mask, _ = mask.max(1)

            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            #current tid [1, 4, 19, 34, 1]
            tids, _ = tids.max(1)
            tcls,_=tcls.max(1)
            #now tid [1, 19, 34, 1]
            #eliminate mask for testing
            #tids = tids[emb_mask]
            #embedding = p_emb[emb_mask].contiguous()
            embedding = p_emb
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()

            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros((0, self.emb_dim + 1), device=local_device)
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt

            if len(embedding) > 0:
                embedding = embedding
                #print("embedding.device", embedding.device)
                print(meminfo.used)

                logits = classifier(embedding)
                class_infer=class_classifier(embedding)
                lid = self.IDLoss(logits.permute((0,3,1,2)), tids.squeeze(dim=-1))
                lclass=self.classLoss(class_infer.permute((0,3,1,2)),tcls.squeeze(dim=-1))

            # Sum loss components
            lbox, lconf, lid,lclass = lbox, lconf, lid,lclass
            loss = torch.exp(-self.s_r) * lbox + torch.exp(-self.s_c) * lconf + torch.exp(-self.s_id) * (lid+lclass) + \
                   (self.s_r + self.s_c + self.s_id)

            loss *= 0.5
            loss_dict = dict(
                loss=loss.cpu().item(),
                lbox=lbox.cpu().item(),
                lconf=lconf.cpu().item(),
                lid=lid.cpu().item(),
                lclass=lclass.cpu().item(),
                nT=nT
            )

            return loss, loss_dict

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1).contiguous(), dim=-1)
            p_cls=class_classifier(p_emb)
            # p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            # p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            # TODO 这里全部预测为 0 类，需要更改。
            #p_cls = torch.zeros((self.nB, self.nA, self.nGh, self.nGw, 1), device=local_device)  # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            # p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(self.nB, -1, p.shape[-1])

    def create_grids(self, img_size):
        """
        Create grids yolo for given size

        :param self:
            the yolo layer object
        :param img_size:
            original image size
        :return:
        """
        self.stride = img_size[0] / self.nGw
        assert self.stride == img_size[1] / self.nGh, \
            "{} v.s. {}/{} please keep them the same".format(self.stride, img_size[1], self.nGh)

        # build xy offsets
        grid_x = torch.arange(self.nGw, device=self.s_c.device).repeat((self.nGh, 1)).view((1, 1, self.nGh, self.nGw)).float()
        grid_y = torch.arange(self.nGh, device=self.s_c.device).repeat((self.nGw, 1)).transpose(0, 1).view((1, 1, self.nGh, self.nGw)).float()
        # grid_y = grid_x.permute(0, 1, 3, 2)
        self.grid_xy = torch.stack((grid_x, grid_y), 4)  # (1, 1, nGh, nGw, 1)

        # build wh gains
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(self.nA, 2)

    def build_targets_max(self, target):
        """

        :param targets: (N, 7)
            (batch_id, cls, id, x, y, w, h)
        :return:
        """
        txy = torch.zeros((self.nB, self.nA, self.nGh, self.nGw, 2), device=self.anchor_wh.device)  # batch size, anchors, grid size
        twh = torch.zeros((self.nB, self.nA, self.nGh, self.nGw, 2), device=self.anchor_wh.device)
        tconf = torch.zeros((self.nB, self.nA, self.nGh, self.nGw), device=self.anchor_wh.device).long()
        tcls = torch.zeros((self.nB, self.nA, self.nGh, self.nGw, self.nC), device=self.anchor_wh.device).byte()  # nC = number of classes
        tid = -1 * torch.ones((self.nB, self.nA, self.nGh, self.nGw, 1), device=self.anchor_wh.device).long()
        for b in range(self.nB):
            t = target[target[:, 0] == b] # select targets in this batch
            t_id = t[:, 2].clone().long()
            t = t[:, [1, 3, 4, 5, 6]]
            nTb = len(t)  # number of targets
            if nTb == 0:
                continue

            gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
            gxy[:, 0] = gxy[:, 0] * self.nGw
            gxy[:, 1] = gxy[:, 1] * self.nGh
            gwh[:, 0] = gwh[:, 0] * self.nGw
            gwh[:, 1] = gwh[:, 1] * self.nGh
            gi = torch.clamp(gxy[:, 0], min=0, max=self.nGw - 1).long()
            gj = torch.clamp(gxy[:, 1], min=0, max=self.nGh - 1).long()

            # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
            # gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
            # gi, gj = gxy.long().t()

            # iou of targets-anchors (using wh only)
            box1 = gwh
            box2 = self.anchor_wh.unsqueeze(1)
            inter_area = torch.min(box1, box2).prod(2)
            iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

            # Select best iou_pred and anchor
            iou_best, a = iou.max(0)  # best anchor [0-2] for each target

            # Select best unique target-anchor combinations
            if nTb > 1:
                _, iou_order = torch.sort(-iou_best)  # best to worst

                # Unique anchor selection
                u = torch.stack((gi, gj, a), 0)[:, iou_order]
                # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
                first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative
                i = iou_order[first_unique]
                # best anchor must share significant commonality (iou) with target
                i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
                if len(i) == 0:
                    continue

                a, gj, gi, t = a[i], gj[i], gi[i], t[i]
                t_id = t_id[i]
                if len(t.shape) == 1:
                    t = t.view(1, 5)
            else:
                if iou_best < 0.60:
                    continue

            tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
            gxy[:, 0] = gxy[:, 0] * self.nGw
            gxy[:, 1] = gxy[:, 1] * self.nGh
            gwh[:, 0] = gwh[:, 0] * self.nGw
            gwh[:, 1] = gwh[:, 1] * self.nGh

            # XY coordinates
            txy[b, a, gj, gi] = gxy - gxy.floor()

            # Width and height
            twh[b, a, gj, gi] = torch.log(gwh / self.anchor_wh[a])  # yolo method
            # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

            # One-hot encoding of label
            tcls[b, a, gj, gi, tc] = 1
            tconf[b, a, gj, gi] = 1
            tid[b, a, gj, gi] = t_id.unsqueeze(1)
        tbox = torch.cat([txy, twh], -1)
        return tconf, tbox, tid

    def build_targets_thres(self, target):
        ID_THRESH = 0.5
        FG_THRESH = 0.5
        BG_THRESH = 0.4
        assert (len(self.anchor_wh) == self.nA)

        tbox = torch.zeros((self.nB, self.nA, self.nGh, self.nGw, 4), device=self.anchor_wh.device)  # batch size, anchors, grid size
        tconf = torch.zeros((self.nB, self.nA, self.nGh, self.nGw), device=self.anchor_wh.device).long()
        tid = -1 * torch.ones((self.nB, self.nA, self.nGh, self.nGw, 1), device=self.anchor_wh.device).long()
        tclass = -1 * torch.ones((self.nB, self.nA, self.nGh, self.nGw, 1), device=self.anchor_wh.device).long()
        for b in range(self.nB):
            t = target[b,:]
            t_id = t[:, 1].clone().long()
            t_class=t[:, 0].clone().long()
            t = t[:, [0, 2, 3, 4, 5]]
            nTb = len(t)  # number of targets
            if nTb == 0:
                continue

            gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
            gxy[:, 0] = gxy[:, 0] * self.nGw
            gxy[:, 1] = gxy[:, 1] * self.nGh
            gwh[:, 0] = gwh[:, 0] * self.nGw
            gwh[:, 1] = gwh[:, 1] * self.nGh
            gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=self.nGw - 1)
            gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=self.nGh - 1)

            gt_boxes = torch.cat([gxy, gwh], dim=1).cuda()  # Shape Ngx4 (xc, yc, w, h)

            anchor_mesh = generate_anchor(self.nGh, self.nGw, self.anchor_wh)
            anchor_list = anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # Shpae (nA x nGh x nGw) x 4
            # print(anchor_list.shape, gt_boxes.shape)
            iou_pdist = bbox_iou(anchor_list, gt_boxes)  # Shape (nA x nGh x nGw) x Ng
            iou_max, max_gt_index = torch.max(iou_pdist, dim=1)  # Shape (nA x nGh x nGw), both

            iou_map = iou_max.view(self.nA, self.nGh, self.nGw)
            gt_index_map = max_gt_index.view(self.nA, self.nGh, self.nGw)

            # nms_map = pooling_nms(iou_map, 3)

            id_index = iou_map > ID_THRESH
            cl_index=iou_map>ID_THRESH
            fg_index = iou_map > FG_THRESH
            bg_index = iou_map < BG_THRESH
            ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
            tconf[b][fg_index] = 1
            tconf[b][bg_index] = 0
            tconf[b][ign_index] = -1

            gt_index = gt_index_map[fg_index]
            gt_box_list = gt_boxes[gt_index]
            gt_id_list = t_id[gt_index_map[id_index]]
            gt_cl_list=t_class[gt_index_map[cl_index]]
            Logger.logger.logger.debug("gt_index shape {}; gt_index_map shape {}; gt_boxes shape {}".format(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape))

            if torch.sum(fg_index) > 0:
                tid[b][id_index] = gt_id_list.unsqueeze(1)
                tclass[b][id_index] = gt_cl_list.unsqueeze(1)
                fg_anchor_list = anchor_list.view(self.nA, self.nGh, self.nGw, 4)[fg_index]
                delta_target = encode_delta(gt_box_list, fg_anchor_list)
                tbox[b][fg_index] = delta_target
        return tconf, tbox, tid,tclass


class YoloHead(nn.Module):
    '''
    Yolo Head Module
    '''
    def __init__(self, config, yolo_defs):
        """
        Constructor

        :param config dict: (this dict is also generated from the .cfg file under the `[net]` block)
            `nID`: number of IDs
            `width`: width of the image
            `height`: height of the image
            `embedding_dim`: dimensions of the embedding
            `conf_threshold`: threshold for confidence score
            `nms_thres`: NMS threshold (IoU Threshold)
        :param yolo_defs:
            this dictionary is generated from the yolo configuration file: eg yolov3.cfg
            please refer to function YoloTrainModel.extract_net_and_yolo_param for more details
        """
        super(YoloHead, self).__init__()
        self.yolo_list = nn.ModuleList()
        self.nID = int(config['nID'])
        self.classifier = nn.Linear(int(config['embedding_dim']),int(config['nID']))
        self.class_classifier=nn.Linear(int(config['embedding_dim']),int(config['classes']))
        self.img_size = (int(config['width']), int(config['height']))
        self.conf_thres = config.get('conf_threshold', 0.5)
        self.nms_thres = config.get('nms_threshold', 0.7)

        for defs in yolo_defs:
            anchor_idxs = [int(x) for x in defs['mask'].split(',')]

            # Extract anchors
            anchors = [float(x) for x in defs['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]

            nC = int(defs['classes'])
            yolo_layer = YOLOLayer(anchors, nC, int(config['nID']),
                                   int(config['embedding_dim']))
            self.yolo_list.append(yolo_layer)

    def non_max_suppression(self, prediction, method='standard'):
        """
        Removes detections with lower object confidence score than 'conf_thres'
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        Args:
            prediction,
            conf_thres,
            nms_thres,
            method = 'standard' or 'fast'
        """

        output = [None for _ in range(len(prediction))]
        for image_i, pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            # Get score and class with highest confidence

            v = pred[:, 4] > self.conf_thres
            v = v.nonzero().squeeze()
            if len(v.shape) == 0:
                v = v.unsqueeze(0)

            pred = pred[v]

            # If none are remaining => process next image
            nP = pred.shape[0]
            if not nP:
                continue
            # From (center x, center y, width, height) to (x1, y1, x2, y2)
            pred[:, :4] = xywh2xyxy(pred[:, :4])

            # Non-maximum suppression
            if method == 'standard':
                nms_indices = nms(pred[:, :4], pred[:, 4], self.nms_thres)
            elif method == 'fast':
                nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=self.nms_thres, conf_thres=self.conf_thres)
            else:
                raise ValueError('Invalid NMS type!')
            det_max = pred[nms_indices]

            if len(det_max) > 0:
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

            # 只选取前 nID 个结果
            if output[image_i] is not None and len(output[image_i]) > self.nID:
                output[image_i] = output[image_i][:self.nID]

        return output

    def forward(self, features, targets=None, test_emb=False):
        """
        :param features:
            List[torch.Tensor] each tensor is extracted from the corresponding position in the backbone
        :param img_size:
            (h, w) the height and width of the original image
        :param targets: (N, 7)
            targets of the images, in training process
            (batch_id, cls, id, x, y, w, h)
        :param test_emb:
            test embedding output, reserved
        :return:
            for training, it returns
                loss, loss_dict
            loss is the total loss of this model,
            loss_dict is a disctionary with all the loss items in side (for logging)

            for inference, it returns
                B, N  all the predictions
                (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """
        if self.training:
            assert targets is not None, "Training Mode without targets!"
            total_loss = 0
            loss_dict = {}
            for f, layer in zip(features, self.yolo_list):
                loss, loss_dict = layer(f, self.img_size, targets, self.classifier,self.class_classifier, test_emb)
                total_loss += loss
                for name, value in loss_dict.items():
                    v_update = loss_dict.get(name, 0.)
                    v_update += value
                    loss_dict[name] = v_update
            return total_loss, loss_dict
        # inference
        else:
            output = []
            for f, layer in zip(features, self.yolo_list):
                pred = layer(f, self.img_size,targets=None, classifier=None,class_classifier=self.class_classifier)
                output.append(pred)

            pred = self.non_max_suppression(torch.cat(output, 1))
            return pred



