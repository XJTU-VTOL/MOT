import glob
import math

import os.path as osp
import random
import time
from collections import OrderedDict
import  xml.dom.minidom

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

transforms = T.Compose([T.ToTensor()])
def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def parser_for_xml(path):
    '''

    :param path: annotations path
    :return: [[14, -1, 978, 215, 304, 272]]
               cls，id,x,y,x,y

    '''
    # 读取文件
    dom = xml.dom.minidom.parse(path)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取 student
    objects = data.getElementsByTagName('object')
    ans = []
    for object in objects:
        cls = object.getElementsByTagName("name")[0].childNodes[0].nodeValue
        cls = str(cls)
        if ("_"in cls):
            num=cls.split("_")
            id=eval(num[-1])
        else:
             id=-1

        cls = (ord(cls[1]) - 65)*4 + ord(cls[4]) - ord('0')

        xmin = object.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
        xmin = eval(xmin)
        xmax = object.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
        xmax = eval(xmax)
        ymin = object.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
        ymin = eval(ymin)
        ymax = object.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        ymax = eval(ymax)
        ans.append([cls,id, xmin, ymin, xmax, ymax])
    ans = np.array(ans,'float32')

    return ans


'''
 JointDataset would allow you to gather data from various datasets in a mixed order while LoadImagesAndLabels can only assign data by the order your files are placed
'''



class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):

        '''
                :param paths:  the path of .train file  view datatest/test.train for example  to generate .train file run dataset/walk.py
                :param img_size: the size of input img
                :param augment: whether to use data augmentation
                :param transforms: the function in torchvision.transforms, usually we only use ToTensor(). This is to change the numpy array pictures to tensors
        '''
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path):
        img_path=Path(img_path)
        label_path=Path(label_path)
        height = self.height
        width = self.width
        img = cv2.imread(str(img_path))  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = False


        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if (label_path.is_file()):
            labels0 = parser_for_xml(str(label_path))
            #print(labels0)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] =  (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] =  (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] =  (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])



        plotFlag = False
        if plotFlag:
            import matplotlib
            #matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)


        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height



        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)
    '''
    img: images in tensor form if transformed else in np array
    labels:cls,id,x,y,w,h in list 
    img_path:the path for the output images
    (h,w): the height and width for the output images
    '''

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [l for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root,paths, img_size=(1088, 608), augment=False, transforms=transforms):
        '''
        :param root: the root of your dataset
        :param paths: {"key":"the path of .train file"} view datatest/test.train for example  to generate .train file run dataset/walk.py
        :param img_size: the size of input img
        :param augment: whether to use data augmentation
        :param transforms: the function in torchvision.transforms, usually we only use ToTensor(). This is to change the numpy array pictures to tensors
        '''
        dataset_names = paths.keys()
        root=Path(root)
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [str(root.joinpath(Path(x.replace('\n','')))) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(str(x)) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('img', 'ann').replace('.png', '.xml').replace('.jpg', '.xml')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                #lb = parser_for_xml(lp) for out xml annotation
                lb=parser_for_xml(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):
        """
        Iterator function for train dataset
        """
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c
        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        labels = torch.from_numpy(labels).float()
        return imgs, labels, img_path, (h, w)

    '''
        img: images in tensor form if transform function in input is not none else in np array
        labels:cls,id,left_up_x,left_up_y,right_down_x,right_down_y in list 
        img_path:the path for the output images
        (h,w): the height and width for the output images
     '''


