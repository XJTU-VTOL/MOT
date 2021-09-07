import torch

def collate_fn(*batch):
    batch_data = batch[0]
    images = []
    labels = []
    paths = []
    sizes = []
    for id, b in enumerate(batch_data):
        image, label, path, size = b
        num_label = len(label)
        batch_id = torch.ones((num_label, 1)) * id
        label = torch.cat([batch_id, label], dim=1)
        labels.append(label)
        images.append(image)
        paths.append(path)
        sizes.append(size)

    return torch.stack(images, dim=0), torch.cat(labels, dim=0), paths, sizes

from pathlib import Path
import random
def generate_train_file(dataset_path,save_name):
    #filePath = 'D:\\multi-object\\MOT_lyh\\MOT\\data'
    dataset_path=Path(dataset_path)
    f=open(save_name,'w')
    datakind=[]
    for path in dataset_path.iterdir():
        #print(i,j,k)
        #print(path)
        datakind.append(path)
    datasets=[]
    label=Path("img")
    for path in datakind:
        path=path.joinpath(label)
        datasets.append(path)
    for path in datasets:
        #print(path)
        for img in path.glob("*.jpg"):

            kind=Path(img.parent.parent.name)
            type=Path(img.parent.name)
            name=Path(img.name)
            img=kind.joinpath(type.joinpath(name))
            f.write(str(img)+"\n")
    f.close()
def generate_train_and_validation_file(dataset_path,train_file,val_file,ratio):
    """

    :param dataset_path: the path of your dataset
    :param train_file: the path to save .train file
    :param val_file:  the path to save .val file
    :param ratio: the ration of train files to validion file e.g. ration=3 means 3/4 are for training and 1/4 are for val
    :return: none
    """
    #filePath = 'D:\\multi-object\\MOT_lyh\\MOT\\data'
    dataset_path=Path(dataset_path)
    train=open(train_file,'w')
    val=open(val_file,'w')
    account=1/(1+ratio)
    datakind=[]
    for path in dataset_path.iterdir():
        #print(i,j,k)
        #print(path)
        datakind.append(path)
    datasets=[]
    data_all=[]
    label=Path("img")
    for path in datakind:
        path=path.joinpath(label)
        datasets.append(path)
    for path in datasets:
        #print(path)
        for img in path.glob("*.jpg"):

            kind=Path(img.parent.parent.name)
            type=Path(img.parent.name)
            name=Path(img.name)
            img=kind.joinpath(type.joinpath(name))
            data_all.append(img)
    total=len(data_all)
    thres=total*account
    for img in data_all:
        decide=random.randint(1,total)
        if(decide<=thres):
            val.write(str(img)+'\n')
        else:
            train.write(str(img)+'\n')
    val.close()
    train.close()

if(__name__=="__main__"):
    generate_train_and_validation_file('E:\\3d_detector\\Data2021\\MIX','..\\big.train','..\\big.val',2)