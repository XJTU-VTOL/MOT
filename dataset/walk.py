from pathlib import Path
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

if(__name__=="__main__"):
    generate_train_file('D:\\multi-object\\MOT_lyh\\MOT\\data','test.train')