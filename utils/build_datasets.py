import opendatasets as od
import os
import shutil
import argparse
import random


def build_bowl_2018(download, target):
    od.download("https://www.kaggle.com/competitions/data-science-bowl-2018", data_dir=download)
    from zipfile import ZipFile
    os.makedirs(f'{download}/data-science-bowl-2018/stage1_train')
    with ZipFile(f'{download}/data-science-bowl-2018/stage1_train.zip', 'r') as z:
        z.extractall(path=f'{download}/data-science-bowl-2018/stage1_train')
    items = os.listdir(f'{download}/data-science-bowl-2018/stage1_train')
    if os.path.exists(f'{download}/data-science-bowl-2018/stage1_valid'):
            remove_folder_content(f'{download}/data-science-bowl-2018/stage1_valid')
            shutil.rmtree(f'{download}/data-science-bowl-2018/stage1_valid')
    if not(os.path.exists(f'{download}/data-science-bowl-2018/stage1_valid')):
        os.mkdir(f'{download}/data-science-bowl-2018/stage1_valid')         
    for item in random.sample(items, k=70):
        shutil.copytree(f'{download}/data-science-bowl-2018/stage1_train/{item}'
                        , f'{download}/data-science-bowl-2018/stage1_valid/{item}')    
    shutil.copytree(f'{download}/data-science-bowl-2018/stage1_train'
                    , f'{target}/train')
    shutil.copytree(f'{download}/data-science-bowl-2018/stage1_valid'
                , f'{target}/dev')
    shutil.copytree(f'{download}/data-science-bowl-2018/stage1_valid'
                , f'{target}/test') # Test sets is the same as dev test.

    
def build_isic_2017(download, target):
    od.download('https://www.kaggle.com/datasets/mnowak061/isic2017-256x256-jpeg', data_dir=download)
    shutil.copytree(f'{download}/isic2017-256x256-jpeg/ISIC_2017_256x256/train'
                        , f'{target}/train')
    shutil.copytree(f'{download}/isic2017-256x256-jpeg/ISIC_2017_256x256/valid'
                    , f'{target}/dev')
    shutil.copytree(f'{download}/isic2017-256x256-jpeg/ISIC_2017_256x256/test'
                    , f'{target}/test')
    

def remove_folder_content(folder):
    if os.listdir(folder) == []:
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
    return None     
    
        
def get_args():
    parser = argparse.ArgumentParser(description='Split raw data into train/dev/test sets.')
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["bowl 2018", "isic 2017"]
                        , help='choose between datasets')
    parser.add_argument("--datapath", type=str
                        , default="./data/processed", help='target datapath')
    parser.add_argument("--downfolder", type=str
                        , default="./data/raw", help='download folder')
    return parser.parse_args()


# nguyendqdai - f56dd07d7a52ae4dab0008b690945b87
if __name__ == "__main__":
    args = get_args()
    datapath = args.datapath
    dl_folder = args.downfolder
    for _, dir in enumerate([f'{datapath}/train', f'{datapath}/dev', f'{datapath}/test']):
        if os.path.exists(dir):
            remove_folder_content(dir)
            shutil.rmtree(dir)
    if args.dataset == "bowl 2018":
        build_bowl_2018(dl_folder, datapath)
    elif args.dataset == "isic 2017":
        build_isic_2017(dl_folder, datapath)
        