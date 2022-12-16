import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class bowl_18_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.imageids = sorted(os.listdir(data_path))
        self.transform = transforms
    def __len__(self):
        return len(self.imageids)
    def __getitem__(self, index):
        idx = self.imageids[index]
        file_name = os.listdir(f'{self.data_path}/{idx}/images')[0]
        image = Image.open(f'{self.data_path}/{idx}/images/{file_name}')
        msks = os.listdir(f'{self.data_path}/{idx}/masks')
        mask = torch.zeros(1, image.size[1], image.size[0], dtype=torch.bool)
        for msk in msks:
            part_mask = Image.open(f'{self.data_path}/{idx}/masks/{msk}')
            part_mask = transforms.PILToTensor()(part_mask)
            part_mask = torch.gt(part_mask, 0)
            mask = torch.bitwise_or(mask, part_mask)
        mask = (mask.byte())*255
        mask = transforms.ToPILImage()(mask)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
 
 
class isic_17_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.img_path = f'{data_path}/images/images'
        self.msk_path = f'{data_path}/masks/masks'
        self.imgs = sorted(os.listdir(self.img_path))
        self.masks = sorted(os.listdir(self.msk_path))
        self.transform = transforms
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        image = Image.open(f'{self.img_path}/{self.imgs[index]}')
        id = self.imgs[index].split('.')[0]
        mask = Image.open(f'{self.msk_path}/{id}_segmentation.jpg')
        mask = mask.convert('L')
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask    


def get_isic17_dataloader(path:str, batch_size: int):
    transform_4train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    transform_4test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


    train_dataset = isic_17_dataset(
        f'{path}/train',
        transforms=transform_4train
    )

    test_dataset = isic_17_dataset(
        f'{path}/test',
        transforms=transform_4test
    )

    val_dataset = isic_17_dataset(
        f'{path}/dev',
        transforms=transform_4test
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        **kwargs,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        **kwargs,
    )

    return train_dataloader, test_dataloader, val_dataloader


def get_bowl18_dataloader(path:str, batch_size: int):
    transform_4train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    transform_4test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


    train_dataset = bowl_18_dataset(
        f'{path}/train',
        transforms=transform_4train
    )

    test_dataset = bowl_18_dataset(
        f'{path}/test',
        transforms=transform_4test
    )

    val_dataset = bowl_18_dataset(
        f'{path}/dev',
        transforms=transform_4test
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        **kwargs,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        **kwargs,
    )

    return train_dataloader, test_dataloader, val_dataloader


if __name__ == "__main__":
    trainloader, _, valloader = get_bowl18_dataloader('data/processed', 128)
    print(len(valloader))
