import sys
import os
import argparse
import time
import numpy as np
import logging
import json
import torch
import torch.nn as nn

from model.dataloader import get_isic17_dataloader, get_bowl18_dataloader
from model.net import UNet
from utils.custom_metrics import DiceScore
from utils.custom_loss import SoftDiceLoss


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )
    l = np.mean(loss_accumulator)
    logging.info(f'Train epoch {epoch}, loss: {l}')
    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )
    p = np.mean(perf_accumulator)
    logging.info(f'Test epoch {epoch}, performance: {p}')
    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # load hyperparameters
    with(open(f'{args.model_dir}/param.json')) as f:
        hyperparams = json.load(f)
    batch_size = hyperparams["batch size"]
    n_epoch = hyperparams["num epochs"]
    lr = hyperparams["learning rate"]
    lrs = hyperparams["learning rate scheduler"]
    lrs_min = hyperparams["learning rate scheduler minimum"]
    
    # Load dataloaders...
    if args.dataset == "Bowl18":
        train_dataloader, _, val_dataloader = get_bowl18_dataloader(args.data_root, batch_size)
    elif args.dataset == "ISIC17":
        train_dataloader, _, val_dataloader = get_isic17_dataloader(args.data_root, batch_size)
        
    # get loss function
    Dice_loss = SoftDiceLoss()
    # get Performance metrics
    perf = DiceScore()

    # load model
    model = UNet(n_channels=3, n_classes=1)

    # Multi gpu option
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        perf,
        model,
        optimizer,
        n_epoch, 
        lrs, 
        lrs_min
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        perf,
        model,
        optimizer,
        n_epoch, 
        lrs, 
        lrs_min
    ) = build(args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    prev_best_test = None
    if lrs == "true":
        if lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, n_epoch + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                args.model_dir + args.dataset + "best_performance.pt",
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-d", "--model_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["Bowl18", "ISIC17"])
    parser.add_argument("--data_root", type=str, default='data/processed')
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(filename=f'{args.model_dir}/train.log', 
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    train(args)


if __name__ == "__main__":
    main()
    