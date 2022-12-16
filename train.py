import sys
import os
import argparse
import time
import numpy as np
import glob
import json
import torch
import torch.nn as nn

from model.dataloader import get_isic17_dataloader
from model.net import UNet
from utils import custom_metrics
from utils import custom_loss


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
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

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with(open(args.model_dir)) as f:
        hyperparams = json.load(f)
    batch_size = hyperparams["batch size"]
    n_epoch = hyperparams["num epochs"]
    train_dataloader, _, val_dataloader = get_isic17_dataloader(args.data_root, batch_size)

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()

    model = UNet()

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning rate"])

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
        n_epoch
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss
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
                "Trained models/FCBFormer_" + args.dataset + ".pt",
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default='data/processed')
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    # main()
    with(open('experiments/base_model/param.json')) as f:
        hyperparams = json.load(f)
    print(hyperparams["batch size"])
    