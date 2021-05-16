import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


from dataset import XrayDataset
from loss import DiceLoss
from model import Unet


train_image_dir = 'data/train/JPEGImages'
train_mask_dir = 'data/mask_train'
val_image_dir = 'data/val/JPEGImages'
val_mask_dir = 'data/mask_val'
test_image_dir = 'data/test/JPEGImages'
test_mask_dir = 'data/mask_test'


def data_loaders(image_size=(512, 512),
                 batch_size=16,
                 num_workers=4):
    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    train_dataset = XrayDataset(train_image_dir, train_mask_dir, image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )

    val_dataset = XrayDataset(val_image_dir, val_mask_dir, image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init
    )

    return train_loader, val_loader


def train_net(image_size,
              batch_size,
              num_epochs,
              lr,
              num_workers,
              checkpoint):
    train_loader, val_loader = data_loaders(image_size=(image_size, image_size), batch_size=batch_size)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = Unet().to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    criterion = DiceLoss().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    logging.info(f'Start training:\n'
                 f'Num epochs:               {num_epochs}\n'
                 f'Batch size:               {batch_size}\n'
                 f'Learning rate:            {lr}\n'
                 f'Num workers:              {num_workers}\n'
                 f'Scale image size:         {image_size}\n'
                 f'Device:                   {device}\n'
                 f'Checkpoint:               {checkpoint}\n')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}: ')
        train_batch_losses = []
        val_batch_losses = []
        best_val_loss = 9999

        for x_train, y_train in tqdm(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(x_train)

            optimizer.zero_grad()
            loss = criterion(y_pred, y_train)
            train_batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_losses.append(sum(train_batch_losses) / len(train_batch_losses))
        print(f'-----------------------Train loss: {train_losses[-1]} -------------------------------')

        for x_val, y_val in tqdm(val_loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_pred = model(x_val)

            loss = criterion(y_pred, y_val)
            val_batch_losses.append(loss.item())

        val_losses.append(sum(val_batch_losses) / len(val_batch_losses))
        print(f'-----------------------Val loss: {val_losses[-1]} -------------------------------')
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            if not os.path.isdir('weights/'):
                os.mkdir('weights/')
            torch.save(model.state_dict(), f'weights/checkpoint{epoch+1}.pth')
            print(f'Save checkpoint in: weights/checkpoint{epoch+1}.pth')


def get_args():
    parser = argparse.ArgumentParser(
        description='Training U-Net model for segmentation of xray'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help='initial learning rate (default: 1e-3)',
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to checkpoint (default: None)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="input image size (default: 512)",
    )

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    train_net(image_size=args.image_size,
              batch_size=args.batch_size,
              num_epochs=args.num_epochs,
              lr=args.lr,
              num_workers=args.num_workers,
              checkpoint=args.weights)
