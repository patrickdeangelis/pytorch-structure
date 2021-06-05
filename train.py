import logging
import argparse

import torch
from torch import utils, nn, optim
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.dataset import DataSet
from model import Model, ModelCNN


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def main(parser):
    # ---- Data ----
    # Training and validation dataset
    train = DataSet(True, transforms.ToTensor())
    val = DataSet(False, transforms.ToTensor())

    device = torch.device("cyda:0" if torch.cuda.is_available() else "cpu")
    num_worker = 4 * int(torch.cuda.device_count())

    # Dataloaders
    train_loader = utils.data.DataLoader(dataset=train, shuffle=True, batch_size=parser.batch_size, num_workers=num_worker, pin_memory=True)
    validation_loader = utils.data.DataLoader(dataset=val, batch_size=parser.batch_size)


    # ---- Network ----
    loss = nn.CrossEntropyLoss()
    model = ModelCNN()
    # GPU
    model.to(device)
    model.share_memory()
    model.half().float()

    optimizer = optim.SGD(model.parameters(), lr=parser.lr)

    # ---- Training and validation ----
    for epoch in range(1, parser.epochs):
        logging.info(f"Training step: {epoch}")
        model.train()
        for step, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            yPred = model(X.view(-1, 28*28))
            error = loss(yPred, y)
            error.backward()
            optimizer.step()

        logging.info(f"Validation step: {epoch}")
        model.eval()
        val_loss = 0
        corrects = 0
        for step, (X, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            yPred = model(X.view(-1, 28*28))
            val_loss += loss(yPred, y).item()
            predicted = torch.max(yPred, 1)[1]
            corrects += (predicted == y).sum()


        if epoch % 5 == 0:
            weights = f"{parser.dir_save}/{epoch}_weights.pt"
            accuracy = float(corrects) / (len(validation_loader) * parser.batch_size)
            logging.info(f"Loss error: {val_loss:.4f}, Accuracy: {accuracy}")

            chkpt = {'epoch': epoch, 'model': model.state_dict()}
            torch.save(chkpt, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dir_save', default='./weights')
    parser = parser.parse_args()

    main(parser)