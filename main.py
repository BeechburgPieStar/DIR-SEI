import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import TensorDataset
from util.dataaug import Rotate_DA
import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy as LSR
import csv
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from util.get_dataset import TestDataset
from model.MRAN import MRAN
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2023)

class Config:
    def __init__(
        self,
        batch_size: int = 16,
        test_batch_size: int = 16,
        epochs: int = 1000,
        lr: float = 0.001,
        wd:float = 0,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.wd = wd

conf = Config()
modelweightfile = f'save_model/DIR-MRAN_10%.pth'
def train(model, loss, train_dataloader, optimizer, epoch):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target = data_nn
        data = torch.from_numpy(Rotate_DA(data, 0).astype('float32'))
        data = data.permute(0, 2, 1)
        target = torch.cat((target, target, target, target), dim=0)
        target = target.long()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        lam = np.random.beta(20,20)
        index = torch.randperm(data.size()[0]).cuda()

        target_a, target_b = target, target[index]

        tmp = data
        data = lam * tmp + (1 - lam) * tmp[index, :, :]
        data.cuda()
        target_a = torch.cat([target, target_a])
        target_b = torch.cat([target, target_b])
        data = torch.cat([tmp, data], dim=0)

        optimizer.zero_grad()
        embedding, output = model(data)
        output = F.log_softmax(output, dim=1)
        result_loss = lam * loss(output, target_a) + (1 - lam) * loss(output, target_b)
        result_loss.backward()

        optimizer.step()
        all_loss += result_loss.item() * data.size()[0]
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target_a.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / (len(train_dataloader.dataset) * 8),
        correct,
        len(train_dataloader.dataset) * 8,
        100.0 * correct / (len(train_dataloader.dataset) * 8))
    )
    return all_loss / (len(train_dataloader.dataset) * 8), 100.0 * correct / (len(train_dataloader.dataset) * 8)


def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.permute(0, 2, 1)
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()*data.size()[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return test_loss, 100.0 * correct / len(test_dataloader.dataset)

def test(model, test_dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.permute(0, 2, 1)
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(100.0 * correct / len(test_dataloader.dataset))


def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, save_path):
    tr_loss = []
    ev_loss = []
    tr_acc = []
    ev_acc = []
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, loss_function, train_dataloader, optimizer, epoch)
        test_loss, test_acc = evaluate(model, loss_function, val_dataloader, epoch)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        tr_loss.append(train_loss)
        ev_loss.append(test_loss)
        tr_acc.append(train_acc)
        ev_acc.append(test_acc)
    return tr_loss, ev_loss, tr_acc, ev_acc

def TestDataset_prepared():
    x_test, y_test = TestDataset()
    return x_test, y_test

x_train = np.load("dataset/ADS-B/3%data/x_train_3.npy")
y_train = np.load("dataset/ADS-B/3%data/y_train_3.npy")
x_val = np.load("dataset/ADS-B/3%data/x_val_3.npy")
y_val = np.load("dataset/ADS-B/3%data/y_val_3.npy")


train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

model = MRAN()

optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay = conf.wd)
if torch.cuda.is_available():
    model = model.cuda()

loss = LSR()
if torch.cuda.is_available():
    loss = loss.cuda()

tr_loss, ev_loss, tr_acc, ev_acc = train_and_evaluate(model,
    loss_function=loss,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optim,
    epochs=conf.epochs,
    save_path=modelweightfile)

combined_data = zip(tr_loss, ev_loss, tr_acc, ev_acc)
with open('loss/DIR-MRAN_3%.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for row in combined_data:
        writer.writerow(row)

x_test, y_test = TestDataset_prepared()
y_test = y_test.astype(np.int64)
test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=conf.test_batch_size, shuffle=True)
model = torch.load(modelweightfile)
test(model,test_dataloader)
