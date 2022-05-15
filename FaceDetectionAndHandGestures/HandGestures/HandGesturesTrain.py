# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

from FaceDetectionAndHandGestures.HandGestures.HandGesturesDataset import HandGesturesDataset
from FaceDetectionAndHandGestures.HandGestures.HandGesturesModel import HandGesturesModel


class HandGesturesTrain(object):
    """
        Neural network training class
    """

    def __init__(self, path, epochs=5):
        #
        self.path = path
        self.bs = 32
        self.lr = 0.01
        self.epochs = epochs
        self.model = HandGesturesModel()
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr)
        self.lost_fn = nn.CrossEntropyLoss()

    @staticmethod
    def _loss_batch(model, loss_fn, inputs, labels, opt=None):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(inputs), len(labels)

    def save(self, path):
        torch.save(self.model, path)

    def fit(self):
        train_transformer = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        val_transformer = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()])

        images = glob.glob(os.path.join(self.path, 'leapGestRecog/**/**/*.png'))
        labels = [int(os.path.basename(i).split('_')[2]) - 1 for i in images]
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        train_dataset = HandGesturesDataset(x_train, y_train, train_transformer)
        val_dataset = HandGesturesDataset(x_val, y_val, val_transformer)

        train_dl = DataLoader(train_dataset, self.bs, shuffle=True)
        val_dl = DataLoader(val_dataset, self.bs)

        for epoch in range(self.epochs):
            running_items, running_right = 0.0, 0.0
            self.model.train()

            for idx, (inputs, labels) in enumerate(train_dl):
                run_loss, _, run_item = self._loss_batch(self.model, self.lost_fn, inputs, labels, self.optim)

                running_items += run_item
                running_right += run_loss
                res_loss_train = run_loss / running_items

                if idx % 100 == 0 or (idx + 1) == len(train_dl):
                    running_loss, running_items = 0.0, 0.0
                    print(f'TRAIN - Epoch [{epoch + 1}/{self.epochs}]. '
                          f'Step [{idx + 1}/{len(train_dl)}]. '
                          f'Loss: {res_loss_train:.3f}. ')

            self.model.eval()
            with torch.no_grad():
                test_loss, test_nums, test_item = zip(
                    *[self._loss_batch(self.model, self.lost_fn, xb, yb) for xb, yb in val_dl]
                )
            val_loss = np.sum(np.multiply(test_loss, test_nums)) / np.sum(test_nums)

            print(f'TEST - Epoch [{epoch + 1}/{self.epochs}]. '
                  f'Test acc:{val_loss:.3f}.')
