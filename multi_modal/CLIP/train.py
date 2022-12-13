from PIL import Image
import torch
import argparse
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from matplotlib.pyplot import imshow
import torchtext
import nltk, re, string, collections
from nltk.util import ngrams
import collections


class MemeDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, captions in tqdm(data.items(), desc='data loading : '):
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in tqdm(data, desc='second loading : '):
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.img_paths_set.index(path) for path in self.img_paths_set}

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def dataloader(image_path, label_path):
    image_class = os.listdir(image_path)
    image_class.sort()
    label_class = os.listdir(label_path)
    label_class.sort()
    data_results = {}

    for i in range(len(image_class)):
        image_dir = os.path.join(image_path, image_class[i])
        labels = pd.read_csv(os.path.join(label_path, label_class[i]))
        labels_data = labels.values
        for data in labels_data:
            img_num = datasplit(data[0])
            img_id = img_num + '.jpg'
            img_path = os.path.join(image_dir, img_id)
            label = wordsplit(data[1:])
            data_results[img_path] = label

    return data_results

def datasplit(data):
    start_idx = data.find("'")
    end_idx = data.rfind("'")
    split_data = data[start_idx + 1:end_idx]

    return split_data


def wordsplit(data):
    word_list = []
    words = data[0].split(',')
    for word in words:
        split_word = datasplit(word)
        word_list.append(split_word)

    str = " ".join(word_list)
    return str

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_images', type=str, default='/workspace/DATA/images/train')
    parser.add_argument('--train_labels', type=str, default='/workspace/DATA/labels/train')
    parser.add_argument('--val_images', type=str, default='/workspace/DATA/images/val')
    parser.add_argument('--val_labels', type=str, default='/workspace/DATA/labels/val')
    parser.add_argument('--save_path', type=str, default='/workspace/result')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    image = preprocess(Image.open("/workspace/DATA/images/test/Arts-Photography/00111.jpg")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    print(image.shape, text.shape)

    d_train = dataloader(args.train_images, args.train_labels)
    d_test = dataloader(args.val_images, args.val_labels)

    train_dataset = MemeDataset(d_train, preprocess)
    test_dataset = MemeDataset(d_test, preprocess)
    print('train data : {0} / val data : {1}'.format(len(train_dataset), len(test_dataset)))

    train_labels = torch.tensor([item[2] for item in train_dataset])
    train_sampler = BalancedBatchSampler(train_labels, args.batch_size, 1)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

    test_labels = torch.tensor([item[2] for item in test_dataset])
    test_sampler = BalancedBatchSampler(test_labels, args.batch_size, 1)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    if device == "cpu":
        model.float()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * args.epoch)

    best_te_loss = 1e5
    best_ep = -1
    for epoch in range(args.epoch):
        print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            images, texts, _ = batch
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            #         print(images.shape, texts.shape)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(args.batch_size).to(device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            tr_loss += total_loss.item()
            if device == "cpu":
                optimizer.step()
                scheduler.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                scheduler.step()
                clip.model.convert_weights(model)
            pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
        tr_loss /= step

        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                images, texts, _ = batch
                images = images.to(device)
                texts = clip.tokenize(texts).to(device)
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(args.batch_size).to(device)

                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                te_loss += total_loss.item()
                test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
            te_loss /= step

        if te_loss < best_te_loss:
            best_te_loss = te_loss
            best_ep = epoch
            torch.save(model.state_dict(), args.save_path + "best_model.pt")
        print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
    torch.save(model.state_dict(), args.save_path + "last_model.pt")
