import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BookCover
from models import ImageModel
from utils import score_function


@torch.no_grad()
def test_no_label(model, test_loader):
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader), ncols=120)

    preds, img_names = [], []
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        img_name = batch_item['img_name']

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)
        preds.extend(torch.softmax(pred, dim=1).clone().detach().cpu().numpy())  # probabillity, not label
        img_names.extend(img_name)
    return preds, img_names


@torch.no_grad()
def test_with_label(args, model, test_loader):
    model.eval()
    total_test_score = 0
    preds = []
    answer = []
    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader), ncols=120)

    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)

        test_score = score_function(label, pred)
        total_test_score += test_score
        preds.extend(torch.argmax(pred, dim=1).clone().cpu().numpy())
        answer.extend(label.cpu().numpy())

        log = f'[TEST] Test Acc : {test_score.item():.4f}({total_test_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[TEST] Test Acc : {total_test_score / (batch_idx + 1):.4f}'

        batch_iter.set_description(log)
        batch_iter.update()

    test_mean_acc = total_test_score / len(batch_iter)

    batch_iter.close()

    return test_mean_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dd', '--data_dir', type=str, default='/hdd/sy/food-kt')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        default='/hdd/sy/weights/food-kt/tf_efficientnet_b4_ns_0925_180424/ckpt_best.pt')
    parser.add_argument('-m', '--model', type=str, default='tf_efficientnet_b4_ns')
    parser.add_argument('-av', '--aug_ver', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-is', '--img_size', type=int, default=384)
    parser.add_argument('-nw', '--num_workers', type=int, default=8)

    parser.add_argument('--amp', type=bool, default=True)

    parser.add_argument('-md', '--mode', type=str, default='no_label', choices=['no_label', 'with_label'])
    parser.add_argument('-sd', '--save_dir', type=str, default='./submissions')
    parser.add_argument('-cv', '--csv_name', type=str, default='test')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SET DATASET ####
    label_description = sorted(os.listdir(os.path.join(args.data_dir, 'train')))
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    # test_data = sorted(glob(f'{os.path.join(args.data_dir, "val")}/*/*.jpg'))
    test_data = sorted(glob(f'{os.path.join(args.data_dir, "test")}/*/*.jpg'))
    #####################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='test')
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print('> MODEL BUILT')
    ####################

    if args.mode == 'no_label':  # w/o label
        #### LOAD DATASET ####
        test_dataset = FoodKT(args, test_data, labels=None, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        print('> DATAMODULE BUILT')
        ######################

        #### INFERENCE START ####
        print('> START INFERENCE ')
        preds, img_names = test_no_label(model, test_loader)
        preds = np.argmax(preds, axis=1)
        preds = np.array([label_decoder[val] for val in preds])

        submission = pd.DataFrame()
        submission['image_name'] = img_names
        submission['label'] = preds

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        submission.to_csv(f'{args.save_dir}/{args.csv_name}.csv', index=False)
        #########################
    elif args.mode == 'with_label':
        # w/ label
        test_label = [data.split('/')[-2] for data in test_data]  # '가자미전'
        test_labels = [label_encoder[k] for k in test_label]  # 0

        #### LOAD DATASET ####
        test_dataset = FoodKT(args, test_data, labels=test_labels, mode='valid')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        print('> DATAMODULE BUILT')
        ######################

        #### INFERENCE START ####
        print('> START INFERENCE ')
        test_acc = test_with_label(args, model, test_loader)

        print("=" * 50 + "\n\n")
        print(f"final accuracy: {test_acc * 100:.2f}% \n\n")
        print("=" * 50 + "\n\n")
        #########################
