import argparse
from glob import glob

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BookCover
from models import ImageModel
from utils import *


@torch.no_grad()
def test(model, val_loader):
    model.eval()
    batch_iter = tqdm(enumerate(val_loader), 'Validating', total=len(val_loader), ncols=120)

    preds, answer = [], []
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)
        preds.extend(torch.argmax(pred, dim=1).clone().cpu().numpy())
        answer.extend(label.cpu().numpy())

    preds = np.array([label_decoder[int(val)] for val in preds])
    answer = np.array([label_decoder[int(val)] for val in answer])
    confusion_matrix = pd.crosstab(answer, preds, rownames=['answer'], colnames=['preds'])
    confusion_matrix.to_csv(f'./comparison_{args.model}.csv', index=True)
    return preds, answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--save_dir', type=str, default='/hdd/sy/weights/food-kt/submissions')
    parser.add_argument('-cv', '--csv_name', type=str, default='test')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        default='/hdd/sy/weights/food-kt/efficientnetv2_rw_s_fold_0/ckpt_best_fold_0.pt')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-is', '--img_size', type=int, default=384)
    parser.add_argument('-nw', '--num_workers', type=int, default=8)
    parser.add_argument('-m', '--model', type=str, default='efficientnetv2_rw_s')
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('-se', '--seed', type=int, default=42)

    # data split configs:
    parser.add_argument('-ds', '--data_split', type=str, default='StratifiedKFold',
                        choices=['Split_base', 'StratifiedKFold'])
    parser.add_argument('-ns', '--n_splits', type=int, default=5)
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.2)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SET DATASET ####
    label_description = sorted(os.listdir('/hdd/sy/food-kt/train'))
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    train_data = sorted(glob('/hdd/sy/food-kt/train/*/*.jpg'))  # len(train_data): 10,000
    train_label = [data.split('/')[-2] for data in train_data]  # '가자미전'
    train_labels = [label_encoder[k] for k in train_label]  # 0

    folds = set_data_split(args, train_data, train_labels)
    #####################
    # fold = 0
    for fold in range(len(folds)):
        train_data, train_lb, val_data, val_lb = folds[fold]

        #### LOAD DATASET ####
        train_dataset = FoodKT(args, train_data, train_lb, mode='train')
        val_dataset = FoodKT(args, val_data, val_lb, mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        iter_per_epoch = len(train_loader)
        print('> DATAMODULE BUILT')
        ######################

        #### LOAD MODEL ####
        model = ImageModel(model_name=args.model, class_n=len(label_description), mode='valid')
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
        model = model.to(args.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        print('> MODEL BUILT')
        ####################

        #### INFERENCE START ####
        print('> START INFERENCE ')
        preds, answers = test(model, val_loader)
        #########################
