import argparse
from glob import glob

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BookCover
from models import ImageModel
from utils import *


def train(args, model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb):
    model.train()
    total_train_loss = 0
    total_train_score = 0
    batch_iter = tqdm(enumerate(train_loader), 'Training', total=len(train_loader), ncols=120)

    for batch_idx, batch_item in batch_iter:
        optimizer.zero_grad()
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        if epoch <= args.warm_epoch:
            warmup_scheduler.step()

        # cutmix
        mix_decision = np.random.rand()
        if args.cutmix and epoch < args.cutmix_stop and mix_decision < args.mix_prob:
            img, mix_labels = cutmix(img, label, 1.)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)
            if args.cutmix and epoch < args.cutmix_stop and mix_decision < args.mix_prob:
                train_loss = criterion(pred, mix_labels[0]) * mix_labels[2] + criterion(pred, mix_labels[1]) * (
                        1. - mix_labels[2])
            else:
                train_loss = criterion(pred, label)

        if args.amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()

        if args.scheduler == 'cycle':
            scheduler.step()

        train_score = score_function(label, pred)
        total_train_loss += train_loss
        total_train_score += train_score

        log = f'[EPOCH {epoch}] Train Loss : {train_loss.item():.4f}({total_train_loss / (batch_idx + 1):.4f}), '
        log += f'Train Acc : {train_score.item():.4f}({total_train_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Train Loss : {total_train_loss / (batch_idx + 1):.4f}, '
            log += f'Train Acc : {total_train_score / (batch_idx + 1):.4f}, '
            log += f"LR : {optimizer.param_groups[0]['lr']:.2e}"

        batch_iter.set_description(log)
        batch_iter.update()

    _lr = optimizer.param_groups[0]['lr']
    train_mean_loss = total_train_loss / len(batch_iter)
    train_mean_acc = total_train_score / len(batch_iter)

    batch_iter.set_description(log)
    batch_iter.close()

    wandb.log({'train_mean_loss': train_mean_loss, 'lr': _lr, 'train_mean_acc': train_mean_acc}, step=epoch)


@torch.no_grad()
def valid(args, model, val_loader, criterion, epoch, wandb):
    model.eval()
    total_val_loss = 0
    total_val_score = 0
    batch_iter = tqdm(enumerate(val_loader), 'Validating', total=len(val_loader), ncols=120)

    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)
            val_loss = criterion(pred, label)

        val_score = score_function(label, pred)
        total_val_loss += val_loss
        total_val_score += val_score

        log = f'[EPOCH {epoch}] Valid Loss : {val_loss.item():.4f}({total_val_loss / (batch_idx + 1):.4f}), '
        log += f'Valid Acc : {val_score.item():.4f}({total_val_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Valid Loss : {total_val_loss / (batch_idx + 1):.4f}, '
            log += f'Valid Acc : {total_val_score / (batch_idx + 1):.4f}, '

        batch_iter.set_description(log)
        batch_iter.update()

    val_mean_loss = total_val_loss / len(batch_iter)
    val_mean_acc = total_val_score / len(batch_iter)
    batch_iter.set_description(log)
    batch_iter.close()

    wandb.log({'valid_mean_loss': val_mean_loss, 'valid_mean_acc': val_mean_acc}, step=epoch)

    return val_mean_loss, val_mean_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--save_dir', type=str, default='/hdd/sy/weights/food-kt')
    parser.add_argument('-m', '--model', type=str, default='convnext_tiny_in22ft1k')
    parser.add_argument('-is', '--img_size', type=int, default=384)
    parser.add_argument('-se', '--seed', type=int, default=42)

    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-we', '--warm_epoch', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-nw', '--num_workers', type=int, default=8)

    parser.add_argument('-l', '--loss', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('-ot', '--optimizer', type=str, default='adamw',
                        choices=['adam', 'radam', 'adamw', 'adamp', 'ranger', 'lamb'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-sc', '--scheduler', type=str, default='cos_base', choices=['cos_base', 'cos', 'cycle'])
    parser.add_argument('-mxlr', '--max_lr', type=float, default=3e-3)  # scheduler - cycle
    parser.add_argument('-mnlr', '--min_lr', type=float, default=1e-6)  # scheduler - cos
    parser.add_argument('-tm', '--tmax', type=float, default=20)  # scheduler - cos
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05)

    # data split configs:
    parser.add_argument('-ds', '--data_split', type=str, default='Split_base',
                        choices=['Split_base', 'StratifiedKFold'])
    parser.add_argument('-ns', '--n_splits', type=int, default=5)
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.2)

    # cut mix
    parser.add_argument('-cm', '--cutmix', type=bool, default=True)
    parser.add_argument('-mp', '--mix_prob', type=float, default=0.3)
    parser.add_argument('-cms', '--cutmix_stop', type=int, default=45)

    # amp config:
    parser.add_argument('--amp', type=bool, default=True)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SEED EVERYTHING ####
    seed_everything(args.seed)
    #########################

    #### SET DATASET ####
    label_description = sorted(os.listdir('/hdd/sy/food-kt/train'))
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    train_data = sorted(glob('/hdd/sy/food-kt/train/*/*.jpg'))  # len(train_data): 10,000
    train_label = [data.split('/')[-2] for data in train_data]  # '가자미전'
    train_labels = [label_encoder[k] for k in train_label]  # 0

    folds = set_data_split(args, train_data, train_labels)
    #####################

    for fold in range(len(folds)):
        train_data, train_lb, val_data, val_lb = folds[fold]

        #### SET WANDB ####
        run = None
        wandb.init(config=args)
        config = wandb.config
        ###################

        #### LOAD DATASET ####
        train_dataset = FoodKT(args, train_data, train_lb, mode='train')
        val_dataset = FoodKT(args, val_data, val_lb, mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                shuffle=False)
        iter_per_epoch = len(train_loader)
        print('> DATAMODULE BUILT')
        ######################

        #### LOAD MODEL ####
        model = ImageModel(model_name=config.model, class_n=len(label_description), mode='train')
        model = model.to(args.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        print('> MODEL BUILT')
        ####################

        #### SET TRAINER ####
        optimizer = set_optimizer(config, model)
        criterion = set_loss(config)
        scheduler = set_scheduler(config, optimizer, iter_per_epoch)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.warm_epoch) if config.warm_epoch else None
        print('> TRAINER SET')
        #####################

        best_val_acc = .0
        best_val_loss = 9999.
        best_epoch = 0

        print('> START TRAINING')
        for epoch in range(1, config.epochs + 1):
            train(config, model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb)
            val_loss, val_acc = valid(config, model, val_loader, criterion, epoch, wandb)

            if config.scheduler in ['cos_base', 'cos']:
                scheduler.step()

        del model
        del optimizer, scheduler
        del train_dataset, val_dataset
