import argparse
from glob import glob

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BookCover
from models import ImageModel, DOLG
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

    batch_iter.close()

    if args.wandb:
        wandb.log({'train_mean_loss': train_mean_loss, 'lr': _lr, 'train_mean_acc': train_mean_acc}, step=epoch)


@torch.no_grad()
def valid(args, model, val_loader, criterion, epoch, wandb):
    model.eval()
    total_val_loss = 0
    total_val_score = 0
    preds = []
    answer = []
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
        preds.extend(torch.argmax(pred, dim=1).clone().cpu().numpy())
        answer.extend(label.cpu().numpy())

        log = f'[EPOCH {epoch}] Valid Loss : {val_loss.item():.4f}({total_val_loss / (batch_idx + 1):.4f}), '
        log += f'Valid Acc : {val_score.item():.4f}({total_val_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Valid Loss : {total_val_loss / (batch_idx + 1):.4f}, '
            log += f'Valid Acc : {total_val_score / (batch_idx + 1):.4f}, '

        batch_iter.set_description(log)
        batch_iter.update()

    val_mean_loss = total_val_loss / len(batch_iter)
    val_mean_acc = total_val_score / len(batch_iter)

    batch_iter.close()

    if args.wandb:
        wandb.log({'valid_mean_loss': val_mean_loss, 'valid_mean_acc': val_mean_acc}, step=epoch)

    return val_mean_loss, val_mean_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default='/hdd/book-covers-split/')
    parser.add_argument('-sd', '--save_dir', type=str, default='/hdd/model/book')
    parser.add_argument('-m', '--model', type=str, default='tf_efficientnet_b4')
    parser.add_argument('-is', '--img_size', type=int, default=224)
    parser.add_argument('-se', '--seed', type=int, default=42)
    parser.add_argument('-av', '--aug_ver', type=int, default=9)

    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-we', '--warm_epoch', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-nw', '--num_workers', type=int, default=4)

    parser.add_argument('-l', '--loss', type=str, default='ce', choices=['ce', 'focal', 'smoothing_ce'])
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.5)
    parser.add_argument('-ot', '--optimizer', type=str, default='adam',
                        choices=['adam', 'radam', 'adamw', 'adamp', 'ranger', 'lamb', 'adabound'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-2)

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
    parser.add_argument('-mp', '--mix_prob', type=float, default=0.5)
    parser.add_argument('-cms', '--cutmix_stop', type=int, default=51)

    # wandb config:
    parser.add_argument('--wandb', type=bool, default=True)

    # amp config:
    parser.add_argument('--amp', type=bool, default=True)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SEED EVERYTHING ####
    seed_everything(args.seed)
    #########################

    #### SET DATASET ####
    label_description = sorted(os.listdir(os.path.join(args.data_dir, 'train')))
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    train_data = sorted(glob(f'{os.path.join(args.data_dir, "train")}/*/*.jpg'))
    train_label = [data.split('/')[-2] for data in train_data]
    train_labels = [label_encoder[k] for k in train_label]

    val_data = sorted(glob(f'{os.path.join(args.data_dir, "val")}/*/*.jpg'))
    val_label = [data.split('/')[-2] for data in val_data]
    val_labels = [label_encoder[k] for k in val_label]
    #####################

    c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')
    save_dir = os.path.join(args.save_dir, f'{args.model}_{c_date}_{c_time}')
    os.makedirs(save_dir)

    #### SET WANDB ####
    run = None
    if args.wandb:
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        wandb.login(key=wandb_api_key)
        run = wandb.init(project='book-cover', name=f'{args.model}_{c_date}_{c_time}')
        wandb.config.update(args)
    ###################

    #### LOAD DATASET ####
    train_dataset = BookCover(args, train_data, train_labels, mode='train')
    val_dataset = BookCover(args, val_data, val_labels, mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    iter_per_epoch = len(train_loader)
    print('> DATAMODULE BUILT')
    ######################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='train')
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print('> MODEL BUILT')
    ####################

    #### SET TRAINER ####
    optimizer = set_optimizer(args, model)
    criterion = set_loss(args)
    scheduler = set_scheduler(args, optimizer, iter_per_epoch)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm_epoch) if args.warm_epoch else None
    print('> TRAINER SET')
    #####################

    best_val_acc = .0
    best_val_loss = 9999.
    best_epoch = 0

    print('> START TRAINING')
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb)
        val_loss, val_acc = valid(args, model, val_loader, criterion, epoch, wandb)
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_loss = min(val_loss, best_val_loss)
            best_val_acc = max(val_acc, best_val_acc)

            torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch, },
                       f'{save_dir}/ckpt_best.pt')
            print(f'> SAVED model ({epoch:02d}) at {save_dir}/ckpt_best.pt')

        if args.scheduler in ['cos_base', 'cos']:
            scheduler.step()

    del model
    del optimizer, scheduler
    del train_dataset, val_dataset

    if args.wandb:
        run.finish()
