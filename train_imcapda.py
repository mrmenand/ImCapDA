import datetime
import logging
import os
import time
import warnings

import configargparse
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import clip
from models.ImCapDA import ImCapDA
from utils.dset_loader import get_dset_loader_imcapda
from utils.loss import vlm_text_contrastive_loss
from utils.tools import str2bool, AverageMeter, set_random_seed
from utils.val_test import test_acc_imcapda

warnings.filterwarnings("ignore")
scaler = GradScaler()


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args):
    best_acc = 0

    # Zero short Acc.
    _, acc_zero_shot_clip, _ = test_acc_imcapda(model, target_test_loader, args)
    acc_zero_shot_clip_msg = f"CLIP Zero-Shot Acc: {round(acc_zero_shot_clip.item(), 2)}"
    logging.info(acc_zero_shot_clip_msg);
    tqdm.write(acc_zero_shot_clip_msg)

    pbar = tqdm(total=args.max_iter, desc='Train: ', postfix=dict, mininterval=0.3)
    for global_step in range(args.max_iter):

        model.train()
        optimizer.zero_grad()

        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()

        if global_step % len(source_loader) == 0:
            iter_source = iter(source_loader)
        if global_step % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)

        data_source, label_source, caption_src = next(iter_source)
        data_target, label_target, caption_tgt = next(iter_target)
        data_source, label_source = data_source.to(args.device), label_source.to(args.device)
        data_target, label_target = data_target.to(args.device), label_target.to(args.device)

        inputs = torch.cat((data_source, data_target))

        with autocast("cuda"):
            feats_img, outputs_clip, outputs = model(inputs)

            feats_src, feats_tgt = feats_img.chunk(2, dim=0)
            outputs_src, outputs_tgt = outputs.chunk(2, dim=0)
            outputs_src_clip, outputs_tgt_clip = outputs_clip.chunk(2, dim=0)

            classifier_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)(outputs_src, label_source)

            # transfer_loss = compute_im_loss(outputs_tgt)
            # transfer_loss = torch.tensor(0)
            transfer_loss = model.cmkd(outputs_tgt, outputs_tgt_clip, outputs_src_clip, label_source)

            blip_caption = caption_src + caption_tgt
            blip_texts = torch.cat([clip.tokenize(p) for p in blip_caption]).to(args.device)
            blip_text_features = model.clip_model.clip_model.encode_text(blip_texts)
            blip_text_features = blip_text_features / blip_text_features.norm(dim=-1, keepdim=True)  # 归一化

            contrastive_loss = vlm_text_contrastive_loss(feats_img, blip_text_features)

            pseudo_label = torch.softmax(outputs_tgt_clip, dim=-1)
            max_probs, label_p = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(0.9).float()
            loss_u = (torch.nn.CrossEntropyLoss(reduction="none")(outputs_tgt, label_p) * mask).sum() / mask.sum()

        loss = classifier_loss + transfer_loss + args.alpha * loss_u + args.beta * contrastive_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # learning rate scheduler update
        scheduler.step()

        # training loss update
        train_loss_clf.update(classifier_loss.item())
        train_loss_transfer.update(transfer_loss.item())
        train_loss_total.update(loss.item())

        pbar.set_postfix({'Cls Loss': train_loss_clf.avg, 'Transfer Loss': train_loss_transfer.avg,
                          'Total Loss': train_loss_total.avg})
        pbar.update(1)

        if (global_step + 1) % args.eval_every == 0:
            info = f'Iter Step: {global_step}/{args.max_iter}, cls_loss:{train_loss_transfer.avg:.2f} transfer_loss: {train_loss_transfer.avg:.2f}, total_Loss: {train_loss_total.avg:.2f}'
            test_acc, test_acc_clip, test_loss = test_acc_imcapda(model, target_test_loader, args)
            info += f', test_loss {test_loss:.2f}, test_acc_fc: {test_acc:.2f}, test_acc_clip: {test_acc_clip:.2f}'

            if best_acc < test_acc:
                best_acc = test_acc

            logging.info(info);
            tqdm.write(info)
            time.sleep(1)

    best_acc_msg = f"Best Accuracy: {round(best_acc.item(), 2)}"
    tqdm.write(best_acc_msg);logging.info(best_acc_msg)


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(description="Transfer learning config parser",
                                           config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter, )

    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--log_dir', type=str, default='log')

    # network related
    parser.add_argument('--model_name', type=str, default='ViT-B/16')

    # data loading related
    parser.add_argument('--datasets', type=str, default='office_home')
    parser.add_argument('--num_class', type=int, default=65)
    parser.add_argument('--data_dir', type=str, default="/root/OfficeHomeDataset_10072016")
    parser.add_argument('--src_domain', type=str, default="Real_World")
    parser.add_argument('--tgt_domain', type=str, default="Art")
    parser.add_argument('--vlm_text', type=str, default="blip3")

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)

    # optimizer related
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--multiple_lr_classifier', type=float, default=1000)

    # loss related
    parser.add_argument('--lambda1', type=float, default=0.25)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.025)

    # learning rate scheduler related
    parser.add_argument('--scheduler', type=str2bool, default=True)
    # linear scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lambda_', type=float, default=1.0)

    parser.add_argument('--fuse_mode', default=False, action='store_true')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(args.seed)

    source_loader, target_train_loader, target_test_loader = get_dset_loader_imcapda(args)
    args.classnames = target_test_loader.dataset.classes

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = f'log/{args.model_name}/{args.datasets}/{args.src_domain}2{args.tgt_domain}/{time_str}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setattr(args, "log_dir", log_dir)

    model = ImCapDA(args).to(args.device)

    optimizer = torch.optim.SGD(model.get_parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (
            args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)))

    print(f"Base Network: {args.model_name}")
    print(f"Source Domain: {args.src_domain}")
    print(f"Target Domain: {args.tgt_domain}")

    # 记录参数到日志文件
    setattr(args, "script_name", os.path.basename(__file__))
    logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f'{arg}: {value}')

    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)


if __name__ == "__main__":
    main()
