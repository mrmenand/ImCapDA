import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.tools import AverageMeter


def test_acc_imcapda(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True

    with torch.no_grad():
        for data, target in  tqdm(iterable=target_test_loader, desc="Testing",position=0, leave=True, dynamic_ncols=False):
            data, target = data.to(args.device), target.to(args.device)
            _, output_clip, output_fc = model(data)

            loss = criterion(output_fc, target)
            test_loss.update(loss.item())

            pred_fc = torch.max(output_fc, 1)[1]
            pred_clip = torch.max(output_clip, 1)[1]

            if first_test:
                all_pred_fc = pred_fc
                all_pred_clip = pred_clip
                all_label = target
                first_test = False
            else:
                all_pred_fc = torch.cat((all_pred_fc, pred_fc), 0)
                all_pred_clip = torch.cat((all_pred_clip, pred_clip), 0)
                all_label = torch.cat((all_label, target), 0)

    acc_fc = torch.sum(torch.squeeze(all_pred_fc).float() == all_label) / float(all_label.size()[0]) * 100
    acc_clip = torch.sum(torch.squeeze(all_pred_clip).float() == all_label) / float(all_label.size()[0]) * 100

    return acc_fc,acc_clip,test_loss.avg


def test_acc_imcapfusionda(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True

    with torch.no_grad():
        for data, target, blip_caption in  tqdm(iterable=target_test_loader, desc="Testing", position=0, leave=True,
                                 dynamic_ncols=False):

            data, target = data.to(args.device), target.to(args.device)
            _, _, output_clip, output_fc = model(data, blip_caption)

            loss = criterion(output_fc, target)
            test_loss.update(loss.item())

            pred_fc = torch.max(output_fc, 1)[1]
            pred_clip = torch.max(output_clip, 1)[1]

            if first_test:
                all_pred_fc = pred_fc
                all_pred_clip = pred_clip
                all_label = target
                first_test = False
            else:
                all_pred_fc = torch.cat((all_pred_fc, pred_fc), 0)
                all_pred_clip = torch.cat((all_pred_clip, pred_clip), 0)
                all_label = torch.cat((all_label, target), 0)

    acc_fc = torch.sum(torch.squeeze(all_pred_fc).float() == all_label) / float(all_label.size()[0]) * 100
    acc_clip = torch.sum(torch.squeeze(all_pred_clip).float() == all_label) / float(all_label.size()[0]) * 100

    return acc_fc, acc_clip, test_loss.avg


def test_acc_imcapda_visda(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True

    with torch.no_grad():
        for data, target in tqdm(iterable=target_test_loader, desc="Testing",position=0, leave=True, dynamic_ncols=False):
            data, target = data.to(args.device), target.to(args.device)
            _, output_clip, output_fc = model(data)

            loss = criterion(output_fc, target)
            test_loss.update(loss.item())

            pred_fc = torch.max(output_fc, 1)[1]
            pred_clip = torch.max(output_clip, 1)[1]

            if first_test:
                all_pred_fc = pred_fc
                all_pred_clip = pred_clip
                all_label = target
                first_test = False
            else:
                all_pred_fc = torch.cat((all_pred_fc, pred_fc), 0)
                all_pred_clip = torch.cat((all_pred_clip, pred_clip), 0)
                all_label = torch.cat((all_label, target), 0)

    acc_blanced = metrics.balanced_accuracy_score(all_label.cpu().numpy(),
                                          torch.squeeze(all_pred_fc).float().cpu().numpy()) * 100
    cm = metrics.confusion_matrix(all_label.cpu().numpy(),
                                  torch.squeeze(all_pred_fc).float().cpu().numpy())
    per_classes_acc = list(((cm.diagonal() / cm.sum(1)) * 100).round(2))
    per_classes_acc = list(map(str, per_classes_acc))
    per_classes_acc = ', '.join(per_classes_acc)



    return acc_blanced, per_classes_acc,test_loss.avg



def test_acc_imcapfusionda_visda(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True

    with torch.no_grad():
        # for data, target, blip_caption in target_test_loader:
        for data, target, blip_caption  in tqdm(iterable=target_test_loader, desc="Testing", position=0, leave=True,
                                 dynamic_ncols=False):

            data, target = data.to(args.device), target.to(args.device)
            _, _, output_clip, output_fc = model(data, blip_caption)

            loss = criterion(output_fc, target)
            test_loss.update(loss.item())

            pred_fc = torch.max(output_fc, 1)[1]
            pred_clip = torch.max(output_clip, 1)[1]

            if first_test:
                all_pred_fc = pred_fc
                all_pred_clip = pred_clip
                all_label = target
                first_test = False
            else:
                all_pred_fc = torch.cat((all_pred_fc, pred_fc), 0)
                all_pred_clip = torch.cat((all_pred_clip, pred_clip), 0)
                all_label = torch.cat((all_label, target), 0)

    acc_blanced = metrics.balanced_accuracy_score(all_label.cpu().numpy(),
                                                  torch.squeeze(all_pred_fc).float().cpu().numpy()) * 100
    cm = metrics.confusion_matrix(all_label.cpu().numpy(),
                                  torch.squeeze(all_pred_fc).float().cpu().numpy())
    per_classes_acc = list(((cm.diagonal() / cm.sum(1)) * 100).round(2))
    per_classes_acc = list(map(str, per_classes_acc))
    per_classes_acc = ', '.join(per_classes_acc)

    return acc_blanced, per_classes_acc, test_loss.avg
