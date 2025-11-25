import torch
import torch.nn as nn
from models import cmkd
import logging
import torch.nn.functional as F
import copy

_logger = logging.getLogger(__name__)

import torch.nn as nn
import clip

class CLIP(nn.Module):
    def __init__(self,args):
        super(CLIP, self).__init__()
        if args.model_name == "RN50":
            model, preprocess = clip.load("RN50", device="cuda")
            self.output_num = 1024
        elif args.model_name == "RN101":
            model, preprocess = clip.load("RN101", device="cuda")
            self.output_num = 512
        elif args.model_name == "VIT-B":
            model, preprocess = clip.load("ViT-B/16", device="cuda")
            self.output_num = 512
        if args.datasets=="office_home":
            class_list = ['an image of a alarm clock', 'an image of a backpack', 'an image of a batteries', 'an image of a bed', 'an image of a bike', 'an image of a bottle', 'an image of a bucket', 'an image of a calculator', 'an image of a calendar', 'an image of a candles', 'an image of a chair', 'an image of a clipboards', 'an image of a computer', 'an image of a couch', 'an image of a curtains', 'an image of a desk lamp', 'an image of a drill', 'an image of a eraser', 'an image of a exit sign', 'an image of a fan', 'an image of a file cabinet', 'an image of a flipflops', 'an image of a flowers', 'an image of a folder', 'an image of a fork', 'an image of a glasses', 'an image of a hammer', 'an image of a helmet', 'an image of a kettle', 'an image of a keyboard', 'an image of a knives', 'an image of a lamp shade', 'an image of a laptop', 'an image of a marker', 'an image of a monitor', 'an image of a mop', 'an image of a mouse', 'an image of a mug', 'an image of a notebook', 'an image of a oven', 'an image of a pan', 'an image of a paper clip', 'an image of a pen', 'an image of a pencil', 'an image of a postit notes', 'an image of a printer', 'an image of a push pin', 'an image of a radio', 'an image of a refrigerator', 'an image of a ruler', 'an image of a scissors', 'an image of a screwdriver', 'an image of a shelf', 'an image of a sink', 'an image of a sneakers', 'an image of a soda', 'an image of a speaker', 'an image of a spoon', 'an image of a tv', 'an image of a table', 'an image of a telephone', 'an image of a toothbrush', 'an image of a toys', 'an image of a trash can', 'an image of a webcam']

        self.model = model
        self.args = args
        self.text = clip.tokenize(class_list).cuda()
        text_features = self.encode_text().detach().cuda()
        self.text_features = text_features / text_features.norm(dim=1, keepdim=True)

    def forward_features(self, x):
        feature = self.model.encode_image(x)
        return feature

    def encode_text(self):
        text_features = self.model.encode_text(self.text)
        return text_features

    def forward_head(self,image_features, return_text_logit=False):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        if return_text_logit:
            return logits_per_image,logits_per_text
        else:
            return logits_per_image

    def forward(self, x):
        image_features = self.forward_features(x)
        logits_per_image = self.forward_head(image_features)

        return logits_per_image




def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
       m.eval()

class TransferNet(nn.Module):
    def __init__(self, args, train=True):
        super(TransferNet, self).__init__()
        # define the network
        # get the feature extractor and the pretrained head
        self.args = args
        self.num_class = args.num_class
        self.base_network = CLIP(args).cuda()
        # self.teacher_model = copy.deepcopy(self.base_network)
        # self.teacher_model.eval()

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, self.num_class,bias=False))
        self.classifier_layer.apply(weights_init_classifier)

         # define the loss functions
        self.cmkd = cmkd.CMKD(args)
        self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(self, source, target_img, source_label, target_strong=None, label_set=None):
        # self.base_network.apply(fix_bn)
        source = self.base_network.forward_features(source)

        # calculate source classification loss Lclf
        source_logits = self.classifier_layer(source)
        clf_loss = self.clf_loss(source_logits, source_label)


        source_logits_clip = self.base_network.forward_head(source)

        target = self.base_network.forward_features(target_img)
        # calculate calibrated probability alignment loss Lcpa
        target_clip_logits = self.base_network.forward_head(target)
        target_logits = self.classifier_layer(target)

        # calculate calibrated gini impurity loss Lcgi
        transfer_loss = self.cmkd(target_logits, target_clip_logits, source_logits_clip, source_label,label_set)

        feats_img = torch.cat((source, target), dim=0)
        logits_clip = torch.cat((source_logits, target_logits), dim=0)

        return clf_loss, transfer_loss, feats_img, logits_clip



        # if self.args.fixmatch and target_strong is not None:
        #     target_pred = F.softmax(target_logits, dim=1)
        #     if label_set is not None:
        #         compl_label_set = list(set(torch.range(0, 64).tolist()) - set(label_set))
        #         compl_label_set = [int(item) for item in compl_label_set]
        #         target_pred[:, compl_label_set] = 0.0
        #     max_prob, pred_u = torch.max(target_pred, dim=-1)
        #     target_strong_feature = self.base_network.forward_features(target_strong)
        #     target_strong = self.classifier_layer(target_strong_feature)
        #     fixmatch_loss = self.args.fixmatch_factor * (F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
        #                                                  max_prob.ge(self.args.fixmatch_threshold).float().detach()).mean()
        #
        #     target_pred_clip = F.softmax(target_clip_logits,dim=-1)
        #     if label_set is not None:
        #         target_pred_clip[:, compl_label_set] = 0.0
        #
        #     max_prob, pred_u = torch.max(target_pred_clip, dim=-1)
        #     target_strong = self.base_network.forward_head(target_strong_feature)
        #     fixmatch_loss += self.args.fixmatch_factor * (
        #                 F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
        #                 max_prob.ge(self.args.fixmatch_threshold).float().detach()).mean()
        #     transfer_loss += fixmatch_loss
        #
        # if self.args.pda:
        #     clf_loss = 0.5 * clf_loss
        #     transfer_loss = 0.1 * transfer_loss

        #return clf_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params=[
            {'params': self.base_network.model.visual.parameters(), 'lr': initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}
]
        return params

    def predict(self, x):
        features = self.base_network.forward_features(x)
        logit = self.classifier_layer(features)
        return logit

    def clip_predict(self, x):
        logit = self.base_network(x)
        return logit