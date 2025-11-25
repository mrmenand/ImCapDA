import torch
import torch.nn as nn

import clip
from models.ImCapDA import CLIP, weights_init_classifier, CMKD, fix_bn


class ImCapFusionDAC(nn.Module):
    def __init__(self, args):
        super(ImCapFusionDAC, self).__init__()
        self.args = args

        clip_model = CLIP(args)
        self.output_dim = clip_model.clip_model.visual.output_dim
        self.clip_model = clip_model.cuda()

        # define the task head
        self.classifier_layer = nn.Sequential(nn.BatchNorm1d(self.output_dim * 2),
                                              nn.LayerNorm(self.output_dim * 2, eps=1e-6),
                                              nn.Linear(self.output_dim * 2, args.num_class, bias=False))
        self.classifier_layer.apply(weights_init_classifier)

        self.cmkd = CMKD(args)

    def forward(self, image, blip_text):
        self.clip_model.apply(fix_bn)
        image_features, logits_clip = self.clip_model(image)

        blip_texts = torch.cat([clip.tokenize(p, truncate=True) for p in blip_text]).to(self.args.device)
        blip_text_features = self.clip_model.clip_model.encode_text(blip_texts)
        blip_text_features = blip_text_features / blip_text_features.norm(dim=-1, keepdim=True)  # 归一化

        features_combine = torch.cat((image_features, blip_text_features), dim=1)
        logits_fc = self.classifier_layer(features_combine)

        return image_features, blip_text_features, logits_clip, logits_fc

    def forward_test(self, image):
        self.clip_model.apply(fix_bn)
        image_features, logits_clip = self.clip_model(image)
        logits_fc = self.classifier_layer(image_features)

        return image_features, logits_clip, logits_fc

    def get_parameters(self, initial_lr=1.0):
        params = [{'params': self.clip_model.clip_model.transformer.parameters(), 'lr': initial_lr},
                  {'params': self.clip_model.clip_model.visual.parameters(), 'lr': initial_lr},
                  {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}, ]
        return params


class ImCapFusionDAA(nn.Module):
    def __init__(self, args):
        super(ImCapFusionDAA, self).__init__()
        self.args = args

        clip_model = CLIP(args)
        self.output_dim = clip_model.clip_model.ln_final.weight.shape[0]
        self.clip_model = clip_model.cuda()

        # define the task head
        self.classifier_layer = nn.Sequential(nn.BatchNorm1d(self.output_dim), nn.LayerNorm(self.output_dim, eps=1e-6),
                                              nn.Linear(self.output_dim, args.num_class, bias=False))
        self.classifier_layer.apply(weights_init_classifier)

        self.cmkd = CMKD(args)

    def forward(self, image, blip_text):
        self.clip_model.apply(fix_bn)
        image_features, logits_clip = self.clip_model(image)

        blip_texts = torch.cat([clip.tokenize(p) for p in blip_text]).to(self.args.device)
        blip_text_features = self.clip_model.clip_model.encode_text(blip_texts)
        blip_text_features = blip_text_features / blip_text_features.norm(dim=-1, keepdim=True)  # 归一化

        # features_combine = torch.cat((image_features, blip_text_features),dim=1)

        features_combine = torch.add(image_features, blip_text_features)

        logits_fc = self.classifier_layer(features_combine)

        return image_features, blip_text_features, logits_clip, logits_fc

    def forward_test(self, image):
        self.clip_model.apply(fix_bn)
        image_features, logits_clip = self.clip_model(image)
        logits_fc = self.classifier_layer(image_features)

        return image_features, logits_clip, logits_fc

    def get_parameters(self, initial_lr=1.0):
        params = [{'params': self.clip_model.clip_model.transformer.parameters(), 'lr': initial_lr},
                  {'params': self.clip_model.clip_model.visual.parameters(), 'lr': initial_lr},
                  {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}, ]
        return params
