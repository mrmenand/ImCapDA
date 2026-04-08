import clip
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.cmkd import CMKD


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

class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        clip_model, _ = clip.load(args.model_name, device="cuda")

        prompt_template = getattr(args, "templates", "an image of a {}")
        classnames_lower = [name.replace("_", " ").lower() for name in args.classnames]
        naive_prompts = [prompt_template.format(classname) for classname in classnames_lower]
        prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])
        prompts = prompts.to(args.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.clip_model = clip_model
        self.text_features = text_features.detach().cuda()
        #self.text_features = torch.load("utils/zeroshot_weights_gpt_both.pt",weights_only=True).t().cuda()

    def forward(self, image):

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        # logits_per_image = logit_scale * image_features @ self.text_features.t()
        # logits_per_text = logit_scale *self.text_features @ image_features.t()
        # logits = (logits_per_image+logits_per_text.t())/2
        return image_features, logits


class ImCapDA(nn.Module):
    def __init__(self, args):
        super(ImCapDA, self).__init__()
        self.args = args

        clip_model = CLIP(args)
        self.output_dim = clip_model.clip_model.visual.output_dim
        self.clip_model = clip_model.cuda()

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.output_dim),
            nn.LayerNorm(self.output_dim, eps=1e-6),
            nn.Linear(self.output_dim, args.num_class, bias=False))
        self.classifier_layer.apply(weights_init_classifier)
        self.cmkd = CMKD(args)


    def forward(self, image ):
        self.clip_model.apply(fix_bn)
        image_features, logits_clip = self.clip_model(image)

        logits_fc = self.classifier_layer(image_features)

        return image_features,logits_clip, logits_fc

    def get_parameters(self, initial_lr=1.0):
        params = [
                  {'params': self.clip_model.clip_model.transformer.parameters(), 'lr': initial_lr},
                  {'params': self.clip_model.clip_model.visual.parameters(), 'lr': initial_lr},
                  {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}
                  ]
        return params

