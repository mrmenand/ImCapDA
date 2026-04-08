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
