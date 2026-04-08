import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


def contrastive_loss(feature, center, t=5):
    feat_reshape = feature.unsqueeze(1).repeat(1, center.size(0), 1)
    prods = torch.exp(torch.sum(feat_reshape * center, dim=-1) / t)
    sum = torch.sum(prods, dim=-1)
    p = prods / sum.unsqueeze(1)
    contrastive_loss = -torch.sum(p * torch.log(p)) / p.size(0)

    return contrastive_loss


def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())  # 计算余弦相似度
    return similarity


def weight_contrastive(infonce=None, t_indx=None, labels=None, sor_img_con=None, all_ref_fea=None, tgt_pre_label=None):
    total_contrastive_loss = Variable(torch.tensor(0.).cuda())
    contrastive_label = torch.tensor([0]).cuda()
    all_sam_indx, all_in, _ = np.intersect1d(t_indx, t_indx, return_indices=True)
    # MarginNCE
    gamma = 0.

    nll = nn.NLLLoss()
    if len(all_in) > 0:
        for idx in range(len(all_in)):
            pairs4q = infonce.get_posAndneg(features=sor_img_con, labels=labels, tgt_label=tgt_pre_label,
                                            feature_q_idx=t_indx[all_in[idx]], co_fea=all_ref_fea[all_in[idx]].cuda())

            # calculate cosine similarity [-1 1]
            result = cosine_similarity(all_ref_fea[all_in[idx]].unsqueeze(0).cuda(), pairs4q)

            # MarginNCE
            # softmax
            numerator = torch.exp((result[0][0]) / gamma)
            denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
            # log
            result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
            # nll_loss
            contrastive_loss = nll(result, contrastive_label) * sam_confidence[t_indx[all_in[idx]]]
            total_contrastive_loss = total_contrastive_loss + contrastive_loss
        total_contrastive_loss = total_contrastive_loss / len(all_in)
    return total_contrastive_loss


def cluster_cont_loss(feature, center, t=5):
    feat_reshape = feature.unsqueeze(1).repeat(1, center.size(0), 1)
    prods = torch.exp(torch.sum(feat_reshape * center, dim=-1) / t)
    sum = torch.sum(prods, dim=-1)
    p = prods / sum.unsqueeze(1)
    contrastive_loss = -torch.sum(p * torch.log(p)) / p.size(0)

    return contrastive_loss


def weigh_cont_loss(feature, inst, center, t=5):
    # feat_reshape = feature.unsqueeze(1).repeat(1, inst.size(0), 1)
    prods = torch.exp(torch.sum(feature.mm(inst.t()), dim=-1) / t)
    sum = torch.sum(prods, dim=-1)
    p = prods / sum

    # feat_reshape = feature.unsqueeze(1).repeat(1, center.size(0), 1)
    prods = torch.exp(torch.sum(feature.mm(center.t()), dim=-1) / 1)
    sum = torch.sum(prods, dim=-1)
    p_c = prods / sum

    w_contrastive_loss = -torch.sum(p_c * torch.log(p))

    return w_contrastive_loss


def entroy_mim(x, eps=1e-5):
    p = F.softmax(x, dim=-1)
    entroy = -torch.mean(torch.sum(p * torch.log(p + eps), 1))
    return entroy


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + target_weight / torch.sum(
            target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCEWithLogitsLoss(reduction='none')(ad_out, dc_target)) / torch.sum(
            weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


class MomentumSoftmax:
    def __init__(self, num_class, m=1):
        self.softmax_vector = torch.zeros(num_class).detach() + 1.0 / num_class
        self.m = m
        self.num = m

    def update(self, mean_softmax, num=1):
        self.softmax_vector = ((self.softmax_vector * self.num) + mean_softmax * num) / (self.num + num)
        self.num += num

    def reset(self):
        # print(self.softmax_vector)
        self.num = self.m


def adentropy(F1, feat, lamda, eta=1.0):
    _, _, out_softmax = F1(feat, reverse=True, eta=eta)
    loss_adent = lamda * torch.mean(torch.sum(out_softmax * (torch.log(out_softmax + 1e-5)), 1))
    return loss_adent


def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)
    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


def bsp_loss(feature):
    train_bs = feature.size(0) // 2
    feature_s = feature.narrow(0, 0, train_bs)
    feature_t = feature.narrow(0, train_bs, train_bs)
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    sigma *= 0.0001
    return sigma


def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

    # return -torch.mean(torch.log(torch.mean(predictions, 0) + 1e-6))


class MinimumClassConfusionLoss(nn.Module):

    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0),
                                          predictions)  # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


def get_entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def compute_im_loss(logits):
    softmax_out = torch.nn.Softmax(dim=1)(logits)
    entropy_loss = torch.mean(get_entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
    im_loss = entropy_loss - gentropy_loss
    return im_loss


def vlm_text_contrastive_loss(features_img,vlm_text_features,temperature=0.07):
    # 计算目标域图像和文本的相似性矩阵
    similarity_matrix = torch.matmul(features_img, vlm_text_features.T)

    # 计算对比损失
    logits_per_image = similarity_matrix / temperature  # 图像->文本
    logits_per_text = similarity_matrix.T / temperature  # 文本->图像

    # 使用对角线上的元素作为正样本，其余为负样本
    labels = torch.arange(features_img.size(0), device=features_img.device)  # # 正确匹配的索引

    loss_image_to_text = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_text_to_image = torch.nn.CrossEntropyLoss()(logits_per_text, labels)

    contrastive_loss = (loss_image_to_text + loss_text_to_image) / 2

    return contrastive_loss





