import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder


class CachedBLIPDataset(ImageFolder):
    def __init__(self, root, captions_json, transform=None):
        """
        初始化 CachedBLIPDataset。

        Args:
            root (str): 数据集根目录，与 ImageFolder 格式一致
            captions_json (str): JSON 文件路径，保存 blip 生成的描述（{image_path: caption} 格式）
            transform (callable, optional): 图像变换
        """
        # 调用 ImageFolder 的初始化
        super().__init__(root, transform=transform)

        # 加载 blip 生成的描述
        with open(captions_json, "r") as f:
            self.captions = json.load(f)

    def __getitem__(self, index):
        """
        重载 __getitem__ 方法，添加 blip 描述支持。

        Args:
            index (int): 数据索引

        Returns:
            tuple: (image, target, caption, index) -> 图像张量、类别索引、文本描述和样本索引
        """
        # 获取图像和类别（父类方法会返回 (image, target)）
        image, target = super().__getitem__(index)

        # 获取样本路径（通过 self.samples 访问所有图像路径）
        image_path, _ = self.samples[index]

        # 获取对应的 blip 描述
        caption = self.captions.get(image_path, "No caption available")

        return image, target, caption


def get_transform():
    # 公共的标准化转换配置
    normalize_transform = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),  # Normalization: mean
        std=(0.26862954, 0.26130258, 0.27577711)  # Normalization: std
    )

    # 训练模式的转换
    common_transforms = [transforms.Resize([256, 256]),  # Resize for uniformity
        transforms.RandomCrop(224),  # Random crop for data augmentation
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.ToTensor(),  # Convert to Tensor
        normalize_transform]

    # 测试模式的转换
    test_transforms = [transforms.Resize([224, 224]),  # Resize for testing
        transforms.ToTensor(), normalize_transform]

    transform_train = transforms.Compose(common_transforms)
    transform_test = transforms.Compose(test_transforms)
    return transform_train, transform_test


def get_visda_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Define the transformation pipeline for training
    transform_train = transforms.Compose([transforms.Resize([224, 224]),  # Resize images for uniformity
                                          transforms.RandomHorizontalFlip(),
                                          # Data augmentation via random horizontal flip
                                          transforms.CenterCrop(224),  # Center crop to 224x224
                                          transforms.ToTensor(),  # Convert image to Tensor
                                          transforms.Normalize(mean, std)  # Normalize the tensor with mean and std
                                          ])

    # Define the transformation pipeline for testing
    transform_test = transforms.Compose([transforms.Resize([224, 224]),  # Resize images for uniformity
                                         transforms.CenterCrop(224),  # Center crop to 224x224
                                         transforms.ToTensor(),  # Convert image to Tensor
                                         transforms.Normalize(mean, std)  # Normalize the tensor with mean and std
                                         ])

    return transform_train, transform_test



def get_dset_loader_imcapda(args):
    transform_train, transform_test = get_visda_transform() if args.datasets == "visda" else get_transform()

    def get_dataset(data_dir, domain, vlm_text, suffix, transform):
        """辅助函数，用于简化数据集的创建"""
        return CachedBLIPDataset(os.path.join(data_dir, domain) + suffix,
            f"VLM-Text/{vlm_text}/{vlm_text}_captions_{domain}{suffix}.json", transform)

    train_suffix, test_suffix = ("_train", "_test") if args.datasets == "minidomainnet" else ("", "")
    source_dataset = get_dataset(args.data_dir, args.src_domain, args.vlm_text, train_suffix, transform_train)
    target_dataset = get_dataset(args.data_dir, args.tgt_domain, args.vlm_text, train_suffix, transform_train)

    if args.fuse_mode:
        test_dataset = get_dataset(args.data_dir, args.tgt_domain, args.vlm_text, test_suffix, transform_test)
    else:
        test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.tgt_domain) + test_suffix, transform=transform_test)

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return source_loader, target_loader, test_loader



def get_dset_loader_imcapda_sfda(args):
    transform_train, transform_test = get_visda_transform() if args.datasets == "visda" else get_transform()

    def get_dataset(data_dir, domain, vlm_text, suffix, transform):
        """辅助函数，用于简化数据集的创建"""
        return CachedBLIPDataset(os.path.join(data_dir, domain) + suffix,
            f"VLM-Text/{vlm_text}/{vlm_text}_captions_{domain}{suffix}.json", transform)

    train_suffix, test_suffix = ("_train", "_test") if args.datasets == "minidomainnet" else ("", "")
    target_dataset = get_dataset(args.data_dir, args.tgt_domain, args.vlm_text, train_suffix, transform_train)

    if args.fuse_mode:
        test_dataset = get_dataset(args.data_dir, args.tgt_domain, args.vlm_text, test_suffix, transform_test)
    else:
        test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.tgt_domain) + test_suffix, transform=transform_test)

    target_train_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,drop_last=True)
    target_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return  target_train_loader, target_test_loader










