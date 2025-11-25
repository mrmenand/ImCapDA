### ImCapDA: Fine-tuning CLIP via Image Captions for Unsupervised Domain Adaptation

#### 1. Prepare Datasets

To begin with, download the domain adaptation datasets by following the instructions provided in the [dataset documentation](https://github.com/zdcuob/transferlearning/blob/891cb7c89e000dc18414d96a64f3b97ea59b4c3d/data/dataset.md).

Once downloaded, ensure the dataset path is correctly set in the following locations:
- Update the path in `VLM-Text/blip{}/blip{}_caption_{domain}.json` to point to the correct dataset location.
- Modify the `--data_dir` default parameter in `train_imcapda.py` or `configs/office_home.yaml` to reflect the correct dataset directory.


#### 2. Training

You can use different networks to train on various datasets depending on your specific objectives. Below are commands for training using **ViT-B/16** on different domain adaptation tasks.

**For ImCapDA**:

When training Art → Product on the Office-Home dataset, run the following command:

```bash
python3 train_imcapda.py --config configs/office_home.yaml --src_domain Art --tgt_domain Product --model_name ViT-B/16 
```

For Amazon → DSLR on the Office-31 dataset:从
```bash
python3 train_imcapda.py --config configs/office31.yaml --src_domain amazon --tgt_domain dslr --model_name ViT-B/16
```

**For ImCapFusionDA:**

Similarly, for training with ImCapFusionDA, use the following commands. c
```bash
python3 train_imcapfusiondac.py --config configs/office_home.yaml --src_domain Art --tgt_domain Product  --model_name ViT-B/16 
python3 train_imcapfusiondac.py --config configs/office31.yaml --src_domain amazon --tgt_domain dslr --model_name ViT-B/16
```


This code is heavily inspired by and closely follows the implementation found in the [VLP-UDA repository](https://github.com/Wenlve-Zhou/VLP-UDA).