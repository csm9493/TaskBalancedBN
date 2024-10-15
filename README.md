# Rebalancing Batch Normalization for Exemplar-based Class-Incremental Learning [CVPR 2023]

Sungmin Cha, Sungjun Cho, Dasol Hwang, Sunwon Hong, Moontae Lee, and Taesup Moon

Paper: [[Arxiv](https://arxiv.org/abs/2201.12559)]

Abstract: Batch Normalization (BN) and its variants has been extensively studied for neural nets in various computer vision tasks, but relatively little work has been dedicated to studying the effect of BN in continual learning. To that end, we develop a new update patch for BN, particularly tailored for the exemplar-based class-incremental learning (CIL). The main issue of BN in CIL is the imbalance of training data between current and past tasks in a mini-batch, which makes the empirical mean and variance as well as the learnable affine transformation parameters of BN heavily biased toward the current task -- contributing to the forgetting of past tasks. While one of the recent BN variants has been developed for "online" CIL, in which the training is done with a single epoch, we show that their method does not necessarily bring gains for "offline" CIL, in which a model is trained with multiple epochs on the imbalanced training data. The main reason for the ineffectiveness of their method lies in not fully addressing the data imbalance issue, especially in computing the gradients for learning the affine transformation parameters of BN. Accordingly, our new hyperparameter-free variant, dubbed as Task-Balanced BN (TBBN), is proposed to more correctly resolve the imbalance issue by making a horizontally-concatenated task-balanced batch using both reshape and repeat operations during training. Based on our experiments on class incremental learning of CIFAR-100, ImageNet-100, and five dissimilar task datasets, we demonstrate that our TBBN, which works exactly the same as the vanilla BN in the inference time, is easily applicable to most existing exemplar-based offline CIL algorithms and consistently outperforms other BN variants.

-------


## Environment

See environment.yml

-------

## Implementation Guide

### Setting Up the Environment and Folder
1. Create the environment: conda env create -f environment.yml
2. Clone the repository: git clone https://github.com/csm9493/TaskBalancedBN.git
3. Navigate to the project directory: cd TaskBalancedBN
4. Create a folder for storing results: mkdir results data
5. Download [ImageNet_config.zip](https://drive.google.com/file/d/11Sx_iuaQCLhEM933ZUNsR3nBAHllJdUe/view?usp=share_link) and unzip this file in './data/'.
6. Update 'YOUR_IMAGENET_DATASET_DIRECTORY' in Line 63 of './src/datasets/base_dataset.py'.

### Run Class-Incremental Learning Experiments
1. Navigate to the project directory: cd TaskBalancedBN/src/
2. Run the ImageNet100 script: ./run_imagenet100.sh
3. Run the CIFAR100 script: ./run_cifar100.sh

### *Implementation of TaskBalancedBN(TBBN) and other normalization layers*

You can find the TBBN code in './TaskBalancedBN/src/networks/tbbn.py' and the code for other normalization layers in './TaskBalancedBN/src/networks/norm_layers.py.'

### Acknolwedgement

This code is implemented based on the framework for analysis of class-incremental learning ([FACIL](https://github.com/mmasana/FACIL)).

#### Note: For ViT models trained on CIFAR-100, we used the pre-trained model from HuggingFace(Ahmed9275/Vit-Cifar100).

-------
## Citation
@inproceedings{cha2023rebalancing,
  title={Rebalancing batch normalization for exemplar-based class-incremental learning},
  author={Cha, Sungmin and Cho, Sungjun and Hwang, Dasol and Hong, Sunwon and Lee, Moontae and Moon, Taesup},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20127--20136},
  year={2023}
}

-------