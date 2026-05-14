# AdaManip

This is the official codebase for [**AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning** (ICLR 2025)](https://arxiv.org/abs/2502.11124)

## Setup

1. **Create a Conda environment**

```sh
conda create -n adamanip python=3.8
conda activate adamanip
```

2. **Install PyTorch & IsaacGym**

```sh
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Download IsaacGym from [Nvidia Website](https://developer.nvidia.com/isaac-gym/download]) and Install the package following the instructions of the official [documentation](https://docs.robotsfan.com/isaacgym/install.html).

3. **Install Pointnet++ & pytorch3d**

+ Install [Pointnet2_Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch) following its instructions.
+ Git clone the [pytorch3d](https://github.com/facebookresearch/pytorch3d) and install it locally.

```sh
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```

4. **Install other dependencies**

```sh
pip install -r requirements.txt
```

## Download Assets

Download all assets from this [Google Drive link](https://drive.google.com/drive/folders/1ho2wSdd9hWNb6XuwVnFVkUakdNqVKXmb?usp=sharing). 

Place the downloaded `assets` folder in the root directory of this repository:

```
AdaManip/
├── assets/
├── scripts/
├── cfg/
├── ...
```

## Run Experiments

All scripts for data collection, policy training, and evaluation are provided in the `scripts` folder. The following example demonstrates usage with the open microwave task:

### Data Collection

```sh
sh scripts/microwave/collect_mv.sh
```

### Policy Training

```sh
sh scripts/train.sh microwave manip
```

### Evaluation

```sh
sh scripts/eval.sh microwave
```

Note: Configuration files are located in the **cfg** folder. Please ensure the model checkpoint path is correctly set before running evaluation.

## BibTeX

If you found AdaManip useful, please consider citing:

```tex
@inproceedings{wang2025adamanip,
    title={AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning},
    author={Wang, Yuanfei and Zhang, Xiaojie and Wu, Ruihai and Li, Yu and Shen, Yan and Wu, Mingdong and He, Zhaofeng and Wang, Yizhou and Dong, Hao},
    booktitle={International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=Luss2sa0vc}
  }
```

## Contact

If you have any suggestion or questions, please get in touch at yuanfei_wang@pku.edu.cn
