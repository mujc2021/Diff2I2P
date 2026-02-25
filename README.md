# Diff$^2$I2P: Differentiable Image-to-Point Cloud Registration with Diffusion Prior

[ICCV 2025] [Diff$^2$I2P: Differentiable Image-to-Point Cloud Registration with Diffusion Prior](https://arxiv.org/abs/2507.06651).

[Juncheng Mu](https://mujc2021.github.io/), Chengwei Ren, [Weixiang Zhang](https://weixiang-zhang.github.io/), [Liang Pan](https://ethan7899.github.io/), [Xiao-ping Zhang](https://sites.google.com/view/xiaopingzhang/), and [Yue Gao](https://www.gaoyue.org/).

## Introduction

Learning cross-modal correspondences is essential for image-to-point cloud (I2P) registration. Existing methods achieve this mostly by utilizing metric learning to enforce feature alignment across modalities, disregarding the inherent modality gap between image and point data. Consequently, this paradigm struggles to ensure accurate cross-modal correspondences. To this end, inspired by the cross-modal generation success of recent large diffusion models, we propose Diff$^2$I2P, a fully Differentiable I2P registration framework, leveraging a novel and effective Diffusion prior for bridging the modality gap. Specifically, we propose a Control-Side Score Distillation (CSD) technique to distill knowledge from a depth-conditioned diffusion model to directly optimize the predicted transformation. However, the gradients on the transformation fail to backpropagate onto the cross-modal features due to the non-differentiability of correspondence retrieval and PnP solver. To this end, we further propose a Deformable Correspondence Tuning (DCT) module to estimate the correspondences in a differentiable way, followed by the transformation estimation using a differentiable PnP solver. With these two designs, the Diffusion model serves as a strong prior to guide the cross-modal feature learning of image and point cloud for forming robust correspondences, which significantly improves the registration. Extensive experimental results demonstrate that Diff$^2$I2P consistently outperforms SoTA I2P registration methods, achieving over 7% improvement in registration recall on the 7-Scenes benchmark.

## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
# under cuda 11.8
conda create -n diff2i2p python==3.8
conda activate diff2i2p

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py38_cu118_pyt241.tar.bz2

pip install -r requirements.txt

cd vision3d
python setup.py build develop
cd ..

```

The code has been tested on Python 3.8, PyTorch 2.4.1, Ubuntu 22.04, GCC 11.4 and CUDA 11.8.

## Training

### Data preparation

The dataset can be downloaded [here](https://drive.google.com/file/d/1nEpkIDOZopLITlKhDYSIQzr-B7Wy-TZU/view?usp=sharing). The data should be organized as follows:

```text
--data
    --7Scenes--metadata
        |--data
            |--chess
            |--...
            |--stairs
    --RGBDScenesV2--metadata
        |--data
            |----rgbd-scenes-v2-scene_01
            |--...
            |--rgbd-scenes-v2-scene_14
```

### Training

The training scripts for **7Scenes** and **RGBD-Scenes-V2** are located in the `experiments/` directory.
Here we take **7Scenes** as an example:

#### Single-GPU Training

```bash
cd experiments/7scenes
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

#### Multi-GPU Training

```bash
cd experiments/7scenes
CUDA_VISIBLE_DEVICES=[GPUS] python -m torch.distributed.launch \
    --master_port=[PORT] \
    --nproc_per_node=[N_GPUS] \
    trainval.py
    
## example

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --master_port=29501 \
    --nproc_per_node=4 \
    trainval.py
```


## Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh [EPOCH]
```

`EPOCH` is the epoch id.

### Pre-trained Weights

We provide pre-trained weights [here](https://github.com/mujc2021/Diff2I2P/releases/tag/ckpts).

Use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/path/to/Diff2I2P/ckpts/7scenes.pth
CUDA_VISIBLE_DEVICES=0 python eval.py --test_epoch=-1
```

## Citation

```bibtex
@inproceedings{mu2025diff2i2p,
  title={Diff2I2P: Differentiable Image-to-Point Cloud Registration with Diffusion Prior},
  author={Mu, Juncheng and Ren, Chengwei and Zhang, Weixiang and Pan, Liang and Zhang, Xiao-Ping and Gao, Yue},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={25777--25787},
  year={2025}
}
```

## Acknowledgements

- [2D3DMATR](https://github.com/minhaolee/2D3DMATR)
- [diffusers](https://github.com/huggingface/diffusers)
