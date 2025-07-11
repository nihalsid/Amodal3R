<h1 align="center">Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images</h1>
  <p align="center">
    <a href="https://sm0kywu.github.io/CV/CV.html">Tianhao Wu</a>
    ·
    <a href="https://chuanxiaz.com/">Chuanxia Zheng</a>
    ·
    <a href="https://www.singaporetech.edu.sg/directory/faculty/frank-guan">Frank Guan</a>
    .
    <a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a>
    .
    <a href="https://personal.ntu.edu.sg/astjcham/index.html">Tat-Jen Cham</a>

  </p>
  <h3 align="center">ICCV 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2503.13439">Paper</a> | <a href="https://sm0kywu.github.io/Amodal3R/">Project Page</a> | <a href="https://huggingface.co/Sm0kyWu/Amodal3R">Pretrain Weight</a> | <a href="https://huggingface.co/spaces/Sm0kyWu/Amodal3R">Demo</a></h3>
  <div align="center"></div>
</p>

### Demo Video

<img src="asset/teaser.gif" width="100%" alt="Demo Video">

### Setup
This code has been tested on Ubuntu 22.02 with torch 2.4.0 & CUDA 11.8. We sincerely thank [TRELLIS](https://github.com/Microsoft/TRELLIS) for providing the environment setup and follow exactly as their instruction in this work.

Create a new conda environment named `amodal3r` and install the dependencies:
```sh
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```
The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
```sh
Usage: setup.sh [OPTIONS]
Options:
    -h, --help              Display this help message
    --new-env               Create a new conda environment
    --basic                 Install basic dependencies
    --train                 Install training dependencies
    --xformers              Install xformers
    --flash-attn            Install flash-attn
    --diffoctreerast        Install diffoctreerast
    --vox2seq               Install vox2seq
    --spconv                Install spconv
    --mipgaussian           Install mip-splatting
    --kaolin                Install kaolin
    --nvdiffrast            Install nvdiffrast
    --demo                  Install all dependencies for demo
```

### Pretrained models
We have provided our pretrained weights of both sparse structure module and SLAT module on [HuggingFace](https://huggingface.co/Sm0kyWu/Amodal3R).

### Training
To train the sparse structure module with our designed mask-weighted cross-attention and occlusion-aware attention, please run:
```sh
. ./train_ss.sh
```
To train the sparse structure module with our designed mask-weighted cross-attention and occlusion-aware attention, please run:
```sh
. ./train_slat.sh
```
The output folder where the model will be saved can be changed by modifying ``--vis'' parameter in the script.


### inference
We have prepared examples under ./example folder. It supports both single and multiple image as input. For inference, please run:
```sh
python ./inference.py
```

If you want to try on you own data. You should prepare: 1) original image and 2) mask image (background is white (255,255,255), visible area is gray (188,188,188), occluded area is black (0,0,0)).

You can use [Segment Anything](https://github.com/facebookresearch/segment-anything) to obtain the corresponding mask, which is used for our in-the-wild examples in the paper and also in our demo.



