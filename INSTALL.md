# Installation

## Dependencies
Create an new conda virtual environment
```sh
conda create -n massive-activations python=3.9 -y
conda activate massive-activations
```

Install [Pytorch](https://pytorch.org/)>=2.0.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.15.0 following official instructions. For example:
```sh
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

Install additional dependencies:
```sh
pip install timm==0.9.12 transformers==4.36.0 accelerate==0.23.0 datasets==2.14.5 matplotlib==3.8.0 seaborn sentencepiece protobuf
```

## Pretrained Models

- **LLM Models**: To use pretrained LLM models, update the `CACHE_DIR_BASE` variable in the [model_dict.py](lib/model_dict.py) file to point to the directory containing the pretrained model weights.

- **DINOv2-reg Models**: To use the DINOv2-reg model for linear classification, download the pretrained linear classification head from this [link](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-heads---image-classification). Set the `--linear_head_path` argument in the [main_vit.py](main_vit.py) script to the directory where you've stored the downloaded weights.
