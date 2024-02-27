# Training GPT-2 with Explicit Attention Biases

We provide the code and pretrained checkpoints for the experiments in Section 5.2 on "Explicit attention biases". The code for training GPT-2 is based on the open-source [nanoGPT](https://github.com/karpathy/nanoGPT) repository.

---
<p align="center">
<img src="../assets/equation.png" width=80% height=80% 
class="center">
</p>
We propose to augment the self-attention mechanism with explicit attention biases, by inserting auxiliary key and value parameters.  

[model_attn_bias.py](model_attn_bias.py) contains the model definition of GPT-2 augmented with explicit attention biases.

## Setup

- *data*: Follow [here](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#reproducing-gpt-2) to setup the training and validation data from OpenWebText2.

- *pretrained models*: Here we provide the model checkpoints for three GPT-2 models we trained, each with 50k iterations

| model name | download path | validation perplexity |
|:---:|:---:|:---:|
| default | [model](https://drive.google.com/file/d/1_oiybR7wmJ5ibPZMM3sGjtJRqVpMp0H6/view?usp=drive_link) | 3.04 |
| sink | [model](https://drive.google.com/file/d/1ZnhFxN-A7qc9Cghcp_jLpxQXDRE2BqSc/view?usp=drive_link) | 3.04 |
| attn_bias | [model](https://drive.google.com/file/d/1jSpGpNGqJ9Ff_goqoFjSRA5EN7Qmdv1U/view?usp=drive_link) | 3.04 |

**Note**: For the config files in [config](config), set `out_dir` to the directory of the downloaded pretrained models and `data_dir` to the directories of the prepared OpenWebText2 dataset.

## Evalutate

Running the following commands will evaluate the three GPT-2 checkpoints.
```sh
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_default.py ### gpt2 default architecture
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_sink.py ### gpt2 sink token
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_attn_bias.py ### gpt2 attention biases
```

## Training
Running the following commands will train the three GPT-2 models from scratch: (can adjust the number of GPUs for training on multiple GPUs)
```sh
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_default.py ### gpt2 default architecture
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_sink.py ### gpt2 sink token
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_attn_bias.py ### gpt2 attention biases
```

## Analysis
We provide the commands for visualizing the activaiton magnitudes of an intermediate feature and also layerwise largest activation magnitudes:
```sh
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_default.py
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_sink.py
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_attn_bias.py
```