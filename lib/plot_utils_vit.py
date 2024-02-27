import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configuration settings for matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'figure.titlesize': 28
})
matplotlib.rcParams['text.usetex'] = False

def plot_3d_feat_vit_sub(ax, feat, layer_id, model_name, model_size):
    model_title={"dinov2_reg": f"DINOv2-reg ViT-{model_size}", "mistral_7b": "Mistral-7B", 
               "llama2_13b": "LLaMA-2-13B", "llama2_70b": "LLaMA-2-70B", "mistral_moe":"Mixtral-8x7B"}

    num_channels = feat.shape[2]

    inp_seq = ["CLS", "reg 1", "reg 2", "reg 3", "reg 4",
               "patch 1", "patch 2", "patch i", "patch n"]

    xbase_index = [0,1,2,3,4,5,7,9]
    num_tokens = len(xbase_index)
    xdata = np.array([xbase_index for i in range(num_channels)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_channels)])
    zdata = feat[0,:num_tokens,:].abs().numpy().T
    ax.plot_wireframe(xdata, ydata, zdata, rstride=0, color="royalblue", linewidth=2.5)

    ax.set_title(model_title[model_name]+f", Layer {layer_id+1}", fontsize=20, fontweight="bold", y=1.015)

    ax.set_yticks([179, 999], [179, 999], fontsize=15, fontweight="heavy")

    xbase_index = [0,1,2,3,4,]
    inp_seq = ["CLS", "reg 1", "reg 2", "reg 3", "reg 4"]
    ax.set_xticks(xbase_index, inp_seq, rotation=60, fontsize=16)
    ax.tick_params(axis='x', which='major', pad=-4)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center", rotation_mode="anchor")

    ax.set_zticks([0, 500, 1000], ["0", "500", "1k"], fontsize=16)
    ax.get_xticklabels()[3].set_weight("heavy")
    plt.setp(ax.get_yticklabels(), ha="left", va="center",rotation_mode="anchor")
    plt.setp(ax.get_zticklabels(), ha="left", va="top", rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-5)
    ax.tick_params(axis='y', which='major', pad=-3)
    ax.tick_params(axis='z', which='major', pad=-5)

def plot_3d_feat_vit(feat, layer_id, model_name, model_size, savedir):
    fig = plt.figure(figsize=(8,6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.)

    ax = fig.add_subplot(1,1, 1, projection='3d')
    plot_3d_feat_vit_sub(ax, feat, layer_id, model_name, model_size)
    plt.savefig(os.path.join(savedir, f"{model_name}_{model_size}_layer_{layer_id+1}.png"), bbox_inches="tight", dpi=200)


def plot_layer_ax_vit_sub(ax, mean, model_family, model_size, colors=["royalblue", "darkorange", "forestgreen", "black"]):
    model_title={"dinov2_reg": "DINOv2-reg", 
                 "dinov2": "DINOv2", "mae": "MAE", "open_clip": "Open CLIP", "openai_clip": "OpenAI CLIP", 
                 "vit_orig": "ViT", "samvit": "SAM-ViT"}

    x_axis = np.arange(mean.shape[-1])+1
    for i in range(3):
        ax.plot(x_axis, mean[i], label=f"Top {i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    ax.plot(x_axis, mean[-1], label=f"median", color=colors[-1], 
                     linestyle="-",  marker="v", markerfacecolor='none', markersize=5)

    ax.set_title(model_title[model_family]+f" ViT-{model_size}", fontsize=18, fontweight="bold")
    ax.set_ylabel("Magnitudes", fontsize=18)

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=18, labelpad=4.0)
    ax.tick_params(axis='x', which='major', pad=2.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')
    ax.grid(axis='y', color='0.75')

def plot_layer_ax_vit(mean, model_family, model_size, savedir):
    fig = plt.figure(figsize=(8,6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.)

    ax = fig.add_subplot(1,1, 1)
    plot_layer_ax_vit_sub(ax, mean, model_family, model_size)
    plt.savefig(os.path.join(savedir, f"{model_family}_{model_size}.png"), bbox_inches="tight", dpi=200)