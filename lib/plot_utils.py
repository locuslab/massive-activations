import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.patheffects import withStroke
from collections import defaultdict
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({
    # 'font.family': 'Arial',
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'figure.titlesize': 28
})
matplotlib.rcParams['text.usetex'] = False

MODEL_TITLE_DICT={"llama2_7b": "LLaMA-2-7B", "mistral_7b": "Mistral-7B", 
        "llama2_13b_chat": "LLaMA-2-13B-chat", "llama2_70b_chat": "LLaMA-2-70B-chat",
        "llama2_7b_chat": "LLaMA-2-7B-chat", "llama2_13b": "LLaMA-2-13B", "llama2_70b": "LLaMA-2-70B", 
        "mistral_moe":"Mixtral-8x7B", "falcon_7b": "Falcon-7B", "falcon_40b": "Falcon-40B", "phi-2": "Phi-2",
        "opt_7b":"OPT-7B", "opt_13b": "OPT-13B", "opt_30b": "OPT-30B", "opt_66b": "OPT-66B",
        "mpt_7b": "MPT-7B", "mpt_30b": "MPT-30B", "pythia_7b": "Pythia-7B", "pythia_12b": "Pythia-12B",
        "gpt2": "GPT-2", "gpt2_large": "GPT-2-Large", "gpt2_xl": "GPT-2-XL", "gpt2_medium": "GPT-2-Medium",
        "mistral_moe_instruct": "Mixtral-8x7B-Instruct", "mistral_7b_instruct": "Mistral-7B-Instruct"}


def plot_3d_feat_sub(ax, obj, seq_id, layer_id, model_name):
    num_tokens = len(obj[f"seq"])
    num_channels = obj[f"{layer_id}"].shape[2]
    inp_seq = obj[f"seq"]
    inp_seq = [x if x != "<0x0A>" else r"\n" for x in inp_seq]
    xdata = np.array([np.linspace(0,num_tokens-1,num_tokens) for i in range(num_channels)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_channels)])
    zdata = obj[f"{layer_id}"][0].abs().numpy().T
    ax.plot_wireframe(xdata, ydata, zdata, rstride=0, color="royalblue", linewidth=2.5)

    ax.set_xticks(np.linspace(0,num_tokens-1,num_tokens), inp_seq, 
                      rotation=50, fontsize=16)

    ax.set_zticks([0, 1000, 2000], ["0", "1k", "2k"], fontsize=15)
    ax.set_yticks([1415, 2533], [1415, 2533], fontsize=15, fontweight="heavy")
    ax.get_xticklabels()[0].set_weight("heavy")

    if seq_id in [0, 1]:
        ax.get_xticklabels()[3].set_weight("heavy")

    ax.set_title(MODEL_TITLE_DICT[model_name], fontsize=18, fontweight="bold", y=1.015)

    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="left",
         rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-4)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-1)
    ax.set_zlim(0,2400)

def plot_3d_feat(obj, layer_id, model_name, savedir):
    fig = plt.figure(figsize=(14,6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    # for i in range(3):
    ax = fig.add_subplot(1,1, 1, projection='3d')
    plot_3d_feat_sub(ax, obj, 0, layer_id, model_name)
    plt.savefig(os.path.join(savedir, f"{model_name}_layer_{layer_id+1}.png"), bbox_inches="tight", dpi=200)


def plot_layer_ax_sub(ax, mean, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey"]

    x_axis = np.arange(mean.shape[-1])+1
    for i in range(3):
        ax.plot(x_axis, mean[i], label=f"Top {i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    ax.plot(x_axis, mean[-1], label=f"Median", color=colors[-1], 
                     linestyle="-",  marker="v", markerfacecolor='none', markersize=5)

    ax.set_title(MODEL_TITLE_DICT[model_name], fontsize=18, fontweight="bold")

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel("Magnitudes", fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')

def plot_layer_ax(obj, model_name, savedir):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    mean = np.mean(obj,axis=0)
    plot_layer_ax_sub(axs, mean, model_name)
    leg = axs.legend(
        loc='center', bbox_to_anchor=(0.5, -0.25),
        ncol=4, fancybox=True, prop={'size': 14}
    )
    leg.get_frame().set_edgecolor('silver')
    leg.get_frame().set_linewidth(1.0)

    plt.savefig(os.path.join(savedir, f"{model_name}.png"), bbox_inches="tight", dpi=200)


def plot_attn_sub(ax, corr, model_name, layer_id):
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, square=True, ax=ax,
                      cmap="YlGnBu",cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect":50})

    ax.set_facecolor("whitesmoke") 
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=18)

    ax.tick_params(axis='x', which='major')
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.tick_params(left=False, bottom=False)
    ax.set_title(f"{MODEL_TITLE_DICT[model_name]}, Layer {layer_id+1}", fontsize=24, fontweight="bold")


def plot_attn(attn_logits, model_name, layer_id, savedir):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.75))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.15)

    corr = attn_logits.numpy()[0].mean(0)
    corr = corr.astype("float64")

    plot_attn_sub(axs, corr, model_name, layer_id)
    plt.savefig(os.path.join(savedir, f"{model_name}_layer{layer_id+1}.pdf"), bbox_inches="tight", dpi=200)