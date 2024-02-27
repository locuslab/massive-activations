import os

import torch
import timm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .model_dict import MODEL_DICT_LLMs


def load_llm(args):
    print(f"loading model {args.model}")
    model_name, cache_dir = MODEL_DICT_LLMs[args.model]["model_id"], MODEL_DICT_LLMs[args.model]["cache_dir"]

    if "falcon" in args.model or "mpt" in args.model or "phi" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, token=args.access_token)
    elif "mistral" in args.model or "pythia" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=args.revision, torch_dtype=torch.float16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", token=args.access_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", token=args.access_token)
    model.eval()

    if "mpt" in args.model or "pythia" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=args.access_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=args.access_token)

    device = torch.device("cuda:0")
    if "mpt_30b" in args.model:
        device = model.hf_device_map["transformer.wte"]
    elif "30b" in args.model or "65b" in args.model or "70b" in args.model or "40b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = torch.device("cuda:"+str(model.hf_device_map["lm_head"]))

    if "llama2_13b" == args.model:
        # device = torch.device("cuda:"+str(model.hf_device_map["lm_head"]))
        device = torch.device("cuda:1")

    seq_len=4096
    if "llama" in args.model or "mistral" in args.model:
        layers = model.model.layers
        hidden_size = model.config.hidden_size 
    elif "falcon" in args.model:
        layers = model.transformer.h 
        hidden_size = model.config.hidden_size 
    elif "mpt" in args.model:
        layers = model.transformer.blocks 
        hidden_size = model.config.d_model
        seq_len=2048
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        hidden_size = model.config.hidden_size
        seq_len = 2048
    elif "gpt2" in args.model:
        layers = model.transformer.h 
        hidden_size = model.transformer.embed_dim 
        seq_len = 1024
    elif "pythia" in args.model:
        layers = model.gpt_neox.layers 
        hidden_size = model.gpt_neox.config.hidden_size 
        seq_len = 2048
    elif "phi-2" in args.model: 
        layers = model.model.layers 
        hidden_size = model.config.hidden_size 

    return model, tokenizer, device, layers, hidden_size, seq_len

def load_vit(args):
    if args.model_family == "mae":
        patch_size=14 if args.model_size == "huge" else 16
        model = timm.create_model(f'vit_{args.model_size}_patch{patch_size}_224.mae', pretrained=True)
    elif args.model_family == "openai_clip":
        patch_size=14 if args.model_size == "large" else 16
        model = timm.create_model(f"vit_{args.model_size}_patch{patch_size}_clip_224.openai", pretrained=True)
    elif args.model_family == "dinov2":
        model = timm.create_model(f"vit_{args.model_size}_patch14_dinov2.lvd142m", pretrained=True, num_classes=1000)
    elif args.model_family == "dinov2_reg":
        model = timm.create_model(f"vit_{args.model_size}_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=1000)

    model = model.cuda()
    model = model.eval()

    layers = model.blocks

    data_config = timm.data.resolve_model_data_config(model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    return model, layers, val_transform

def load_dinov2_linear_head(args):
    assert "dinov2" in args.model_family, "this function is only for dinov2 models"
    if args.model_family == "dinov2_reg":
        linear_head_path = os.path.join(args.linear_head_path, f"dinov2_vit{args.model_size[0]}14_reg4_linear_head.pth")
    elif args.model_family == "dinov2":
        linear_head_path = os.path.join(args.linear_head_path, f"dinov2_vit{args.model_size[0]}14_linear_head.pth")

    linear_head_weights = torch.load(linear_head_path)
    return linear_head_weights