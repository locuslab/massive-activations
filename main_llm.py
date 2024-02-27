import argparse
import os
from importlib.metadata import version

import numpy as np
import torch

import lib
import monkey_patch as mp

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument('--seed',type=int, default=1, help='Seed for sampling the calibration data.')
    parser.add_argument("--revision", type=str, default="main")

    parser.add_argument("--exp1", action="store_true", help="plot 3d feature")
    parser.add_argument("--exp2", action="store_true", help="layerwise analysis")
    parser.add_argument("--exp3", action="store_true", help="intervention analysis")
    parser.add_argument("--exp4", action="store_true", help="attention visualization")
    parser.add_argument("--layer_id", type=int, default=1)
    parser.add_argument("--reset_type", type=str, default="set_mean")
    parser.add_argument("--access_token", type=str, default="type in your access token here")
    parser.add_argument("--savedir", type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print("use device ", device)

    if args.exp1: ### visualize the output feature of a layer in LLMs
        layer_id = args.layer_id - 1
        if "llama2" in args.model:
            mp.enable_llama_custom_decoderlayer(layers[layer_id], layer_id)
        elif "mistral" in args.model:
            mp.enable_mistral_custom_decoderlayer(layers[layer_id], layer_id)
        elif "phi-2" in args.model:
            mp.enable_phi2_custom_decoderlayer(layers[layer_id], layer_id)
        else:
            raise ValueError(f"model {args.model} not supported")

        stats = {}
        seq = "Summer is warm. Winter is cold."
        valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            model(valenc)

        seq_decoded = []
        for i in range(valenc.shape[1]):
            seq_decoded.append(tokenizer.decode(valenc[0,i].item()))

        stats[f"seq"] = seq_decoded
        feat_abs = layers[layer_id].feat.abs()

        stats[f"{layer_id}"] = feat_abs

        lib.plot_3d_feat(stats, layer_id, args.model, args.savedir)

    elif args.exp2: ### visualize the layerwise top activation magnitudes
        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            if "llama2" in args.model:
                mp.enable_llama_custom_decoderlayer(layer, layer_id)
            elif "mistral" in args.model:
                mp.enable_mistral_custom_decoderlayer(layer, layer_id)
            elif "phi-2" in args.model:
                mp.enable_phi2_custom_decoderlayer(layers[layer_id], layer_id)
            else:
                raise ValueError(f"model {args.model} not supported")

        testseq_list = lib.get_data(tokenizer, nsamples=10, seqlen=seq_len, device=device)

        stats = []
        for seqid, testseq in enumerate(testseq_list):
            print(f"processing seq {seqid}")
            with torch.no_grad():
                model(testseq)

            seq_np = np.zeros((4, len(layers)))
            for layer_id in range(len(layers)):
                feat_abs = layers[layer_id].feat.abs()
                sort_res = torch.sort(feat_abs.flatten(), descending=True)
                seq_np[:3, layer_id] = sort_res.values[:3]
                seq_np[3, layer_id] = torch.median(feat_abs)

            stats.append(seq_np)

        lib.plot_layer_ax(stats, args.model, args.savedir)

    elif args.exp3: ### intervention analysis
        layer = layers[args.layer_id-1]
        lib.setup_intervene_hook(layer, args.model, args.reset_type)

        f = open(os.path.join(args.savedir, f"{args.model}_{args.reset_type}.log"), "a")

        ds_list = ["wikitext", "c4", "pg19"]
        res = {}
        for ds_name in ds_list:
            ppl = lib.eval_ppl(ds_name, model, tokenizer, args.seed, device)
            res[ds_name] = ppl 
            print(f"{ds_name} ppl: {ppl}", file=f, flush=True)

    elif args.exp4:
        layer_id = args.layer_id - 1
        if "llama2" in args.model:
            modified_attn_layer = mp.enable_llama_custom_attention(layers[layer_id], layer_id)
        elif "mistral" in args.model:
            modified_attn_layer = mp.enable_mistral_custom_attention(layers[layer_id], layer_id)
        elif "phi-2" in args.model:
            modified_attn_layer = mp.enable_phi2_custom_attention(layers[layer_id], layer_id)
        else:
            raise ValueError(f"model {args.model} not supported")

        seq = "The following are multiple choice questions (with answers) about machine learning.\n\n A 6-sided die is rolled 15 times and the results are: side 1 comes up 0 times;"
        valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            model(valenc)

        attn_logit = layers[layer_id].self_attn.attn_logits.detach().cpu()
        lib.plot_attn(attn_logit, args.model, layer_id, args.savedir)