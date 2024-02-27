import os
import types

import torch
import torch.nn as nn
from timm.utils import accuracy

from .load_data import get_test_data


def fix_reg_mean(args, model):
    """
    Modifies the forward pass of each layer in a ViT model to set register token features to pre-computed means.
    """
    def custom_layer_forward(self, x: torch.Tensor) -> torch.Tensor:
        for reg_id in range(4):
            reg_id = int(reg_id)
            cur_reg_feat = self.reg_feat[reg_id].clone()
            x[:,reg_id+1,:] = cur_reg_feat.reshape(1,-1).repeat(x.shape[0],1)

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    # Load the pre-computed means of resgiter token features for the DINOv2-reg ViT-`args.model_size`
    imagenet10k_reg_feat_mean = torch.load(os.path.join(args.reg_feat_mean, f"{args.model_size}_imagemean_bs10k_trainaug.pt"))
    for layer_id in range(len(model.blocks)):
        layer = model.blocks[layer_id]
        layer.reg_feat = imagenet10k_reg_feat_mean[layer_id]

        # Replace the layer's forward function with the custom forward function defined above
        layer.forward = types.MethodType(
            custom_layer_forward, layer 
        )

def setup_dinov2_model_for_eval(model, linear_head_weights):
    """
    Configures a DINOv2 model for ImageNet evaluation by setting up a new linear head with given weights and a custom forward function.

    Args:
    model: The DINOv2 pretrained ViT models.
    linear_head_weights: A dictionary containing the weights and bias for the new linear head.
    """
    in_features, out_features = linear_head_weights["weight"].shape
    model.head = nn.Linear(in_features, out_features, bias=True)
    model.head.weight.data = linear_head_weights["weight"]
    model.head.bias.data = linear_head_weights["bias"]
    model.head.cuda()

    def custom_forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = torch.cat([x[:,0], x[:, self.num_prefix_tokens:].mean(dim=1)], dim=-1)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)

    model.forward_head = types.MethodType(
        custom_forward_head, model
    )

def test_imagenet(model, dataloader):
    acc = 0
    cnt = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0 and batch_idx > 0:
            print(f"batch idx {batch_idx} acc {acc/cnt}")
        images = batch[0].cuda()
        target = batch[-1].cuda()

        with torch.no_grad():
            output = model(images)
        acc1, _ = accuracy(output, target, topk=(1, 5))
        acc += acc1 * images.shape[0]
        cnt += images.shape[0]

    return acc/cnt 

@torch.no_grad()
def eval_ppl(dataset_name, model, tokenizer, seed):
    print(f"Evaluating on {dataset_name}")
    seqlen=4096
    testseq_list = get_test_data(
        dataset_name, seed=seed, tokenizer=tokenizer, seqlen=seqlen, device="cuda:0"
    )

    nlls = []
    with torch.no_grad():
        for test_seq in testseq_list:
            lm_logits = model(test_seq).logits

            shift_logits = lm_logits[:, :-1, :].contiguous()   ## shape: [1, 2047, 50272]
            shift_labels = test_seq[:, 1:]             ## shape: shape: [1, 2047]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            neg_log_likelihood = loss.float() * test_seq.numel()
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (len(testseq_list) * seqlen))
    return ppl.item()