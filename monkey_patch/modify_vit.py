import torch 
import types

def vit_custom_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    self.feat = x.clone().detach().cpu().double()  ## this is to get the output feature of each layer;
    return x

def enable_vit_custom_block(layer, layer_id):
    layer.layer_id = layer_id 
    layer.forward = types.MethodType(vit_custom_block_forward, layer)


def vit_custom_attention_forward(self, x) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)

    # ###################################################
    self.attn_logits = attn.clone().detach().cpu().double()
    # ###################################################

    attn = attn.softmax(dim=-1)

    # ###################################################
    self.attn_probs = attn.clone().detach().cpu().double()
    # ###################################################

    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def enable_vit_custom_attention(layer, layer_id):
    modified_module = layer.attn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(vit_custom_attention_forward, modified_module)
