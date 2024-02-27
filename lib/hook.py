import torch 

class Intervened_Layer:
    def __init__(self, model_name, reset_type):
        self.model_name = model_name 
        self.reset_type = reset_type

    def update(self, inp, out):
        out_feature = out[0]

        if self.reset_type == "set_mean":
            alpha = 1.0 
        elif self.reset_type == "set_zero":
            alpha = 0.0
        else:
            raise ValueError(f"reset_type {self.reset_type} not supported")

        if self.model_name == "llama2_13b":
            out_feature[:, 0, 4743] = - 1223.5 * alpha 
            out_feature[:, 0, 2100] = - 717.95 * alpha 
        elif self.model_name == "llama2_7b":
            feat_abs = out_feature.abs()
            sort_res = torch.sort(feat_abs.flatten(), descending=True)

            top_indices = sort_res.indices[0]
            token_dim = top_indices.item() // feat_abs.shape[2]

            if token_dim != 0:
                out_feature[:, token_dim, 2533] = 2546.8 * alpha 
                out_feature[:, token_dim, 1415] = - 1502.0 * alpha 

            out_feature[:, 0, 2533] = 767.6 * alpha 
            out_feature[:, 0, 1415] = - 458.55 * alpha 

        return (out_feature, *out[1:])

def setup_intervene_hook(layer, model_name, reset_type):
    update_layer = Intervened_Layer(model_name, reset_type)

    def add_batch():
        def modify_hook(_, inp, out):
            update_layer.update(inp, out)
        return modify_hook

    handle = layer.register_forward_hook(add_batch())

    return handle