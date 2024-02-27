# Monkey Patch LLMs

This directory provide the code where we use to get intermediate hidden states from LLMs from [Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models) and ViTs from [timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py). For other LLM model families, you need to write a custom forward function, depending on the model definition.

A sample command to replace the forward function of [LLaMADecoderLayer](https://github.com/huggingface/transformers/blob/v4.36.0/src/transformers/models/llama/modeling_llama.py#L755) with a custom forward function ``llama_custom_decoderlayer_forward``:
```py
import types 

def enable_llama_custom_decoderlayer(layer, layer_id):
    """
    This function modifies a given LlamaDecoderLayer object by setting its layer_id and replacing its forward method with a custom implementation.
    It allows for customization of the layer's behavior during the forward pass, which is when the layer processes input data.
    """
    
    layer.layer_id = layer_id
    # This line assigns a unique identifier to the layer. The `layer_id` parameter is used to specify this identifier,
    # which can be useful for tracking, debugging, or applying specific operations to certain layers within a larger model.
    
    layer.forward = types.MethodType(
        llama_custom_decoderlayer_forward, layer
    )
    # This line replaces the layer's original `forward` method with a new one.
    # `types.MethodType` is used to bind a new method to an existing object. In this case, it binds the 
    # `llama_custom_decoderlayer_forward` function to the `layer` object as its new `forward` method.
    # `llama_custom_decoderlayer_forward` should be a function defined elsewhere that takes the same arguments as the original
    # `forward` method of the layer and implements the desired custom behavior for processing input data.
    # This allows for dynamic modification of the layer's behavior without altering the original class definition.
```