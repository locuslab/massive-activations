CACHE_DIR_BASE = "./model_weights"

MODEL_DICT_LLMs = {
    ### llama2 model
    "llama2_7b": {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_13b": {
        "model_id": "meta-llama/Llama-2-13b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_70b": {
        "model_id": "meta-llama/Llama-2-70b-hf", 
        "cache_dir": CACHE_DIR_BASE
    },

    ### llama2 chat model
    "llama2_7b_chat": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 
    "llama2_13b_chat": {
        "model_id": "meta-llama/Llama-2-13b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 
    "llama2_70b_chat": {
        "model_id": "meta-llama/Llama-2-70b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 

    ### mistral model 
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe": {
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_7b_instruct":{
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe_instruct": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    ### phi-2
    "phi-2": {
        "model_id": "microsoft/phi-2",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### falcon model
    "falcon_7b": {
        "model_id": "tiiuae/falcon-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon_40b": {
        "model_id": "tiiuae/falcon-40b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### mpt model 
    "mpt_7b": {
        "model_id": "mosaicml/mpt-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_30b": {
        "model_id": "mosaicml/mpt-30b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### opt model 
    "opt_125m": {
        "model_id": "facebook/opt-125m", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_350m": {
        "model_id": "facebook/opt-350m", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_1.3b": {
        "model_id": "facebook/opt-1.3b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_2.7b": {
        "model_id": "facebook/opt-2.7b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_7b": {
        "model_id": "facebook/opt-6.7b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_13b": {
        "model_id": "facebook/opt-13b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_30b": {
        "model_id": "facebook/opt-30b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_66b": {
        "model_id": "facebook/opt-66b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### gpt2 model 
    "gpt2": {
        "model_id": "gpt2",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_medium": {
        "model_id": "gpt2-medium",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_large": {
        "model_id": "gpt2-large",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_xl": {
        "model_id": "gpt2-xl",
        "cache_dir": CACHE_DIR_BASE
    },
}