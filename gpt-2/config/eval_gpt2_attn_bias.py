# evaluate gpt2 model with a sink token
# n_layer=12, n_head=12, n_embd=768
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
ckpt_iter = 50000
out_dir="../pretrained-models/attn_bias"
data_dir = '/data/locus/project_data/project_data2/mingjies/nanoGPT/data'
save_dir="results/gpt-2/attn_bias/"
compile = False
model_type = "gpt2_attn_bias"