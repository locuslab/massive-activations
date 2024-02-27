out_dir = 'results/'
data_dir = '/data/locus/project_data/project_data2/mingjies/nanoGPT/data'

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M-default-run'
compile=False 

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

model_type = "gpt2_sink"
num_reg=1