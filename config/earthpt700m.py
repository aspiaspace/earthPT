# launch as the following (e.g. in a tmux session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_earthpt.py

# don't forget to $ mkdir logs

# params
n_layer=36
n_head=20
n_embd=1280
block_size=256

# here we follow chinchilla
learning_rate = 2e-5 # max learning rate
min_lr = 2e-6 # max learning rate

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
init_from = "resume"
batch_size = 16
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 14B
max_iters = 90010
lr_decay_iters = 85500 * 1.1

# eval stuff
eval_interval = 5000
eval_iters = 200 
log_interval = 10
out_dir ='logs/earthpt'
#init_from='resume'

# weight decay
weight_decay = 1e-1
