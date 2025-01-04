num_frames = 16
frame_interval = 3
image_size = (512, 512)

# Define dataset
root = "/workspace/dataset/video"
data_path = "/workspace/dataset/data/train/OpenVid-1M.csv"

use_image_transform = False
num_workers = 8

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained="/workspace/pixart-alpha-checkpoints/PixArt-XL-2-512x512.pth",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "/workspace/outputs/STDiT-512"
wandb = False
torchcodec = False

epochs = 1000
log_every = 10
ckpt_every = 2500
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0
