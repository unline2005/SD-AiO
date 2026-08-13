import os
import gc
import lpips
import clip
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import sys

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from osediff import OSEDiff_reg, OSEDiff_gen
from dataloaders.realsr_dataset_three import PairedSROnlineTxtDataset  #realsr_dataset_three

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from torchvision.utils import save_image
import torch.nn as nn
# import open_clip
from guided_diffusion.script_util import i_DDPM

def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")

def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")

def parse_str_list(arg):
    return arg.split(',')

def custom_collate_fn(batch):
    """
    1. `batch` 是一个列表，每个元素是 `__getitem__` 返回的多个样本。
    2. 需要合并所有样本，确保 batch 里任务比例一致。
    """
    all_samples = [sample for sublist in batch for sample in sublist]  # 展平列表
    batch_dict = {key: torch.stack([s[key] for s in all_samples]) for key in all_samples[0] if isinstance(all_samples[0][key], torch.Tensor)}
    batch_dict["task_name"] = [s["task_name"] for s in all_samples]
    batch_dict["neg_prompt"] = [s["neg_prompt"] for s in all_samples]
    return batch_dict

def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--output_dir", default='experience/osediff')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=128,)
    parser.add_argument("--train_batch_size", type=int, default=6, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=10000,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff", help="The name of the wandb project to log to.")
    parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['./image_paths.txt'], help='A comma-separated list of integers')
    parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1], help='A comma-separated list of integers')
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default="/tn/xmu/model/sd21/", type=str) #/tn/xmu/model/sd_turbo//tn/xmu/model/sd21/

    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument('--save_images_steps', type=int, default=100)  # Add this line to specify the interval for saving images

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def denormalize(tensor, mean, std):
    """将图像从标准化状态恢复到 [0, 1] 范围"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

####调用neg_prompt####
def load_neg_prompt_embeds(load_path="neg_prompt_embeds.pt", device="cuda"):
    if os.path.exists(load_path):
        return torch.load(load_path, map_location=device)
    else:
        raise FileNotFoundError(f"Negative prompt embeddings file not found at {load_path}")

        
def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    print('============================')
    model_gen = OSEDiff_gen(args)
    model_gen.set_train() 
    print('============================')

    print('************************')
    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()
    print('************************')
    
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    neg_prompt_embeds = load_neg_prompt_embeds(load_path="neg_prompt_embeds.pt")
    neg_prompt_embeds = neg_prompt_embeds.expand(7, -1, -1)   
    prompt_embeds = load_neg_prompt_embeds(load_path="prompt_embeds.pt")
    prompt_embeds = prompt_embeds.expand(7, -1, -1) 
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            model_reg.unet_update.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    
    ###add###
    layers_to_opt = layers_to_opt + list(model_gen.unet_block_embeddings.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.unet_de_mlp.parameters()) + list(model_gen.unet_block_mlp.parameters()) + list(model_gen.unet_fuse_mlp.parameters())


    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad ##add
            layers_to_opt.append(_p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())

    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad ##add
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    layers_to_opt_reg = []
    for n, _p in model_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset = PairedSROnlineTxtDataset(
        split='train',
        batch_task_ratios={'dehazing': 3, 'deraining': 2, 'denoising': 2}
    )
    

    dl_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

    ddpm_model, _ = i_DDPM()  # 初始化在 CPU
    torch.cuda.empty_cache()

    pretrained_state_dict = torch.load(
        '/tn/work5/UNSB/pretrained/256x256_diffusion_uncond.pt',
        map_location='cpu'
    )
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if 'output_blocks' not in k and ('out.0' not in k) and ('out.2' not in k)
    }
    ddpm_model.load_state_dict(filtered_state_dict, strict=True)

    ddpm_model = ddpm_model.cuda().half()  # 转为 fp16 后再传入 CUDA
    freeze_model_parameters(ddpm_model)

    context_processor = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)
    ).to(device=device, dtype=weight_dtype)


    # Prepare everything with our `accelerator`.
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )
    net_lpips, ddpm_model = accelerator.prepare(net_lpips, ddpm_model)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, model_reg]

            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"] 
                x_tgt = batch["output_pixel_values"]     
                B, C, H, W = x_src.shape
                with torch.no_grad():
                    timesteps = torch.tensor([0], device=x_src.device)  # 确保与输入在同一设备
                    degra_context = ddpm_model(x_src, timesteps)
                    degra_context = degra_context.float()    #[8, 1024, 8, 8]
                    degra_context = context_processor(degra_context)

                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds = model_gen(x_src, degra_context, prompt_embeds, args=args)

                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                # KL loss
                if torch.cuda.device_count() > 1:
                    loss_kl = model_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args) * args.lambda_vsd
                else:
                    loss_kl = model_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args) * args.lambda_vsd
                loss = loss + loss_kl
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # Diff loss: let lora model close to generator
                if torch.cuda.device_count() > 1:
                    loss_d = model_reg.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds, args=args) * args.lambda_vsd_lora
                else:
                    loss_d = model_reg.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds, args=args) * args.lambda_vsd_lora
                accelerator.backward(loss_d)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_kl"] = loss_kl.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)
                    

                    # 反归一化
                    x_src = denormalize(x_src, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    x_tgt = denormalize(x_tgt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    x_tgt_pred = denormalize(x_tgt_pred, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    # 确保图像的像素值在 [0, 1] 范围内
                    x_src = torch.clamp(x_src, 0, 1)
                    x_tgt = torch.clamp(x_tgt, 0, 1)
                    x_tgt_pred = torch.clamp(x_tgt_pred, 0, 1)

                    # Save images of LR, HR, and predicted outputs
                    if global_step % args.save_images_steps == 0:  # Save images at specific intervals
                        # Create directories if they don't exist
                        os.makedirs(os.path.join(args.output_dir, "images", "lr"), exist_ok=True)
                        os.makedirs(os.path.join(args.output_dir, "images", "hr"), exist_ok=True)
                        os.makedirs(os.path.join(args.output_dir, "images", "pred"), exist_ok=True)
                        # Save LR, HR, and predicted images
                        save_image(x_src, os.path.join(args.output_dir, "images", "lr", f"lr_{global_step}.png"))
                        save_image(x_tgt, os.path.join(args.output_dir, "images", "hr", f"hr_{global_step}.png"))
                        save_image(x_tgt_pred, os.path.join(args.output_dir, "images", "pred", f"pred_{global_step}.png"))


                    # Checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args()
    main(args)
