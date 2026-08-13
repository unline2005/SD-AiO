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
import torch.nn as nn

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from pytorch_msssim import ssim


from osediff import OSEDiff_stage2
from dataloaders.realsr_dataset_three import PairedSROnlineTxtDataset 

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from torchvision.utils import save_image
import open_clip
from guided_diffusion.script_util import i_DDPM

####调用neg_prompt####
def load_neg_prompt_embeds(load_path="prompt/neg_prompt_embeds77.pt", device="cuda"):
    if os.path.exists(load_path):
        return torch.load(load_path, map_location=device)
    else:
        raise FileNotFoundError(f"Negative prompt embeddings file not found at {load_path}")


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
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
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
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

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
    parser.add_argument("--pretrained_model_name_or_path", default="/tn/xmu/model/sd21/", type=str)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_rank_decoder", default=16, type=int)
    parser.add_argument('--save_images_steps', type=int, default=100)  # Add this line to specify the interval for saving images
    parser.add_argument("--osediff_path", type=str, default='/tn/xmu/code/DOD/experience/onlyqkv_lora/checkpoints/model_1.pkl')
        # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference

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

def gradient_loss(pred, target):
    def gradient(x):
        D_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return D_dx, D_dy

    dx_pred, dy_pred = gradient(pred)
    dx_target, dy_target = gradient(target)

    loss = F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)
    return loss

        
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

    model_gen = OSEDiff_stage2(args)
    model_gen.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # layers_to_opt = [p for n, p in model_gen.named_parameters() if "sft_blocks" in n]
    # make the optimizer
    layers_to_opt = []
    for n, _p in model_gen.vae_decoder.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.skip_conv_1.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.skip_conv_2.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.skip_conv_3.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.skip_conv_4.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.sft_blocks_1.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.sft_blocks_2.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.sft_blocks_3.parameters())
    layers_to_opt = layers_to_opt + list(model_gen.vae_decoder.decoder.sft_blocks_4.parameters())


    optimizer = torch.optim.AdamW(
        layers_to_opt, 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler, 
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, 
        power=args.lr_power,
    )

    dataset = PairedSROnlineTxtDataset(
        split='train',
        batch_task_ratios={'dehazing': 1, 'deraining': 1, 'denoising': 1}
    )
    
    # dl_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    dl_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
    trainable_params = [name for name, param in model_gen.named_parameters() if param.requires_grad]

    prompt_embeds = load_neg_prompt_embeds(load_path="./prompt_embeds.pt")
    prompt_embeds = prompt_embeds.expand(3, -1, -1) 
    # 打印可训练参数
    print("可训练参数列表：")
    for param_name in trainable_params:
        print(param_name)

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
    def convert_norm_module_to_float(module):
        if isinstance(module, (torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
            module.float()
        for child in module.children():
            convert_norm_module_to_float(child)

    convert_norm_module_to_float(ddpm_model)
    freeze_model_parameters(ddpm_model)

    context_processor = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)
    ).to(device=device, dtype=weight_dtype)
         
    model_gen, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        model_gen, optimizer, dl_train, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        tracker_config = {k: (str(v) if not isinstance(v, (int, float, str, bool, torch.Tensor)) else v) for k, v in vars(args).items()}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process)
    trainable_params = [name for name, param in model_gen.named_parameters() if param.requires_grad]

    print("=== Parameters in Optimizer ===")
    for i, group in enumerate(optimizer.param_groups):
        for p in group['params']:
            for name, param in model_gen.named_parameters():
                if p is param:
                    print(name)

    # 打印可训练参数
    print("可训练参数列表：")
    for param_name in trainable_params:
        print(param_name)

    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(model_gen):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                with torch.no_grad():
                    timesteps = torch.tensor([0], device=x_src.device)  # 确保与输入在同一设备
                    degra_context = ddpm_model(x_src, timesteps)
                    degra_context = degra_context.float()    #[8, 1024, 8, 8]
                    degra_context = context_processor(degra_context)
                x_tgt_pred, x_tgt_pred_eh = model_gen(x_src, degra_context, prompt_embeds)

                loss0 = F.mse_loss(x_tgt_pred_eh.float(), x_tgt.float(), reduction="mean")
                # loss1 = gradient_loss(x_tgt_pred_eh.float(), x_tgt.float())*0.2
                loss2 = (1 - ssim(x_tgt_pred_eh.float(), x_tgt.float(), data_range=2.0, size_average=True))*0.1
                loss = loss0 + loss2
                # loss = loss0 + loss1 + loss2
                # print(loss)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["loss0"] = loss0.detach().item()
                    # logs["loss1"] = loss1.detach().item()
                    logs["loss2"] = loss2.detach().item()
                    progress_bar.set_postfix(**logs)

                    # 反归一化
                    x_src = denormalize(x_src, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    x_tgt = denormalize(x_tgt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    x_tgt_pred = denormalize(x_tgt_pred, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    x_tgt_pred_eh = denormalize(x_tgt_pred_eh, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    # 确保图像的像素值在 [0, 1] 范围内
                    x_src = torch.clamp(x_src, 0, 1)
                    x_tgt = torch.clamp(x_tgt, 0, 1)
                    x_tgt_pred = torch.clamp(x_tgt_pred, 0, 1)
                    x_tgt_pred_eh = torch.clamp(x_tgt_pred_eh, 0, 1)

                    if global_step % args.save_images_steps == 0:
                        os.makedirs(os.path.join(args.output_dir, "images", "lr"), exist_ok=True)
                        os.makedirs(os.path.join(args.output_dir, "images", "hr"), exist_ok=True)
                        os.makedirs(os.path.join(args.output_dir, "images", "pred"), exist_ok=True)
                        os.makedirs(os.path.join(args.output_dir, "images", "pred_eh"), exist_ok=True)

                        save_image(x_src, os.path.join(args.output_dir, "images", "lr", f"lr_{global_step}.png"))
                        save_image(x_tgt, os.path.join(args.output_dir, "images", "hr", f"hr_{global_step}.png"))
                        save_image(x_tgt_pred, os.path.join(args.output_dir, "images", "pred", f"pred_{global_step}.png"))
                        save_image(x_tgt_pred_eh, os.path.join(args.output_dir, "images", "pred_eh", f"pred_eh_{global_step}.png"))

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_NAFNet_{global_step}.pkl")
                        model = accelerator.unwrap_model(model_gen)
                        model.save_model(outf)

                        # 读取保存的 .pkl 文件并打印参数名
                        checkpoint = torch.load(outf, map_location="cpu")
                        print("Saved parameter keys in 'state_dict_vae':")
                        for k in checkpoint["state_dict_vae"].keys():
                            print(k)


            
                    accelerator.log(logs, step=global_step)




if __name__ == "__main__":
    args = parse_args()
    main(args)

