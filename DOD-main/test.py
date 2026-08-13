import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


from guided_diffusion.script_util import i_DDPM
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

####调用neg_prompt####
def load_neg_prompt_embeds(load_path="prompt/neg_prompt_embeds77.pt", device="cuda"):
    if os.path.exists(load_path):
        return torch.load(load_path, map_location=device)
    else:
        raise FileNotFoundError(f"Negative prompt embeddings file not found at {load_path}")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def denormalize(tensor, mean, std):
    """将图像从标准化状态恢复到 [0, 1] 范围"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def get_validation_prompt(image, device='cuda'):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    return lq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='/tn/xmu/data/test/noisy25/input/', help='path to the input image')   # preset/datasets/test_dataset/noise/BSD68-15   /data/hhz/data/img/haze/SOTS/indoor/hazy  /data/wp/datasets/Test/Derain/Rain100L/input /data/wp/work2/daclip-uir-main/datasets/universal/val/noisy50/LQ/ /data/wp/work2/daclip-uir-main/datasets/universal/val/motion-blurry/LQ /data/wp/work2/daclip-uir-main/datasets/universal/val/low-light/LQ
    parser.add_argument('--output_dir', '-o', type=str, default='/tn/xmu/code/DOD/result/osediff/denoise25', help='the directory to save the output')
    parser.add_argument("--nafnet_path", type=str, default='/tn/xmu/code/DOD/experience/osediff_stage2/checkpoints/model_NAFNet_1.pkl')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="/tn/xmu/model/sd21/", help='sd model path')#sd2.1
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=256)    
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument("--osediff_path", type=str, default='/tn/xmu/code/DOD/experience/osediff/checkpoints/model_1.pkl') #/data/tn/work3_cl/osediff/experience/s3diff_ddpm_sft/model_200001.pkl
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--lora_rank", default=8, type=int)

    args = parser.parse_args()

    # initialize the model
    model = OSEDiff_test(args)

    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.[pj]*[np]*[g]*'))
    else:
        image_names = [args.input_image]
    

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
    ).to("cuda")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16


    
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')
    for image_name in image_names:
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # 分离文件名和扩展名，替换为 .png
        name_without_ext, _ = os.path.splitext(bname)
        output_bname = f"{name_without_ext}.png"  # 确保输出文件为 PNG 格式

        print(output_bname)

        # 获取描述
        lq = get_validation_prompt(input_image)

        prompt_embeds = load_neg_prompt_embeds(load_path="./prompt_embeds.pt")
        prompt_embeds = prompt_embeds.expand(1, -1, -1) 

        # 模型推理
        with torch.no_grad():
            lqq = F.normalize(lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # lqq = lq*2-1    #归一化 0,1  -> -1,1
#             lqq = F.normalize(lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            timesteps = torch.tensor([0], device=lqq.device)  # 确保与输入在同一设备
            degra_context = ddpm_model(lqq, timesteps)
            degra_context = degra_context.float()    #[8, 1024, 8, 8]
            degra_context = context_processor(degra_context)
            output_image = model(lqq, degra_context, prompt_embeds) 

                # 保存图片
            output_image = denormalize(output_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            output_image = torch.clamp(output_image, 0, 1)
            output_pil = transforms.ToPILImage()(output_image[0].cpu())

            if resize_flag:
                output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))

        # 保存为 PNG 格式
        output_pil.save(os.path.join(args.output_dir, output_bname), format='PNG')
