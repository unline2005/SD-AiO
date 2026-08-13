import os
import re
import sys
sys.path.append(os.getcwd())
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from model_qkv import make_1step_sched, my_lora_fwd, my_lora_fwd_sft

from my_utils.vaehook import VAEHook, perfcount
import sys
sys.path.insert(0, './NAFNett')
# from NAFNett.basicsr.models.archs.arch_util import default_init_weights
from NAFNett.basicsr.models.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
from basicsr.archs.rrdbnet_arch import RRDB

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nonlinearity(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in

class Fuse_sft_block_RRDB(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, num_grow_ch=32):
        super().__init__()
        self.encode_enc_1 = ResBlock(2*in_ch, in_ch)
        self.encode_enc_2 = make_layer(RRDB, num_block, num_feat=in_ch, num_grow_ch=num_grow_ch)
        self.encode_enc_3 = ResBlock(in_ch, out_ch)
        # 可学习参数 w，初始化为 1.0（也可以是其他值）
        self.w = nn.Parameter(torch.tensor(1.0))  # 或 torch.ones(1) if you prefer
        print('no weight')

    def forward(self, enc_feat, dec_feat):
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        residual = self.w * enc_feat
        out = dec_feat + residual
        return out

#####lora finetune vae decoder#####
def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    # print(self.current_down_blocks)
    return sample


def my_vae_decoder_fwd(self, sample, current_down_blocks, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        sft_blocks = [self.sft_blocks_1, self.sft_blocks_2, self.sft_blocks_3, self.sft_blocks_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](current_down_blocks[::-1][idx] * self.gamma)
            # print('=============================================================')
            # print(skip_in.size())
            # print(sample.size())
            sample = sft_blocks[idx](skip_in, sample)
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def initialize_vae_decoder(args, return_lora_module_names=False):
    vae_decoder = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae_decoder.requires_grad_(False)
    vae_decoder.encoder.forward = my_vae_encoder_fwd.__get__(vae_decoder.encoder, vae_decoder.encoder.__class__)
    vae_decoder.decoder.forward = my_vae_decoder_fwd.__get__(vae_decoder.decoder, vae_decoder.decoder.__class__)
    vae_decoder.requires_grad_(True)
    vae_decoder.train()
    vae_decoder.decoder.sft_blocks_1 = Fuse_sft_block_RRDB(512, 512).cuda().requires_grad_(True)
    vae_decoder.decoder.sft_blocks_2 = Fuse_sft_block_RRDB(512, 512).cuda().requires_grad_(True)
    vae_decoder.decoder.sft_blocks_3 = Fuse_sft_block_RRDB(512, 512).cuda().requires_grad_(True)
    vae_decoder.decoder.sft_blocks_4 = Fuse_sft_block_RRDB(256, 256).cuda().requires_grad_(True)
    vae_decoder.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae_decoder.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae_decoder.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae_decoder.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae_decoder.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae_decoder.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae_decoder.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae_decoder.decoder.skip_conv_4.weight, 1e-5)
    vae_decoder.decoder.ignore_skip = False
    vae_decoder.decoder.gamma = 1
    return vae_decoder
    
def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]

    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.sched = make_1step_sched(args.pretrained_model_name_or_path)
        self.args = args
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank
        print('only finetune linear layer.')

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
                                            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                                                        "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
                                                                ]

        
        block_embedding_dim = 64
        num_embeddings = 256


        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )


        self.unet_fuse_mlp = nn.Linear(256 + 64, self.lora_rank_unet * 2).to("cuda")

        default_init_weights([self.unet_de_mlp, self.unet_block_mlp, self.unet_fuse_mlp], 1e-5)
        print('init:1e-5')

        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        print("Initializing model with random weights")
        vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian",
                                     target_modules=target_modules_vae)
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet_lora_config = LoraConfig(r=self.lora_rank_unet, init_lora_weights="gaussian",
                                      target_modules=target_modules_unet)
        unet.add_adapter(unet_lora_config)

        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd_sft.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd_sft.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()

        print('add skip connection:no skip')

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet_de_mlp.eval()
        self.unet_block_mlp.eval()
        self.unet_fuse_mlp.eval()
        self.unet_block_embeddings.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet_de_mlp.train()
        self.unet_block_mlp.train()
        self.unet_fuse_mlp.train()
        self.unet_block_embeddings.requires_grad_(True)

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    def forward(self, c_t, degra_context, prompt_embeds, args=None):
        unet_de_c_embed = self.unet_de_mlp(degra_context)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        unet_embeds = self.unet_fuse_mlp(torch.cat([
            unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1),
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0], 1, 1)
        ], -1))

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")
                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    assert block_id < unet_embeds.shape[1], f"block_id {block_id} exceeds unet_embeds dimension."
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.gamma, module.beta = torch.chunk(unet_embed, chunks=2, dim=1)

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32)).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds

    def save_model(self, outf):
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k},
            "state_dict_unet_de_mlp": self.unet_de_mlp.state_dict(),
            "state_dict_unet_block_mlp": self.unet_block_mlp.state_dict(),
            "state_dict_unet_fuse_mlp": self.unet_fuse_mlp.state_dict(),
            "state_dict_unet_block": self.unet_block_embeddings.state_dict()
        }
        torch.save(sd, outf)



class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args)

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss


class OSEDiff_stage2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        

        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        self.sched = make_1step_sched(self.args.pretrained_model_name_or_path)
        self.guidance_scale = 1.07
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank
        self.lora_rank_vae_decoder = self.args.lora_rank_decoder

        vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        vae_decoder = copy.deepcopy(vae).to(self.device)  # Clone for fine-tuning
        unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]

        num_embeddings = 256
        block_embedding_dim = 64


        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = self.unet_de_mlp.to("cuda")

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )
        self.unet_block_mlp = self.unet_block_mlp.to("cuda")
        
        self.unet_fuse_mlp = nn.Linear(256 + 64, self.lora_rank_unet * 2).to("cuda")
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim).to("cuda")

        if self.args.osediff_path is not None:
            sd = torch.load(self.args.osediff_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _unet_de_mlp = self.unet_de_mlp.state_dict()
            for k in sd["state_dict_unet_de_mlp"]:
                _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
            self.unet_de_mlp.load_state_dict(_unet_de_mlp)

            _unet_block_mlp = self.unet_block_mlp.state_dict()
            for k in sd["state_dict_unet_block_mlp"]:
                _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
            self.unet_block_mlp.load_state_dict(_unet_block_mlp)

            _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
            for k in sd["state_dict_unet_fuse_mlp"]:
                _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
            self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

            # embeddings_state_dict = sd["state_embeddings"]
            self.unet_block_embeddings.load_state_dict(sd['state_dict_unet_block'])
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=self.lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd_sft.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd_sft.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()

        self.unet_de_mlp.requires_grad_(False)
        self.unet_block_mlp.requires_grad_(False)

        self.unet_fuse_mlp.requires_grad_(False)

        for p in self.unet_block_embeddings.parameters():
            p.requires_grad = False
        ###############################################################################oral#########################################################       
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae_decoder.decoder.forward = my_vae_decoder_fwd.__get__(vae_decoder.decoder, vae_decoder.decoder.__class__)
        self.unet, self.vae = unet, vae
        self.vae_decoder = initialize_vae_decoder(self.args)
        self.vae_decoder.to("cuda")
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)

        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.decoder.parameters():
            param.requires_grad = True


        self.timesteps = torch.tensor([999], device=self.device).long()


    def forward(self, c_t, degra_context, prompt_embeds, args=None):
        with torch.no_grad():

            unet_de_c_embed = self.unet_de_mlp(degra_context)

            unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)
 
            unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1),unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

            for layer_name, module in self.unet.named_modules():
                if layer_name in self.unet_lora_layers:
                    split_name = layer_name.split(".")
                    if split_name[0] == 'down_blocks':
                        block_id = int(split_name[1])
                        assert block_id < unet_embeds.shape[1], f"block_id {block_id} exceeds unet_embeds dimension."
                        unet_embed = unet_embeds[:, block_id]
                    elif split_name[0] == 'mid_block':
                        unet_embed = unet_embeds[:, 4]
                    elif split_name[0] == 'up_blocks':
                        block_id = int(split_name[1]) + 5
                        unet_embed = unet_embeds[:, block_id]
                    else:
                        unet_embed = unet_embeds[:, -1]
                    # 确保 reshape 后的形状
                    module.gamma, module.beta = torch.chunk(unet_embed, chunks=2, dim=1) 

            lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            skip_feats = self.vae.encoder.current_down_blocks

            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample

            output_image_oral = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        
        x_denoised = self.vae.post_quant_conv(x_denoised / self.vae.config.scaling_factor)
        output_image = self.vae_decoder.decoder(x_denoised, skip_feats).clamp(-1, 1)
        
        return output_image_oral, output_image

    def save_model(self, outf, full=False):
        sd = {}
        # sd["vae_lora_decoder_modules"] = self.lora_vae_modules_decoder 
        sd["rank_vae_decoder"] = self.lora_rank_vae_decoder

        if full:
            sd["state_dict_vae"] = self.vae_decoder.state_dict()
        else:
            sd["state_dict_vae"] = {
                k: v for k, v in self.vae_decoder.state_dict().items()
                if "lora" in k or "sft_blocks" in k or "skip_conv" in k
            }
        torch.save(sd, outf)


    def load_decoder(self, path):
        state_dict = torch.load(path)
        for blk, sd in zip(self.sft_blocks, state_dict["sft_blocks"]):
            blk.load_state_dict(sd, strict=False)

    
    def load_ckpt(self, model):
        # Load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # Load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

class OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(self.args.pretrained_model_name_or_path)
        self.guidance_scale = 1.07
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        vae_decoder = copy.deepcopy(vae).to(self.device)  # Clone for fine-tuning
        unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
                                            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                                                        "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
                                                                ]

        num_embeddings = 256
        block_embedding_dim = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = self.unet_de_mlp.to(self.device, dtype=self.weight_dtype)

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )
        
        self.unet_block_mlp = self.unet_block_mlp.to(self.device, dtype=self.weight_dtype)

        self.unet_fuse_mlp = nn.Linear(256 + 64, self.lora_rank_unet * 2).to(self.device, dtype=self.weight_dtype)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim).to(self.device, dtype=self.weight_dtype)

        if self.args.osediff_path is not None:
            sd = torch.load(self.args.osediff_path, map_location=self.device)
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _unet_de_mlp = self.unet_de_mlp.state_dict()
            for k in sd["state_dict_unet_de_mlp"]:
                _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
            self.unet_de_mlp.load_state_dict(_unet_de_mlp)

            _unet_block_mlp = self.unet_block_mlp.state_dict()
            for k in sd["state_dict_unet_block_mlp"]:
                _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
            self.unet_block_mlp.load_state_dict(_unet_block_mlp)

            _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
            for k in sd["state_dict_unet_fuse_mlp"]:
                _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
            self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

            self.unet_block_embeddings.load_state_dict(sd['state_dict_unet_block'])
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=self.lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd_sft.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd_sft.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}


        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae_decoder.decoder.sft_blocks_1 = Fuse_sft_block_RRDB(512, 512).cuda().requires_grad_(True)
        vae_decoder.decoder.sft_blocks_2 = Fuse_sft_block_RRDB(512, 512).cuda().requires_grad_(True)
        vae_decoder.decoder.sft_blocks_3 = Fuse_sft_block_RRDB(512, 512).cuda().requires_grad_(True)
        vae_decoder.decoder.sft_blocks_4 = Fuse_sft_block_RRDB(256, 256).cuda().requires_grad_(True)

        vae_decoder.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae_decoder.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae_decoder.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae_decoder.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae_decoder.decoder.ignore_skip = False
        vae_decoder.decoder.forward = my_vae_decoder_fwd.__get__(vae_decoder.decoder, vae_decoder.decoder.__class__)
        vae_decoder.decoder.gamma = 1
        self.vae_decoder = vae_decoder.to(self.device, dtype=self.weight_dtype)

        self._load_finetuned_decoder(torch.load(args.nafnet_path, map_location=self.device))


        self.unet, self.vae = unet, vae
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.decoder.parameters():
            param.requires_grad = True
        self.unet_de_mlp.requires_grad_(False)
        self.unet_block_mlp.requires_grad_(False)
        self.unet_fuse_mlp.requires_grad_(False)

        for p in self.unet_block_embeddings.parameters():
            p.requires_grad = False
    
    
    def _load_finetuned_decoder(self, model):
        state_dict = model["state_dict_vae"]
        loaded_params, total_params = 0, 0
        loaded_keys, missing_keys = [], []

        for name, param in self.vae_decoder.named_parameters():
            if "lora" in name or "skip_conv" in name or "sft_blocks" in name:
                num_param = param.numel()
                total_params += num_param
                if name in state_dict:
                    try:
                        param.data.copy_(state_dict[name])
                        loaded_params += num_param
                        loaded_keys.append(name)
                    except Exception as e:
                        print(f"[❌] Failed to load {name}: {e}")
                else:
                    missing_keys.append(name)

        print(f"[✅] 加载参数量（元素个数）: {loaded_params:,} / {total_params:,}")
        print(f"[📦] 加载成功的键数量: {len(loaded_keys)}")
        print(f"[⚠️] 缺失键数量: {len(missing_keys)}")

        if missing_keys:
            print("缺失参数名称如下：")
            for key in missing_keys:
                print("   -", key)

    def forward(self, c_t, degra_context, prompt_embeds, args=None):
        with torch.no_grad():
            c_t = c_t.to(self.device, dtype=self.weight_dtype)
            degra_context = degra_context.to(self.device, dtype=self.weight_dtype)
            prompt_embeds = prompt_embeds.to(self.device, dtype=self.weight_dtype)
            unet_de_c_embed = self.unet_de_mlp(degra_context)
            unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

            unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1),unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))   

            for layer_name, module in self.unet.named_modules():
                if layer_name in self.unet_lora_layers:
                    split_name = layer_name.split(".")


                    if split_name[0] == 'down_blocks':
                        block_id = int(split_name[1])
                        assert block_id < unet_embeds.shape[1], f"block_id {block_id} exceeds unet_embeds dimension."
                        unet_embed = unet_embeds[:, block_id]
                    elif split_name[0] == 'mid_block':
                        unet_embed = unet_embeds[:, 4]
                    elif split_name[0] == 'up_blocks':
                        block_id = int(split_name[1]) + 5
                        unet_embed = unet_embeds[:, block_id]
                    else:
                        unet_embed = unet_embeds[:, -1]

                    # 确保 reshape 后的形状
                    module.gamma, module.beta = torch.chunk(unet_embed, chunks=2, dim=1)

            lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            skip_feats = self.vae.encoder.current_down_blocks

            ## add tile function
            _, _, h, w = lq_latent.size()
            tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
            if h * w <= tile_size * tile_size:
                print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
                model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            else:
                print(f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled")
                # tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to()
                tile_size = min(tile_size, min(h, w))
                tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(c_t.device)

                grid_rows = 0
                cur_x = 0
                while cur_x < lq_latent.size(-1):
                    cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                    grid_rows += 1

                grid_cols = 0
                cur_y = 0
                while cur_y < lq_latent.size(-2):
                    cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                    grid_cols += 1

                input_list = []
                noise_preds = []
                for row in range(grid_rows):
                    noise_preds_row = []
                    for col in range(grid_cols):
                        if col < grid_cols-1 or row < grid_rows-1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size-tile_overlap * row, 0)
                            ofs_y = max(col * tile_size-tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows-1:
                            ofs_x = w - tile_size
                        if col == grid_cols-1:
                            ofs_y = h - tile_size

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size

                        # input tile dimensions
                        input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                        input_list.append(input_tile)

                        if len(input_list) == 1 or col == grid_cols-1:
                            input_list_t = torch.cat(input_list, dim=0)
                            # predict the noise residual
                            model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=prompt_embeds).sample
                        
                            input_list = []
                        noise_preds.append(model_out)

                # Stitch noise predictions for all tiles
                noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
                contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
                # Add each tile contribution to overall latents
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if col < grid_cols-1 or row < grid_rows-1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size-tile_overlap * row, 0)
                            ofs_y = max(col * tile_size-tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows-1:
                            ofs_x = w - tile_size
                        if col == grid_cols-1:
                            ofs_y = h - tile_size

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size

                        noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                        contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                # Average overlapping areas with more than 1 contributor
                noise_pred /= contributors
                model_pred = noise_pred
                
            ####################################
#             encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
            # x_denoised = x_denoised + lq_latent
            # output_image_oral = (self.vae.decode(x_denoised.to(self.device, dtype=self.weight_dtype)  / self.vae.config.scaling_factor).sample).clamp(-1, 1)
            # x_denoised = x_denoised + lq_latent ####add skip connectio
            x_denoised = self.vae.post_quant_conv(x_denoised.to(self.device, dtype=self.weight_dtype) / self.vae.config.scaling_factor)
            output_image = self.vae_decoder.decoder(x_denoised, skip_feats).clamp(-1, 1)
        
        return output_image
    
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))   
