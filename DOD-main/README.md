# Diffusion Once and Done: Degradation-Aware LoRA for All-in-One Image Restoration

**[AAAI 2026 Accepted]** 🎉  

This project is based on the **Stable Diffusion** model and implements a **two-stage multi-task image restoration framework**, capable of handling denoising, deraining, dehazing, and other image degradations in a unified manner.

The overall framework of our method is illustrated below:

![Framework](https://github.com/tonia86/DOD/blob/main/image/frame.jpg)  
*Figure 1. Overall framework of our method: Stage 1 removes major degradations, and Stage 2 restores fine details and consistency.*

---

## Environment Setup

```bash
pip install -r requirements_dod.txt
```

## Dataset Folder Structure
It is recommended to organize your datasets as follows:
```bash
dataset/
├── BSDWED15/                # Noise level 15 for denoising task
│   ├── LQ/                  # Low-quality images (noisy)
│   └── GT/                  # High-quality images (clean)
├── BSDWED25/                # Noise level 25
│   ├── LQ/
│   └── GT/
├── BSDWED50/                # Noise level 50
│   ├── LQ/
│   └── GT/
├── rain100L/                # Raindrop images (deraining task)
│   ├── LQ/
│   └── GT/
├── OTS/                     # OTS dataset for dehazing task
│   ├── LQ/
│   └── GT/
```

## Training Commands

### Stage 1: Degradation Removal

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
    --enable_xformers_memory_efficient_attention \
    --lora_rank=8 \
    --output_dir=experience/osediff \
    --dataset_prob_paths_list 1,1 \
    --tracker_project_name onlyqkv_lora \
    --allow_tf32
```

> ⚠️ Note: Please set the path to the pre-trained model in train.py:
>
> ```python
> parser.add_argument("--pretrained_model_name_or_path", default="/tn/xmu/model/sd21/", type=str)
> pretrained_state_dict = torch.load('/tn/work5/UNSB/pretrained/256x256_diffusion_uncond.pt')
> ```

### Stage 2: Detail Restoration

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch train_stage2.py \
    --enable_xformers_memory_efficient_attention \
    --output_dir=experience/osediff_stage2 \
    --dataset_prob_paths_list 1,1 \
    --tracker_project_name osediff_stage2 \
    --gradient_accumulation_steps 1
```

## Testing

```bash
python test.py
```

### Some Results

![motion-blur-SR](https://github.com/tonia86/DOD/blob/main/image/compare_visual.png)

## Acknowledgments

This code is mainly built on [[OSEDiff](https://github.com/cswry/OSEDiff.git)].
