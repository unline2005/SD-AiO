# SD-AiO 代码逐文件审查指南

> 本文件用于人工逐行复核。每个文件都列出了：职责、函数/类清单、数据流、
> 关键正确性依据、以及审查时最容易出错的检查点。
> 行号对应当前提交（截至 `2fa7396`），后续改动后请以函数名定位为主。

---

## 0. 总体架构

```
train.py ──加载配置──▶ sd_aio/config.py
   │
   ├─ 按 cfg.stage 动态导入 stage 模块（utils.load_stage）
   ├─ sd_aio/data.build_loaders() 建训练/验证 loader
   ├─ stage.build_model() → make_optimizer() → accelerator.prepare()
   ├─ 唯一训练循环：stage.compute_loss() → backward → step
   ├─ 周期性/最终：sd_aio/eval.run_eval()
   └─ checkpoint.save_checkpoint() / restore_training() / save_final()

eval.py ──加载配置 + checkpoint──▶ 重建模型 → load_model_weights()
   ├─ 无 --input：build_loaders() → run_eval() → metrics JSON
   └─ 有 --input：run_inference() → pad → 同一 forward → crop → PNG
```

三个 stage 模块（`classifier.py` / `vae_encoder.py` / `spade.py`）都导出同一
协议：

```python
build_model(cfg, device) -> nn.Module
make_optimizer(model, cfg) -> Optimizer
set_train_mode(model)
set_eval_mode(model)
compute_loss(model, raw_model, batch, cfg) -> (loss_tensor, logs_dict)
eval_step(model, raw_model, batch) -> result_dict
```

其中：
- `model` = `accelerator.prepare()` 之后的模型（可能是 DDP + autocast 包装），
  **必须**用它做前向和反向。
- `raw_model` = `accelerator.unwrap_model(model)`，只用于取属性、取缓存、
  保存权重；**不要**用它做训练前向，否则 DDP 不会同步梯度。

---

## 1. `train.py`（唯一训练入口，约 286 行）

### 函数清单

| 函数 | 行 | 职责 |
|---|---|---|
| `parse_args` | 36 | 只接受 `--config` 和任意 `KEY=VALUE` 覆盖 |
| `setup_logger` | 43 | 主进程写 `train.log` + stdout，非主进程 NullHandler |
| `_resolve_resume_path` | 62 | `null` / 具体路径 / `latest` 三种 resume 语义 |
| `main` | 75 | 唯一训练循环 |

### 执行顺序（逐段审查）

1. `configlib.load_config()`：defaults + 实验配置 + CLI 覆盖 + tasks 文件合并。
2. `utils.load_stage()`：按 `cfg.stage` 导入 `sd_aio.{stage}`。
3. 创建 `Accelerator`：`gradient_accumulation_steps` 与 `mixed_precision` 来自配置。
4. **先建数据再建模型**：数据路径错误时不会白加载 SD 权重。
5. `build_model(cfg, accelerator.device)`。
6. `make_optimizer()` + `get_scheduler()`。
7. `accelerator.prepare(model, optimizer, scheduler, train_loader)`。
8. `resume_from` 非空时：`checkpoint.restore_training(raw_model, optimizer, scheduler, path)`。
9. 可选 EMA：所有 rank 各自初始化；resume 时从 `ema.safetensors` 恢复。
10. 主进程才构建 eval LPIPS（避免多卡重复加载）。
11. `while global_step < max_steps` 训练循环：
    - `utils.move_batch()` 把 batch 搬到设备并 cast 浮点 dtype；
    - `accelerator.accumulate(model)` 内做 `compute_loss` / `backward` / clip / step；
    - 只有 `accelerator.sync_gradients` 时 `global_step += 1`。
12. 周期性 eval 只在主进程执行；之后 **所有 rank** `wait_for_everyone()`，
    再统一 `stage.set_train_mode(raw_model)`。
13. checkpoint 只在主进程保存。
14. `finally`：
    - 无论成功失败都 `wait_for_everyone()`；
    - 主进程保存一份 checkpoint（即使未整除 `checkpointing_steps`）；
    - **只有训练正常完成**才跑最终 eval 和写 `final/`。

### 检查点

- `global_step` 只在 `sync_gradients` 后增加，保证 gradient accumulation 语义正确。
- `loss_value = float(logs.get("loss", loss.detach()))`：所有 stage 都返回 `loss`。
- `eval_lpips` 只在主进程构建，非主进程为 `None`；run_eval 只在主进程调用。
- `completed` 标志避免异常后把损坏模型写成 `final/`。
- resume 时 `restored_dir` 只在上一个 `if resume_path is not None` 分支内使用，
  Python 局部变量作用域正确（同一条件进入）。

---

## 2. `eval.py`（唯一评测/推理入口，约 121 行）

### 函数清单

| 函数 | 行 | 职责 |
|---|---|---|
| `parse_args` | 30 | `--config/--checkpoint/--use_ema/--input/--gt/--save_dir/--prompt/--num_samples/--tile_size` |
| `main` | 49 | 重建模型 → 严格加载 → benchmark 或 inference |

### 执行顺序

1. 先解析 `checkpoint_path`：文件、`checkpoint-*` 目录、`output_dir` 均可。
2. **先 resolve 权重路径，再 build_model**：路径错误不会先白加载 SD。
3. `checkpoint.load_model_weights()` 严格加载。
4. `accelerator.prepare(model)` 使 eval 也享受 autocast；多卡时是 DDP。
5. **只有主进程**执行 benchmark 或 inference，其余 rank 在
   `accelerator.wait_for_everyone()` 等待——多卡不会重复写文件。
6. benchmark 走 `run_eval()`；inference 走 `run_inference()`。

### 检查点

- `--use_ema` 是显式请求：ema 不存在时**必须报错**，不静默回退普通权重。
- `--checkpoint` 未给时默认使用 `cfg.output_dir`（找 `checkpoints/` 或 `final/`）。
- 推理模式要求 `cfg.stage == "spade"`，其余 stage 直接报错。

---

## 3. `sd_aio/utils.py`（共享工具，约 74 行）

| 函数 | 行 | 职责 |
|---|---|---|
| `set_train_mode` | 15 | 递归：可训练叶子 `train()`，冻结叶子 `eval()` |
| `set_eval_mode` | 37 | 整树 `eval()` |
| `move_batch` | 42 | tensor 搬设备；**只 cast 浮点 tensor**，label 不会被错误 cast 成 bf16 |
| `count_parameters` | 57 | 参数统计 |
| `weight_dtype_for` | 61 | `bf16/fp16` → torch dtype，其余 fp32 |
| `load_stage` | 65 | `importlib` 导入 stage；只拦截目标模块不存在，不吞 stage 内部依赖错误 |

### 最重要的正确性点

`set_train_mode` 是本项目修复旧 bug 的关键：

- 它不会简单调用 `model.train()`；
- 冻结的 VAE/DINO/GroupNorm 永远留在 `eval()`，使用 running statistics；
- 含有 LoRA 的模块会继续递归，最终只让 `lora_*` 叶子进入 train。

审查时可对照 `tests/test_stages_smoke.py` 的 DDP / LoRA 测试。

---

## 4. `sd_aio/config.py`（配置系统，约 165 行）

| 函数 | 行 | 职责 |
|---|---|---|
| `required` | 55 | 取 `a.b.c`，key 不存在即抛错；值本身为 `None` 不抛 |
| `apply_overrides` | 67 | `KEY=VALUE` 点路径覆盖，OmegaConf 自动推断类型 |
| `_resolve_single_path` | 77 | 只解析 `./ ../ ~` 或绝对路径；保留 HF repo id |
| `resolve_paths` | 89 | 递归解析白名单中的路径 key |
| `warn_unknown_keys` | 111 | 顶层拼写错误警告 |
| `_merge_tasks_file` | 121 | 把 `data.tasks_file` 的 train/test 合并进配置 |
| `load_config` | 141 | defaults + experiment + overrides + tasks 合并 |
| `snapshot` | 159 | 把**完整解析后**配置写入 `output_dir/config.yaml` |

### 检查点

- 路径解析白名单：`sd_path/output_dir/save_dir/resume_from/...`。
  注意 `dino_path` 可能是本地目录或 HF id，只有 `./ ../ ~` 开头才会被改。
- `tasks_file` 相对项目根解析，与启动目录无关。
- 输出快照是绝对路径，eval 用同一份配置重建模型，训练/评测不会 drift。
- 顶层未知 key 只警告，不阻断；但 stage 需要的 key 用 `required()` / 直接
  `.key` 访问，缺失会报错。

---

## 5. `sd_aio/data.py`（全部数据逻辑，约 482 行）

### 函数/类清单

| 项 | 行 | 职责 |
|---|---|---|
| `list_images` | 32 | 目录扫描 + 空目录直接报错 |
| `pair_images` | 47 | LQ/GT 按 stem / prefix / 同目录去噪配对 |
| `parse_sigma_from_name` | 88 | 从任务名解析 `_15` / `_25.5` |
| `add_gaussian_noise` | 94 | `sigma/127.5` 换算到 [-1,1]；eval 用 crc32 种子 |
| `build_deg_types` | 106 | 保持首次出现顺序 |
| `PairedTransform` | 110 | 成对增强；eval 返回全图 |
| `PairedImageDataset` | 159 | 配对 + repeat + 在线噪声 |
| `ClassificationDataset` | 212 | Stage1 单图数据集；去噪类在线合成 |
| `RoundRobinSampler` | 302 | 每组退化各取 1 张构成 batch |
| `_collate_paired` | 328 | 堆叠 `lq/gt/task_name/deg_type` |
| `_collate_classification` | 337 | 堆叠 `lq/label` |
| `build_loaders` | 351 | train/test loader 构建 |

### 关键数据契约

- `batch` 键：
  - 恢复 stage：`lq`, `gt`, `task_name: list[str]`, `deg_type: list[str]`
  - 分类 stage：`lq`, `label`
- `lq/gt` 数值范围 **[-1, 1]**。
- 去噪任务磁盘上必须是 clean-clean，任务名必须带 `_<sigma>` 或显式
  `noise_sigma`；否则构建期直接报错。
- eval 噪声 `crc32(task_name:real_index)` 确定；train 噪声随机。
- test loader 固定 `num_workers=0`，保证 eval 可复现。
- round-robin 会把任务按 `deg_type` **重排为连续组**，再构造 sampler，
  因此任意 YAML 任务顺序都不会破坏组边界。

### 检查点

- `pair_images` 中 LQ/GT 同目录但非 denoise 会直接报错，防止“恒等训练”被静默接受。
- 坏图不重试、不跳过；DataLoader 会直接把错误抛给训练者。
- `ClassificationDataset` 的去噪样本从 clean GT 目录读取并在线上噪。
- `drop_last=True` 会丢弃不满 batch 的尾样本，这是当前训练语义的一部分。

---

## 6. `sd_aio/metrics.py`（指标层，约 92 行）

| 函数/类 | 行 | 职责 |
|---|---|---|
| `to_numpy_rgb` | 16 | [-1,1] tensor → [0,1] HWC numpy |
| `compute_psnr` | 24 | RGB PSNR，data_range=1 |
| `compute_ssim` | 28 | RGB SSIM，channel_axis=-1 |
| `compute_lpips` | 39 | 单样本 LPIPS 标量 |
| `load_lpips` | 45 | LPIPS 构建；永远冻结、不进 checkpoint |
| `MetricAccumulator` | 62 | per-task 均值 + task-equal overall |

### 检查点

- overall 是 **先 per-task 平均，再对所有 task 平均**，不是按样本数加权。
- LPIPS 只在需要时构建；它被注册为 `SpadeRestorer.lpips` 子模块但不保存
  （`save_model_weights` 只存 `requires_grad=True` 的参数）。

---

## 7. `sd_aio/checkpoint.py`（权重与恢复，约 243 行）

> 这是审查权重报错逻辑的核心文件。

### 函数/类清单

| 项 | 行 | 职责 |
|---|---|---|
| `save_model_weights` | 20 | 只保存可训练参数的 safetensors |
| `load_model_weights` | 34 | 严格加载：空文件 / unexpected / 缺失可训练参数都报错 |
| `iter_checkpoints` | 66 | 按 step 倒序列出 checkpoint 目录 |
| `resolve_weights_path` | 79 | eval 用的路径解析，含 `--use_ema` 严格语义 |
| `save_checkpoint` | 117 | 写 weights + ema? + optimizer.pt，并裁剪旧目录 |
| `restore_training` | 163 | 恢复权重 + optimizer + scheduler，坏 checkpoint 回退 |
| `save_final` | 200 | 写 `final/weights.safetensors` |
| `ModelEMA` | 211 | 只跟踪可训练参数 |

### 权重错误矩阵（逐项审查）

| 情况 | 行为 |
|---|---|
| checkpoint 文件不存在 | `FileNotFoundError` |
| safetensors 为空 | `RuntimeError: empty` |
| checkpoint 有 unexpected key | `RuntimeError`，并列出前 5 个 |
| checkpoint 缺少当前模型的可训练参数 | `RuntimeError`，并列出前 5 个 |
| checkpoint 缺少冻结参数（如 SD/DINO 底座） | **允许**，这是设计：只存可训练参数 |
| shape 不一致 | `load_state_dict` 直接报错 |
| resume 时没有 `optimizer.pt` | 该候选视为坏 checkpoint，警告后回退 |
| resume 传 weights-only 文件 | `ValueError`：无法恢复 optimizer |
| 最新 checkpoint 损坏 | 警告后尝试第二新；全坏则 `RuntimeError` |
| `--use_ema` 但 checkpoint 无 ema | `FileNotFoundError`，不静默降级 |
| `--use_ema` + 显式 `weights.safetensors` 文件 | `ValueError`，语义冲突 |
| EMA 加载 missing/unexpected/shape 不一致 | 全部报错 |

### 目录布局

```text
output_dir/
├── config.yaml
├── checkpoints/
│   └── checkpoint-00001000/
│       ├── weights.safetensors
│       ├── ema.safetensors        # 可选
│       └── optimizer.pt           # optimizer + scheduler + step
└── final/
    ├── weights.safetensors
    └── ema.safetensors            # 可选
```

### 检查点

- `save_model_weights` 没有可训练参数会报错，避免保存空权重。
- `save_checkpoint(ema=None)` 会删除同目录旧 `ema.safetensors`，防止 stale EMA。
- `keep_last < 1` 直接报错。
- `restore_training` 要求 `weights.safetensors` 和 `optimizer.pt` **同时存在**；
  `torch.load(..., weights_only=True)` 安全性也在这里。
- EMA 只在主进程保存（train.py 调用路径），所有 rank 更新 shadow。

---

## 8. `sd_aio/classifier.py`（Stage 1，约 262 行）

| 项 | 行 | 职责 |
|---|---|---|
| `ClassifierHead` | 30 | LN + MLP → `[B,C,2]` |
| `DegradationClassifier` | 52 | DINOv2 + head；`forward_features` 同时返回 cls 和 logits |
| `DegFeatureExtractor` | 82 | `F_Deg = cls + alpha * (p @ deg_embedding)` |
| `focal_loss` | 125 | 多标签 focal BCE |
| `compute_binary_metrics` | 131 | acc/exact/precision/recall/f1 + per-class |
| `build_model` | 169 | 训练用分类器；要求 `model.dino_path` |
| `make_optimizer` | 182 | backbone / head 双学习率组 |
| `compute_loss` | 209 | focal loss + batch accuracy |
| `eval_step` | 223 | 返回 predictions + labels |
| `build_deg_extractor` | 235 | Stage2/3 复用的 F_Deg 提取器 |

### 关键公式与语义

- 分类头输出 `[B, C, 2]`，只取 `logits[..., 0]` 作为正类 logit，
  `sigmoid > 0.5` 为预测 1。
- F_Deg 中 classifier 前向包在 `no_grad` 内，但
  `deg_embedding` / `deg_alpha` 的投影在 no_grad **外**，所以 Stage3 打开
  `train_deg_embedding` 时它们可训练。
- `DegFeatureExtractor` 显式处理 classifier 的 device/dtype，并把输出拉回
  调用者的 device/dtype，避免 Stage2 在 autocast 外调用时 dtype 不匹配。

### 检查点

- `build_model` 和 `build_deg_extractor` 都强制 `model.dino_path` 非空。
- `build_deg_extractor` 强制 `degradation_classifier_path` 非空，杜绝“随机
  DINO 静默参与训练”。
- 旧 `.pth` 分类器 checkpoint 兼容路径同样检查 unexpected + 可训练缺失 key。
- Stage1 最终保存的是 `final/weights.safetensors`，只含 trainable 参数。

---

## 9. `sd_aio/vae_encoder.py`（Stage 2，约 190 行）

| 项 | 行 | 职责 |
|---|---|---|
| `AdaIn` | 24 | F_Deg → γ/β，**零初始化**保证初始恒等 |
| `PreRestoreEncoder` | 43 | 深拷贝 VAE encoder，在 down/mid 后插 AdaIN |
| `_load_pretrained_encoder` | 88 | 兼容 safetensors 与旧 pth |
| `build_model` | 105 | 冻结 VAE + 冻结 F_Deg + 训练 AdaIN |
| `make_optimizer` | 131 | 只优化 `requires_grad=True` 参数 |
| `_latent_mean` | 153 | 冻结 VAE 计算 HQ latent mean |
| `compute_loss` | 160 | `L1(z_lq_mean, z_hq_mean) * lambda_l1` |
| `eval_step` | 179 | decode latent mean 得到预览图 |

### 关键正确性

- diffusers Encoder 第 i 个 down block 输出通道是 `block_out_channels[i]`，
  代码还校验 `len(block_out_channels) == len(down_blocks)`。
- `quant_conv(raw)[:, :4]` 是 latent **mean**，`[:, 4:]` 是 logvar，Stage2
  只对齐 mean。
- 深拷贝自 `vae.encoder`，且源 VAE 已 `requires_grad_(False)`，因此
  **只有 AdaIN 参数可训练**（与原算法一致）。
- Stage2 decode 时直接用 raw latent，**不乘/除 scaling_factor**。

---

## 10. `sd_aio/spade.py`（Stage 3，约 630 行，最复杂）

### 模块清单

| 项 | 行 | 职责 |
|---|---|---|
| `Spade` | 61 | GroupNorm + 条件 γ/β |
| `SpadeWrapper` | 85 | 替换 UNet ResBlock.conv2，兼容 PEFT-wrapped conv |
| `MultiScaleExtractor` | 106 | LQ 金字塔：64/32/16/8 尺度特征 |
| `SpadeConditionModule` | 181 | spatial-only SPADE，`setup` 校验通道 |
| `DegTextFusion` | 255 | F_Deg → 1 个 token 拼到 CLIP 文本前 |
| `DegAwareConditionModule` | 267 | SPADE + 退化 token |
| `_attach_unet_lora` | 294 | UNet LoRA |
| `_attach_vae_lora` | 306 | VAE encoder/decoder LoRA；rank 相同则单 adapter |
| `_mark_lora_trainable` | 330 | 只让 `lora` 参数可训练 |
| `_sample_timestep` | 335 | fixed / range 两种 timestep |
| `SpadeRestorer` | 343 | 唯一前向：encode → 加噪 → denoise → x0 → decode |
| `_populate_prompt_embeddings` | 453 | 把任务 prompt 预编码为 `ParameterDict` |
| `_build_condition_module` | 467 | `none/simple/deg-aware` 三选一 |
| `build_model` | 488 | 组装 SD2.1 全部组件 |
| `compute_loss` | 595 | `λ_l2 * MSE + λ_lpips * LPIPS` |
| `eval_step` | 619 | 训练/评测共用的前向 |

### 单步恢复数学（务必复核）

```text
z0 = encode(LQ) * scaling_factor          # 或 Stage2 pretrained encoder
z_t = sqrt(alpha_bar_t) * z0 + sqrt(1-alpha_bar_t) * eps
x0_hat = z_t + sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t) * (eps - eps_theta)
pred = decode(x0_hat / scaling_factor)
```

对应 `forward()` 与 `_x0_coeff()`。审查时重点核对：
- 编码后 **乘** `scaling_factor`；
- 解码前 **除** `scaling_factor`；
- 系数是 `sqrt(1-ᾱ)/sqrt(ᾱ)`，不是反向。

### 通道/尺度对应（务必复核）

UNet latent 64×64 时：

| UNet 位置 | 空间尺寸 | SPADE 特征 |
|---|---|---|
| down0 | 64 | C320 |
| down1 | 32 | C640 |
| down2 | 16 | C1280_Down |
| down3 / mid / up0 | 8 | C1280_Mid |
| up1 | 16 | C1280_Down |
| up2 | 32 | C640 |
| up3 | 64 | C320 |

`setup()` 会在构建期检查 UNet conv2 输出通道是否等于 `condition_channels`，
配置错误不会留到前向中段。

### 其他审查点

- **LoRA 顺序**：先给 UNet 加 LoRA，再把可能被 PEFT 包裹的 `conv2` 交给
  `SpadeWrapper`；`SpadeWrapper.forward` 通过 `hasattr(base_layer)` 分支兼容。
- **pretrained encoder**：从未加 LoRA 的 `vae.encoder` 深拷贝；若启用
  encoder LoRA，只加在 pretrained encoder 上，主 VAE encoder 不重复加。
- **文本编码器**：保存在普通 dict `_aux` 中，不注册为子模块，因此
  `model.to(device)` 不会把它搬到 GPU。
- `prompt_embeddings` 是 `nn.ParameterDict`（requires_grad=False），随模型
  移动设备，但不进 checkpoint（重建模型时重新编码）。
- `DegAwareConditionModule` 在 `f_deg is None` 时直接报错，避免退化 token
  静默失效。
- `_attach_vae_lora` 在 encoder/decoder rank 相同时只注册一个 adapter，
  避免 PEFT 多 adapter 警告。

---

## 11. `sd_aio/eval.py`（唯一评测/推理核心，约 382 行）

| 项 | 行 | 职责 |
|---|---|---|
| `EvalReport` | 28 | per-task + overall + 可视化路径 |
| `_crop_to_multiple` | 46 | 中心裁到 16 的倍数 |
| `_limit_batch` | 57 | `num_samples` 精确截断 batch |
| `_pad_to_multiple` | 66 | reflect pad 到 64 的倍数 |
| `_save_strip` | 75 | 保存 LQ|Pred|GT 横拼图 |
| `_run_classifier_eval` | 99 | 分类指标聚合 |
| `_run_image_eval` | 136 | crop→pad→forward→crop back→指标 |
| `run_eval` | 216 | 训练周期与独立评测的唯一入口 |
| `list_input_images` | 267 | 单图/目录递归收集 |
| `run_inference` | 281 | pad→forward→crop→存图，可选 GT 打分 |
| `save_report` | 369 | 指标 JSON |

### 评测协议

```text
原图 → center-crop 到 16 的倍数 → reflect pad 到 64 的倍数
     → stage.eval_step（内部走 SpadeRestorer.forward）
     → pred crop 回原 crop 尺寸
     → 对 crop 区域计算 PSNR/SSIM/LPIPS
```

### 检查点

- `stage.set_eval_mode(raw_model)` 每次评测都会调用，训练路径在 barrier 后
  再恢复 train mode。
- 分类和图像两个分支共享 `run_eval`，但聚合逻辑不同。
- `_limit_batch` 保证 `num_samples_per_task` 不会因 batch_size 超采。
- 推理 tiling 使用 diffusers 原生 `enable_tiling()`，并设置
  `tile_sample_min_size` / `tile_latent_min_size`。
- 推理保存时保留输入目录相对结构，避免同名 stem 互相覆盖。
- 推理自定义 prompt 时使用 CPU 上的 text encoder 现场编码。

---

## 12. `configs/`（配置文件）

| 文件 | 内容 |
|---|---|
| `defaults.yaml` | 全项目共享默认值：seed/mixed_precision/优化器/scheduler/trainer/eval/EMA |
| `tasks_3d.yaml` | 唯一数据集定义：8 个训练任务、5 个测试任务 |
| `stage1_classifier.yaml` | Stage1 训练配置 |
| `stage2_vae.yaml` | Stage2 训练配置 |
| `stage3_spade.yaml` | Stage3 训练配置 |

审查时注意：
- `stage3.model.condition_type=deg-aware`、`backbone_type=simple-conv`、
  `timestep=100`、`lambda_l2=2.0`、`lambda_lpips=5.0`。
- `stage2` 与 `stage3` 的 `degradation_classifier_path` 指向 Stage1 的
  `final/weights.safetensors`，跑通前需先完成 Stage1。
- 服务器数据路径集中在 `tasks_3d.yaml`，换机器只改这里。

---

## 13. `tests/`（40 个测试）

| 文件 | 覆盖 |
|---|---|
| `test_config.py` | 合并、覆盖类型、缺失 key、未知 key、快照回读 |
| `test_data.py` | 配对、sigma/127.5、crc32、fail-fast、分类数据、round-robin |
| `test_metrics.py` | 数值范围、PSNR/SSIM、task-equal overall |
| `test_checkpoint.py` | 保存/加载、空文件、unexpected/missing、resume 必需 optimizer、EMA mismatch、`--use_ema` 报错 |
| `test_stages_smoke.py` | 三阶段前向/反向、SPADE 注入、LoRA+SPADE、DDP prepared/raw 接口 |
| `test_eval_smoke.py` | crop/pad 协议、推理存图、GT 打分 |
| `test_train_end_to_end.py` | 真实 train.py 一步训练、resume、eval.py 重建评测 |

验证命令：

```bash
python -m pytest
ruff check train.py eval.py sd_aio tests
ruff format --check train.py eval.py sd_aio tests
python -m compileall -q train.py eval.py sd_aio tests
git diff --check
```

---

## 14. 最容易出错的 10 个检查点

1. `scaling_factor` 乘除位置（编码乘、解码除）。
2. `quant_conv[:, :4]` 是 mean，不是完整 latent。
3. `sigma / 127.5` 与 [-1,1] 空间换算。
4. frozen GroupNorm 是否被 `set_train_mode` 意外切回 train。
5. DDP 下前向必须走 prepared model，属性/缓存走 raw_model。
6. checkpoint 只存可训练参数，load 必须允许 frozen missing、拒绝 trainable missing。
7. resume 必须同时有 `weights.safetensors` 与 `optimizer.pt`。
8. 多卡 eval/inference 只允许主进程写文件，全员 barrier。
9. SPADE 特征尺度与 UNet down/mid/up 的 64/32/16/8 对应。
10. LoRA 与 SPADE 的包裹顺序：先 LoRA，后 SPADE。

---

## 15. 当前已知边界假设

- diffusers 目标版本 `>=0.38,<0.39`；`SpadeWrapper` 兼容普通/ PEFT conv2。
- DDP 是主要分布式后端；当前 checkpoint 机制不专门适配 FSDP/DeepSpeed。
- eval batch_size 默认 1；`_limit_batch` 已保证 num_samples 精确，但更高效
  的 batch eval 尚未启用。
- Stage2 只训练 AdaIN，不微调 encoder 副本。
- 推理默认复用第一条 prompt embedding；多退化输入建议显式 `--prompt`。
- 分类器任务 label 来自 `train` 中 `deg_type` 的首次出现顺序。
