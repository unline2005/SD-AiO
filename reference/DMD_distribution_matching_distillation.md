# DMD: Distribution Matching Distillation 详细讲解

**论文**: One-step Diffusion with Distribution Matching Distillation (CVPR 2024)

**作者**: Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman (MIT & Adobe)

**论文**: <https://arxiv.org/abs/2311.18828> | **代码**: <https://github.com/tianweiy/DMD2>

**续作**: DMD2 (NeurIPS 2024 Oral) — <https://arxiv.org/abs/2405.14867>

---

## 1. 核心思想：为什么要 DMD？

扩散模型生成一张图需要 50-100 步。之前的蒸馏方法（如 Progressive Distillation、Consistency Models）做的是**轨迹匹配**——强制学生的输出和教师逐点一致。问题是：

- 上限被锁死在教师轨迹
- 容易过拟合
- 需要预计算配对数据

DMD 的核心范式转变：**不匹配点，匹配分布**。只需要学生生成的图像"看起来像"预训练扩散模型生成的图——不需要逐点对应。

在 ImageNet 64×64 上达到 FID 2.62，零样本 COCO-30k 上达到 FID 11.49。使用 FP16 推理可达 **20 FPS**（512×512 图像）。

---

## 2. 数学原理

### 2.1 目标函数：逆向 KL 散度

DMD 最小化学生分布 $p_{\text{fake}}$ 与教师分布 $p_{\text{real}}$ 的逆向 KL 散度：

$$
\min_{\theta} \; D_{KL}(p_{\text{fake}} \parallel p_{\text{real}}) = \mathbb{E}_{x \sim G_\theta(z)}\left[\log\frac{p_{\text{fake}}(x)}{p_{\text{real}}(x)}\right]
$$

- **逆向 KL** 的含义：用真实分布来"解释"生成分布。如果学生生成了在真实分布中概率极低的图，损失会急剧增大，从而强制学生生成逼真图像。
- **模式坍塌** vs **模式覆盖**：逆向 KL 倾向于"模式寻求"（mode-seeking）——宁可少覆盖几个模式，也要确保覆盖到的模式是真实的。这对图像生成是合适的。

### 2.2 梯度：两个 Score Function 之差

KL 散度对生成器参数的梯度可以表示为两个"得分函数"的差：

$$
\nabla_\theta D_{KL} \propto \mathbb{E}_{x \sim G_\theta(z)}\left[s_{\text{real}}(x) - s_{\text{fake}}(x)\right] \cdot \nabla_\theta G_\theta(z)
$$

其中 score function $s(x) = \nabla_x \log p(x)$ 表示在数据空间中指向更高概率密度的方向：

- **$s_{\text{real}}$**：指向"更真实图像"的方向（**引力**），由冻结的预训练模型提供
- **$-s_{\text{fake}}$**：指向"远离当前生成分布"的方向（**斥力**），防止模式坍塌，增加多样性
- **合力**：引力把生成图拉向真实分布，斥力把生成图推出 fake 分布的局部模式，两者合力让学生分布收敛到真实分布

### 2.3 为什么要加噪？

学生生成的图可能落在真实数据分布概率极低的区域（初始时，随机噪声生成的质量极差），此时 $s_{\text{real}}$ 没有意义——score function 只在数据分布的支撑集上有定义。

**解决方案**：向生成图加高斯噪声，使两个分布在噪声空间中重叠。

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

时间步采样范围：$t \in [0.02T, 0.98T]$，避免极端噪声状态（太小的 t 噪声不足，太大的 t 结构全毁）。

在噪声空间中，score function 可以近似为去噪网络的输出：

$$
s(x_t, t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

### 2.4 三个模型协作

DMD 训练涉及三个模型，均从同一个预训练扩散模型初始化：

| 模型 | 作用 | 是否更新 |
|------|------|----------|
| **Student Generator $G_\theta$** | 一步生成器（蒸馏目标） | ✅ 训练更新 |
| **Real Score Model** $\epsilon_{\text{real}}$ | 冻结的预训练扩散模型，提供 $s_{\text{real}}$（引力） | ❌ 参数冻结 |
| **Fake Score Model** $\epsilon_{\text{fake}}$ | 动态追踪学生分布，提供 $s_{\text{fake}}$（斥力） | ✅ 在线更新（Diffusion Loss） |

类比 GAN：

- Generator = 学生模型
- Discriminator = 被两个 score model 的差值替代
- 分布匹配的逻辑：不用判别器判断真假，而是用 score 梯度直接引导生成分布

### 2.5 训练流程

```
每个 iteration：
  1. Sample z ~ N(0,I) → Student G_θ(z) → 生成 fake image x
  2. 对 x 加噪声 at random timestep t ∈ [0.02T, 0.98T]
  3. 冻结的 real model 去噪 → s_real
  4. 在线更新的 fake model 去噪 → s_fake
  5. Distribution Matching Loss:
       L_DMD = ||x - (s_real - s_fake)||²   ← 简化的梯度近似
  6. + LPIPS regression loss (DMD1) / + GAN loss (DMD2)
  7. 更新 student G_θ
  8. Fake model 用标准 diffusion loss 更新 (追上新分布)
```

### 2.6 回归损失（仅 DMD1）

仅用分布匹配损失容易导致**模式坍塌/丢失**（只覆盖部分模式）。为此 DMD1 引入额外的 LPIPS 回归损失：

$$
\mathcal{L}_{\text{reg}} = \text{LPIPS}(G_\theta(z), y)
$$

- 需**离线预计算**百万级 $(z, \text{teacher\_output})$ 配对数据
- SDXL 上这一步骤需要 ~700 A100 GPU-days，超过训练本身的 4 倍
- **这是 DMD1 最大的痛点**

---

## 3. DMD1 → DMD2 的演进

DMD2 (NeurIPS 2024 Oral) 解决了 DMD1 的三大痛点：

| 改进点 | DMD1 (CVPR 2024) | DMD2 (NeurIPS 2024 Oral) |
|--------|------------------|--------------------------|
| **回归损失** | 需要预计算百万级 noise-image 对（SDXL 需 700 A100 GPU-days） | 完全移除，无需预计算 |
| **训练稳定性** | Fake model 跟不上生成器变化 | TTUR：每更新 1 次生成器，更新 5 次 Fake Model |
| **质量上限** | 无法超越教师模型的近似 | 加入 GAN Loss（在真实数据上训练），学生可**超越教师** |
| **多步推理** | 仅支持单步生成 | 支持多步采样 + Backward Simulation 消除训练-推理不匹配 |
| **ImageNet FID** | 2.62 | **1.28**（超越教师 1.49） |
| **Zero-shot COCO FID** | 11.49 | **8.35** |

DMD2 是蒸馏领域**首次学生超越教师**的案例。

---

## 4. 与其他蒸馏方法的对比

| | DMD / DMD2 | VSD (OSEDiff/TADSR) | ADD (SD-Turbo) | LCM |
|---|---|---|---|---|
| 数学原理 | $D_{KL}(p_{\text{fake}} \parallel p_{\text{real}})$ | $D_{KL}(p_{\text{student}} \parallel p_{\text{teacher}})$ | Adversarial + Score | PF-ODE 一致性 |
| Score 估计 | 双模型 ($s_{\text{real}}$, $s_{\text{fake}}$) | LoRA 微调的教师 | 无 score 蒸馏 | 无需 score |
| 是否需要 fake model | ✅ 需要在线维护 | ❌ 用 LoRA 教师近似 | ❌ | ❌ |
| 一步质量 | 极高 | 高 | 高 | 中 |
| 训练难度 | 中（需双 score 模型） | 中（需 LoRA 教师） | 低 | 低 |
| 能否超越教师 | ✅ (DMD2) | ❌ | ❌ | ❌ |
| 显存开销 | 高（三模型 + GAN） | 中（双模型） | 中 | 低 |

---

## 5. VSD vs DMD 的关系

OSEDiff/TADSR 用的 VSD（Variational Score Distillation）和 DMD 本质是**同一族方法**——都是最小化 KL 散度的 score-based 蒸馏。区别在于 fake score 的估计方式：

- **VSD**：用预训练扩散模型的一个 LoRA 副本（在真实 HQ 数据上微调）作为 $s_{\text{real}}$ 的替代品
- **DMD**：额外维护一个 fake score model，在**学生输出上训练**，确保斥力项始终追踪学生当前分布

DMD 的方案在数学上更严格（真正的 $s_{\text{fake}}$ 而非近似），但 fake score model 的维护成本更高。对一步图像复原来说，**VSD 可能已经足够**——复原任务有 GT 约束（L2/LPIPS），不像生成任务完全无监督。

---

## 6. 社区讨论与使用

### 中文社区（知乎/CSDN）

- 知乎 [分布蒸馏：图像分布蒸馏 DMD1&2](https://zhuanlan.zhihu.com/p/1983132967820870939) — 详细推导了 KL 散度到 score difference 的完整数学链路
- 知乎 [从 Z-Image 回望 DMD](https://zhuanlan.zhihu.com/p/1979964633361163321) — 分析了 DMD 在业界应用（通义 Z-Image 使用了 Decoupled DMD）
- CSDN [DMD 蒸馏核心：用 KL 散度 + LPIPS 损失搞定扩散模型单步生成（附 PyTorch 实现）](https://blog.csdn.net/stem5/article/details/153809158)
- 整体评价："理论和工程双优的蒸馏框架"，但训练代码复杂（三模型协作）
- DMD2 的 TTUR 和 GAN loss 被认为是工程化的实用改进

### 英文社区（Reddit/Twitter）

- [r/StableDiffusion DMD2 讨论](https://redlib.perennialte.ch/r/StableDiffusion/comments/1d02uoa/)：关注 DMD2 的 SDXL 蒸馏效果，认为比 SD-Turbo 和 LCM 质量好
- 普遍认为 DMD2 是当前图像蒸馏的 SOTA
- 主要吐槽：训练资源需求大（需要两个完整的扩散模型在显存里）、推理管线复杂
- 对一步超分/复原任务的适配讨论较少——DMD 系列主要在 text-to-image 上验证

### 业界应用

| 应用 | 说明 |
|------|------|
| **Z-Image (通义)** | Decoupled DMD 蒸馏 6B 模型，实现又快又好的图像生成 |
| **FastWan** | Sparse Attention + DMD 蒸馏加速视频生成 |
| **video-BLADE** | 稀疏注意力 + DMD 蒸馏 |
| **Phased DMD** (arXiv:2510.27684) | 分阶段 DMD，进一步降低训练成本 |

### 实用性评估

- ✅ 蒸馏后的一步质量极高（FID 1.28 超越教师）
- ✅ DMD2 不需要昂贵的预计算
- ❌ 训练需要双模型 + GAN discriminator，显存开销大（对 24GB 卡不太友好）
- ❌ 对复原任务的直接适配尚未充分验证
- ❌ Fake score model 的动态更新增加了训练 pipeline 复杂度

---

## 7. Key Takeaway

1. **从"轨迹匹配"到"分布匹配"是范式转变**——不需要逐点对齐，只需保证整体分布一致
2. **双 Score Function 的差 = KL 散度的梯度**——这是 DMD 的核心数学洞察
3. **Fake Score Model 是区分 VSD 和 DMD 的关键**——它确保斥力项始终追踪学生当前分布
4. **DMD2 的三个工程改进**（移除回归损失 + TTUR + GAN Loss）让框架真正可用
5. **对图像复原的启示**：VSD/DMD 式的分布正则化可能是解决"PSNR 低于 identity"的关键——用预训练扩散模型的 score 作为分布先验，强制输出"看起来像自然图像"
