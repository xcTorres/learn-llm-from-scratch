# VLM 知识总结（视觉语言模型）

> **系列**：本文档由原《LLM / VLM / Agent 面试题库》拆分而来，配套 [LLM 知识总结](LLM知识总结.md) 与 [Agent 知识总结](Agent知识总结.md)。
> **定位**：每题结构为 **核心答案 → 深入原理 → 权衡 / 追问 → 参考**。⭐ = 高频深挖点。

---

# 第四部分：VLM（视觉语言模型）

### 19. CLIP 的原理与意义 ⭐ `#多模态 #高频`
**【核心答案】** 双塔结构（图像编码器 + 文本编码器），用**对比学习**在海量图文对上把配对的图-文表征拉近、不配对的推远（InfoNCE / 对称交叉熵损失）。训练后获得强大的零样本分类与图文检索能力。

**【深入】**
- 一个 batch 有 N 个图文对，构成 N×N 相似度矩阵，对角线是正样本，其余是负样本，行/列两个方向都做交叉熵。
- 零样本分类：把类别名写成文本 prompt（"a photo of a {class}"），取与图像相似度最高的类别。
- 它学到的视觉编码器成为后续众多 VLM（如 LLaVA）的视觉骨干。

**【权衡 / 追问】**
- 局限：擅长「图文匹配/分类」，但**不能生成**文本回答；细粒度、计数、空间关系弱。
- 追问 **为什么用对比学习而不是分类**：对比学习能用海量弱标注（网络图文对），不依赖固定类别集，泛化与零样本能力强。

📖 参考：CLIP — https://arxiv.org/abs/2103.00020

---

### 20. 主流 VLM 怎么把视觉接入 LLM？对比几种连接方式 ⭐ `#多模态 #高频`
**【核心答案】** 典型三段式：**视觉编码器（如 CLIP ViT）→ 连接模块（projector）→ LLM**。区别主要在连接模块：

**【深入】**
- **LLaVA**：用一个 **MLP projector** 把视觉特征直接投影成 LLM 词嵌入空间的「视觉 token」，拼到文本 token 前。最简单、数据效率高，是当前主流范式。
- **BLIP-2**：用 **Q-Former**——一组可学习 query 通过 cross-attention 从冻结视觉编码器抽取固定数量（如 32 个）视觉 token，再喂给冻结 LLM。参数高效。
- **Flamingo**：在冻结 LLM 中**插入门控 cross-attention 层**注入视觉信息，用 Perceiver Resampler 把变长视觉特征压成固定 token，擅长少样本、图文交错输入。
- **Qwen-VL / InternVL**：强调**高/动态分辨率**（切图成多块），显著提升 OCR、文档、细节理解。

**【权衡 / 追问】**
- MLP（LLaVA）简单但视觉 token 多、吃 context；Q-Former（BLIP-2）token 少但训练更复杂、可能丢细节。
- 追问 **视觉编码器要不要解冻**：早期冻结，后期高质量 VLM 常解冻或用更大视觉塔以提细节。

📖 参考：LLaVA — https://arxiv.org/abs/2304.08485 ｜ BLIP-2 — https://arxiv.org/abs/2301.12597 ｜ Flamingo — https://arxiv.org/abs/2204.14198

---

### 21. LLaVA 的两阶段训练流程 ⭐ `#多模态`
**【核心答案】** ① **特征对齐预训练**：冻结视觉编码器和 LLM，只训 projector，用图文对让视觉特征对齐到 LLM 语义空间；② **指令微调**：解冻 projector 和 LLM，用多模态指令数据（VQA、推理、对话）训练成能对话的多模态助手。

**【深入】**
- 阶段一只学「翻译」——把视觉特征翻译成 LLM 能理解的「外语 token」，所以只动 projector。
- 阶段二的指令数据是关键创新：LLaVA 用纯文本 GPT-4 根据图像的 caption/bbox 生成多样的多模态指令-回答对（self-instruct 思路）。

**【权衡 / 追问】** 追问为什么阶段一冻结 LLM？避免少量对齐数据破坏 LLM 已有的语言能力；先建立模态桥梁再联合微调。

📖 参考：Visual Instruction Tuning (LLaVA) — https://arxiv.org/abs/2304.08485

---

### 22. 视觉 token 太多怎么办？高分辨率怎么处理？ `#多模态`
**【核心答案】** 高分辨率图切块会产生大量视觉 token，吃 context 又费算力。方案：token 压缩/合并（resampler、pooling、pixel shuffle）、Q-Former 用固定数量 query、动态分辨率按需分配 token。

**【深入】**
- LLaVA-1.5/NeXT 用「切图成多个子图 + 缩略图」处理高分辨率，token 数随分辨率增长。
- pixel shuffle / token merging：把相邻视觉 token 合并降数量。
- 视频更极端：要在时间维再做帧采样和 token 压缩。

**【权衡 / 追问】** 追问压缩的代价：token 越少越省算力，但 OCR、细粒度任务越受损，是分辨率-效率权衡。

---

### 23. VLM 常见能力短板与评测基准 `#多模态`
**【核心答案】** 短板：细粒度 OCR、计数、空间/方位关系、长视频时序、物体级幻觉（描述图中不存在的东西）。评测：MMMU、MMBench、MME、DocVQA、TextVQA、ChartQA，幻觉专项 POPE。

**【深入】**
- 物体幻觉常因语言先验过强（模型「猜」常见搭配而非真看图）。
- 评测要区分「感知」（看得见吗）和「认知/推理」（看懂了吗）两类能力。

**【权衡 / 追问】** 追问怎么缓解物体幻觉：增强视觉分辨率、负样本训练、对齐时惩罚无依据描述（如 POPE 式探测）。

---

### 24. 对比式 vs 生成式多模态训练，怎么选？ `#多模态`
**【核心答案】** 对比式（CLIP）给出对齐的图文表征，擅长检索/分类但不能对话；生成式（LLaVA 类）能看图对话、推理，是当前「多模态大模型」主流，且常以对比学习得到的编码器为视觉起点。两者是「表征」与「生成」的分工，常组合使用。

**【深入】** 实际 pipeline 往往是：CLIP 式预训练得到强视觉编码器 → 接 LLM 做生成式指令微调。两条路线互补而非对立。

---

# 第四部分（补充）：扩散模型（Diffusion / 文生图）

> 上面 19–24 讲的是「看图说话」的**理解式**多模态；这一节讲**生成式**多模态——从文本/噪声生成图像与视频，是 Stable Diffusion、Sora 的底层范式。

### 25. 扩散模型（Diffusion Model）的核心原理？⭐ `#多模态 #生成 #高频`
**【核心答案】** 一对方向相反的过程：**前向（扩散）** 逐步给真实图像加高斯噪声，直到变成纯噪声；**反向（去噪）** 训练一个神经网络逐步预测并去掉噪声，从纯噪声还原图像。本质是「学会去噪」，采样时从随机噪声出发反复去噪即可生成新样本。

**【深入】**
- 前向是**固定、无参数**的马尔可夫链，且有闭式解——任意时刻 $x_t$ 可由原图 $x_0$ 一步直接采样（$\bar\alpha_t$ 是噪声调度的累积系数，$\epsilon\sim\mathcal{N}(0,I)$）：

  $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

- 反向用网络 $\epsilon_\theta(x_t,t)$ **预测当初加入的噪声**（DDPM 证明预测噪声比直接预测均值更稳）。训练目标极简：

  $$\mathcal{L} = \mathbb{E}_{x_0,t,\epsilon}\big[\,\|\epsilon - \epsilon_\theta(x_t,t)\|^2\,\big]$$

  即：随机取一张图、一个时间步 $t$、一份噪声，让网络把这份噪声预测出来。
- 去噪网络主干通常是带时间步嵌入与自注意力的 **U-Net**；新一代改用 Transformer（**DiT**，见第 28 题）。

**【权衡 / 追问】**
- 追问 **为什么比 GAN 好训**：没有判别器对抗，训练稳定、模式覆盖好（不易模式崩溃）；代价是**采样慢**（要几十~上千步迭代）。
- 追问 **和 VAE/GAN 的关系**：都是生成模型；扩散可看作「多步、层级化的去噪自编码器」，用计算换质量与稳定性。

📖 参考：DDPM — https://arxiv.org/abs/2006.11239

---

### 26. 扩散采样为什么慢？怎么加速？DDIM 是什么？ `#多模态 #生成`
**【核心答案】** DDPM 采样要沿马尔可夫链一步步去噪（常 1000 步），慢。**DDIM** 把采样改成**确定性、非马尔可夫**过程，可大步跳过、几十步出图且结果可复现；**DPM-Solver** 等高阶 ODE 求解器再压到 ~10–20 步；**蒸馏路线（LCM / 一致性模型）** 甚至做到几步乃至一步。

**【深入】**
- 把反向过程看成求解一条**概率流 ODE/SDE**，于是能套用数值求解器加速。
- DDIM 用同一个训练好的模型，仅在推理时换确定性更新，步数与质量可自由权衡。
- 一致性模型 / LCM 直接学「一步到位」的映射，用于实时生成。

**【权衡 / 追问】** 追问代价：步数越少越快，但细节与保真度下降——典型的速度-质量权衡。

---

### 27. Latent Diffusion / Stable Diffusion 为什么强？文本怎么控制生成？⭐ `#多模态 #生成 #高频`
**【核心答案】** Stable Diffusion 就是 **Latent Diffusion**，三个关键点：① **不在像素空间扩散**，先用 VAE 把图像压到低维 latent 再扩散，算力大降；② 用 **CLIP 文本编码器** 编码 prompt，经 **cross-attention** 注入 U-Net 实现文本条件；③ 用 **classifier-free guidance（CFG）** 增强与文本的贴合度。

**【深入】**
- VAE encoder 把 512×512 图压成 64×64 的 latent，扩散全程在 latent 上做，decoder 再还原像素——**计算量降一两个数量级**，这是 SD 能在消费级显卡上跑的关键。
- 文本条件注入：`prompt → CLIP text encoder → 作为 cross-attention 的 K/V` 喂进 U-Net 每一层。
- 其它条件控制：**ControlNet**（边缘/深度/姿态等结构控制）、**LoRA**（轻量风格微调）、**IP-Adapter**（图像参考）。

**【权衡 / 追问】**
- 追问 **CFG 原理**：训练时随机丢弃文本，得到「有条件 $\epsilon_\theta(x_t,c)$」与「无条件 $\epsilon_\theta(x_t,\varnothing)$」两支；采样时外推放大条件方向（$w$ 为 guidance scale）：

  $$\hat\epsilon = \epsilon_\theta(x_t,\varnothing) + w\,\big(\epsilon_\theta(x_t,c) - \epsilon_\theta(x_t,\varnothing)\big)$$

  $w$ 越大越贴合 prompt，但多样性与自然度下降。
- 追问 **为什么 latent 空间够用**：VAE 已去除像素冗余、保留语义与结构，扩散只需建模「感知上相关」的信息。

📖 参考：Latent Diffusion / Stable Diffusion — https://arxiv.org/abs/2112.10752 ｜ Classifier-Free Guidance — https://arxiv.org/abs/2207.12598

---

### 28. 扩散模型的前沿：DiT、Flow Matching、与自回归生成之争 `#多模态 #生成`
**【核心答案】** 三个趋势：① 主干从 U-Net 转向 **Transformer（DiT）**，更易 scale，被 Sora / SD3 采用；② 训练目标从 DDPM 的噪声预测转向 **Flow Matching / Rectified Flow**（路径更直、采样更少步，SD3 等采用）；③ 图像/视频生成中扩散仍是主流，但 **next-token 自回归生成**（VAR、原生多模态）也在追赶。

**【深入】**
- **DiT**：把 latent 切成 patch 当 token，用 Transformer 做去噪，scaling law 友好。
- **Flow Matching / Rectified Flow**：直接学一条从噪声到数据的「近似直线」概率路径，采样步数更少。
- **视频生成（Sora）** = 时空 patch + DiT + 扩散。

**【权衡 / 追问】** 追问 **扩散 vs 自回归生成图像**：扩散并行去噪、质量高但步数多；自回归逐 token、可与 LLM 统一架构但长序列慢。

📖 参考：DiT — https://arxiv.org/abs/2212.09748 ｜ Flow Matching — https://arxiv.org/abs/2210.02747

---

# 第五部分：视频理解（Video Understanding）

> 视频 = 图像 + **时间维度**。难点不在"看懂一帧"，而在 ①帧数一多 token 就爆炸、撞 context 上限；②要建模**时序**（动作、事件顺序、因果）；③在线/流式场景要边看边答。这一节是第 22 题「视觉 token 太多」在视频上的放大版。

### 29. 视频怎么接入 LLM？基本范式与核心矛盾 ⭐ `#多模态 #视频 #高频`
**【核心答案】** 主流范式是 **帧采样 → 视觉编码器逐帧编码 → token 压缩/聚合 → 拼成视频 token 序列喂给 LLM**。核心矛盾：一段视频抽几十~上千帧，每帧又是几十~几百个 token，**token 数 = 帧数 × 每帧 token**，极易撞穿 LLM 的 context 上限，所以**怎么压 token** 是视频 VLM 的第一命题。

**【深入】**
- **帧采样**：均匀采样（每隔几秒一帧）最简单；长视频常按时长动态调采样率（短视频 1fps、长视频 0.2fps）。进阶用**关键帧/关键片段选择**（按与问题的相关性挑帧），避免均匀采样漏掉关键瞬间。
- **逐帧编码**：复用图像 VLM 那套（CLIP ViT / SigLIP），每帧独立编码成一组 token。
- **聚合成视频 token**：直接拼接（token 最多）、时间维 pooling、Q-Former/Resampler 把每帧或整段压成固定数量 token、相邻帧 token merging。
- 代表：**Video-LLaVA / LLaVA-NeXT-Video**（图像范式直接扩到视频）、**VideoLLaMA**（音视频联合）、**Qwen2.5-VL / InternVL**（动态分辨率 + 动态帧率，支持 1 小时级视频与事件定位）。

**【权衡 / 追问】**
- 追问 **为什么不能像图像一样无脑拼 token**：一段 1 分钟视频 30fps = 1800 帧，每帧 256 token ≈ 46 万 token，远超 context，必须采样+压缩。
- 追问 **采样稀疏的代价**：采太稀会丢掉快速动作/短事件（时序信息损失），采太密 token 爆炸——典型的覆盖率-效率权衡。

📖 参考：Video-LLaVA — https://arxiv.org/abs/2311.10122 ｜ Qwen2.5-VL — https://arxiv.org/abs/2502.13923

---

### 30. 长视频的 token 爆炸怎么解？时序信息怎么保留？ ⭐ `#多模态 #视频 #高频`
**【核心答案】** 两条线：**①空间-时间 token 压缩**（利用视频的时空冗余——静止背景、重复场景——把 token 压到极低保留率）；**②时序位置编码**（让 LLM 知道每个视觉 token 来自第几帧/第几秒，否则模型分不清前后顺序）。

**【深入】**
- **token 压缩**（视频版第 22 题）：
  - 时空冗余裁剪：相邻帧大量重复，做 spatio-temporal token merging / pruning，只留变化大的 token。
  - 极致压缩：每帧压到几个甚至 **1 个 token**（如 STORM 用 Mamba 做时序聚合、"one token per frame"路线），把整段长视频塞进有限 context。
  - 训练-free 路线：推理时基于树结构/相似度直接合并 token，不额外训练（FlashVid 等）。
- **时序位置编码**：关键在让模型感知"时间"。Qwen2-VL 的 **M-RoPE（多模态旋转位置编码）** 把位置拆成时间/高/宽三个维度，视频 token 在时间维上递增，从而保留帧间顺序与绝对时间戳——这是它能做**精确事件定位（temporal grounding）** 的基础。

**【权衡 / 追问】**
- 追问 **压缩到 1 token/帧会丢什么**：丢帧内空间细节（OCR、小物体），但保住了时序覆盖——长视频"看全程"比"看清每一帧"更重要时划算。
- 追问 **为什么时序编码重要**：没有时序位置，"他先开门还是先关灯"这类因果/顺序问题模型只能瞎猜（视觉先验幻觉）。

📖 参考：STORM — https://arxiv.org/abs/2503.04130 ｜ Qwen2-VL (M-RoPE) — https://arxiv.org/abs/2409.12191

---

### 31. 流式 / 在线视频理解（Streaming）和离线有什么不同？ `#多模态 #视频`
**【核心答案】** 离线是"整段视频都拿到了再回答"；**流式是边看边处理、随时可被提问、且要实时响应**（直播、具身机器人、AR 助手）。关键差异：不能预知未来帧、不能把全部历史都塞进 context，需要**记忆机制**增量维护历史，并支持**主动/及时**触发回答。

**【深入】**
- 核心挑战：① 历史无限增长 → 需把过去压成**记忆/summary token**（段级记忆、KV 缓存裁剪）；② 何时该说话 → 模型要学会判断"现在是否到了该回答/预警的时刻"（anticipatory / proactive）。
- 代表方向：VideoLLM-online、Flash-VStream（流式记忆）、StreamAgent（预判式 agent）、段级记忆做多轮视频推理。

**【权衡 / 追问】** 追问 **流式为什么不能直接用离线模型**：离线模型假设全局可见、一次性推理；流式要在信息不完整下增量决策，且对延迟敏感，是两套设定。

📖 参考：StreamingBench — https://arxiv.org/abs/2411.03628

---

### 32. 视频理解怎么评测？有哪些基准和短板？ `#多模态 #视频`
**【核心答案】** 按"视频长度 + 能力维度"分：**短视频 QA**（MSRVTT、ActivityNet-QA）、**综合/长视频**（Video-MME、MLVU、LongVideoBench、EgoSchema 偏长时序推理）、**流式**（StreamingBench）。短板集中在：长时序因果、事件顺序与计数、跨远距离帧的关联、细粒度动作识别。

**【深入】**
- **Video-MME**：全谱评测，900 视频（短 11s ~ 长 60min）、覆盖六大领域，区分感知/推理/信息抽取。
- **LongVideoBench**：长达 1 小时的图文交错（含字幕）长上下文理解。
- **EgoSchema**：第一人称长视频，强调超长时序推理（必须看全程才能答）。
- **StreamingBench**：实时/全源/上下文三类共 18 任务，专测流式能力。

**【权衡 / 追问】**
- 追问 **VLM 在长视频上的典型失败**：因帧采样稀疏，模型常退化成"看几帧 + 语言先验猜"，做不了真正的全程时序推理（评测里换帧顺序答案不变 = 没在用时序）。
- 追问 **怎么判断模型真在用时序**：打乱/反转帧序看答案是否变化；考时间定位（"事件发生在第几秒"）这类必须依赖时序的任务。

📖 参考：Video-MME — https://arxiv.org/abs/2405.21075 ｜ EgoSchema — https://arxiv.org/abs/2308.09126

---

# 附录 A：核心论文索引（多模态 VLM）

**理解式 VLM**
- ✅ CLIP — https://arxiv.org/abs/2103.00020
- ✅ Flamingo — https://arxiv.org/abs/2204.14198
- ✅ BLIP-2 — https://arxiv.org/abs/2301.12597
- ✅ LLaVA (Visual Instruction Tuning) — https://arxiv.org/abs/2304.08485

**生成式 / 扩散模型**
- ✅ DDPM — https://arxiv.org/abs/2006.11239
- ✅ Latent Diffusion / Stable Diffusion — https://arxiv.org/abs/2112.10752
- ✅ Classifier-Free Guidance — https://arxiv.org/abs/2207.12598
- ✅ DiT (Scalable Diffusion with Transformers) — https://arxiv.org/abs/2212.09748
- ✅ Flow Matching — https://arxiv.org/abs/2210.02747

**视频理解**
- ✅ Video-LLaVA — https://arxiv.org/abs/2311.10122
- ✅ Qwen2-VL (M-RoPE) — https://arxiv.org/abs/2409.12191 ｜ Qwen2.5-VL — https://arxiv.org/abs/2502.13923
- ✅ STORM (token-efficient 长视频) — https://arxiv.org/abs/2503.04130
- ✅ Video-MME (评测) — https://arxiv.org/abs/2405.21075 ｜ StreamingBench — https://arxiv.org/abs/2411.03628 ｜ EgoSchema — https://arxiv.org/abs/2308.09126

---

# 附录 C：待补充清单（VLM）

- [x] 视频理解（帧采样 / token 压缩 / 时序编码 / 流式 / 评测）→ 见第 29–32 题
- [ ] 原生多模态、any-to-any、理解+生成统一（Janus / Show-o / Qwen3-VL）
- [ ] 多模态推理 + RL（RLVR：MM-Eureka / PEARL）
- [ ] 扩散进阶（ControlNet / LoRA / IP-Adapter 条件控制、视频扩散 Sora 细节）
