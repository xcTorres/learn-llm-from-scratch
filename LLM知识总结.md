# LLM 知识总结（基础 / 训练 / 推理）

> **系列**：本文档由原《LLM / VLM / Agent 面试题库》拆分而来，配套 [VLM 知识总结](VLM知识总结.md) 与 [Agent 知识总结](Agent知识总结.md)。
> **定位**：从「考前速记」升级为「可持续维护的学习地图」。每题结构为 **核心答案 → 深入原理 → 权衡 / 追问 → 参考**。
>
> **复习用法**
> - 时间紧（考前）：只看每题的 **【核心答案】** 和 **【追问】**，能一句话讲清就过。
> - 系统学（平时）：读 **【深入】**，顺着文末的「核心论文索引」回到原始论文精读。
> - ⭐ = 高频深挖点，务必准备 30 秒口述版（结论→原理→权衡）。
>
> 标签体系：`#基础` `#工程` `#对齐` `#多模态` `#agent` `#高频` `#手撕`。

---

# 第零部分：数学基础（贯穿全篇）

> 这几个量在后面反复出现：**交叉熵**是预训练/SFT 的损失，**KL 散度**是 RLHF/PPO/DPO/GRPO 里约束策略别跑偏的核心项，蒸馏里又用 KL 对齐师生分布。先把它们和彼此的关系讲清楚，后面就不再展开。

### 0.1 交叉熵、KL 散度、熵三者什么关系？⭐ `#基础 #高频`
**【核心答案】** 设真实分布 p、模型分布 q：
- **熵** $H(p) = -\sum_x p(x)\log p(x)$：p 自身的不确定性（理论最小编码长度）。
- **交叉熵** $H(p,q) = -\sum_x p(x)\log q(x)$：用 q 去编码 p 的平均代价。
- **KL 散度** $D_{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$：用 q 近似 p 的「额外」代价。

三者关系一行话：**交叉熵 = 熵 + KL**，即

  $$H(p,q) = H(p) + D_{KL}(p\,\|\,q)$$

**【深入】**
- 因为 H(p) 与模型参数无关（p 是固定的真实标签分布），**最小化交叉熵 ⇔ 最小化 KL**——这就是为什么训练直接用交叉熵当 loss。
- **KL 非负、不对称**：$D_{KL}(p\|q)\neq D_{KL}(q\|p)$，所以它是「散度」不是「距离」。$D_{KL}=0 \iff p=q$。
- 不对称的实际含义：
  - **Forward KL** $D_{KL}(p\|q)$（最大似然用的）：p 有质量处 q 不能为 0，否则惩罚无穷 → q 倾向「覆盖所有模式」（mean-seeking，分布偏胖）。
  - **Reverse KL** $D_{KL}(q\|p)$（变分推断、部分 RL 用）：q 只敢在 p 高的地方放质量 → 倾向「锁定单一模式」（mode-seeking）。

**【权衡 / 追问】**
- 追问 **为什么对称化**：JS 散度 = 两个方向 KL 的平均，对称且有界，GAN 里用过。
- 追问 KL 在 LLM 里具体算什么：是**两个 token 分布的逐位置 KL**，PPO/DPO 里约束「新策略 π_θ 别偏离参考策略 π_ref 太远」，防止 reward hacking（见第 9、10 题）。

### 0.2 语言模型的交叉熵损失长什么样？和困惑度什么关系？⭐ `#基础 #高频`
**【核心答案】** 真实分布是 one-hot（真实下一个词 = 1，其余 = 0），所以交叉熵退化成**负对数似然（NLL）**：只看「模型给正确词的概率」。

  $$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log q_\theta(x_t \mid x_{<t})$$

**【深入】**
- one-hot 下 $\sum_x p(x)\log q(x)$ 只剩正确类那一项，所以交叉熵 = $-\log q(\text{正确词})$，预测得越准（概率越接近 1）loss 越接近 0。
- **困惑度（Perplexity）** = $\exp(\mathcal{L})$，即交叉熵的指数，直观理解为「模型在每步平均在多少个词里纠结」，越低越好。
- 分类任务里 softmax + 交叉熵的梯度极简洁：$\partial \mathcal{L}/\partial z_i = q_i - p_i$（预测概率 − 真实标签），这也是它好训练的原因。

**【权衡 / 追问】**
- 追问 **label smoothing**：把 one-hot 的 1 改成 1−ε、其余分一点 ε，等价于给目标分布掺入均匀分布，缓解过度自信、改善校准。
- 追问 **蒸馏（distillation）为何用 KL 而非交叉熵**：教师输出是「软分布」而非 one-hot，要让学生匹配整个分布，所以最小化 $D_{KL}(p_{teacher}\|q_{student})$（带温度 T 软化）——这里 KL ≠ 交叉熵，因为教师熵不为 0、不可忽略。

📖 参考：交叉熵/KL 基础见《Deep Learning》(Goodfellow) Ch.3 ｜ 蒸馏 — https://arxiv.org/abs/1503.02531

### 0.3 常见激活函数与选型？为什么大模型爱用 GLU 变体？⭐ `#基础 #高频`
**【核心答案】** 激活函数提供**非线性**，否则多层线性叠加仍等价于一层。从 Sigmoid/Tanh → ReLU → GELU/Swish → 现在 LLM 主流的 **SwiGLU / GeGLU**（门控线性单元变体）。

**【深入】**
| 激活 | 公式 | 特点 / 问题 |
|------|------|-----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 输出(0,1)；两端饱和→**梯度消失**，非零均值 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 零均值、仍饱和 |
| ReLU | $\max(0,x)$ | 简单、不饱和、收敛快；但负区恒 0→**神经元死亡** |
| LeakyReLU | $\max(\alpha x,x)$ | 负区给小斜率，缓解死亡 |
| GELU | $x\cdot\Phi(x)$ | 用高斯 CDF 平滑加权，BERT/GPT 常用，平滑可导 |
| Swish/SiLU | $x\cdot\sigma(x)$ | 平滑、非单调，深层表现好 |
| **SwiGLU** | $(\mathrm{Swish}(xW)\otimes xV)W_2$ | **门控**：一支做内容、一支做门控相乘；LLaMA/PaLM 采用 |

- **GLU 变体（SwiGLU/GeGLU）凭什么强**：把 FFN 从「一条路」变成「内容 × 门控」两条路逐元素相乘，**门控让网络动态决定每个维度放多少信息通过**，同等算力下质量更好（GLU Variants 论文实验证实）。
- 注意 SwiGLU 的 FFN 有 3 个权重矩阵（W, V, W_2），为保持参数量持平，中间维度常取 $\frac{2}{3}\times 4d$ 而非 4d。

**【权衡 / 追问】**
- 追问 **为什么不用 Sigmoid 当隐藏层激活**：饱和区梯度趋 0、非零均值导致梯度更新呈锯齿，深层难训；现在 Sigmoid 只用在二分类输出或门控。
- 追问 GELU vs ReLU：GELU 平滑、处处可导、在 Transformer 上略优；ReLU 更省算力。

### 0.4 优化器演进：SGD → Adam → AdamW，大模型怎么选？⭐ `#基础 #高频`
**【核心答案】** 主线是「**动量**（平滑梯度方向）+ **自适应学习率**（每个参数各自缩放）」。LLM 训练几乎默认 **AdamW**。

**【深入】**
- **SGD**：$\theta \leftarrow \theta - \eta\,g$，简单但对学习率敏感、易在沟壑震荡。
- **+ Momentum**：累积历史梯度的指数滑动平均 $v$，沿一致方向加速、抑制震荡。
- **AdaGrad / RMSProp**：用历史梯度平方做自适应步长；AdaGrad 学习率单调衰减易过早停，RMSProp 用滑动平均修正。
- **Adam = Momentum + RMSProp**：同时维护一阶矩 $m_t$（方向）和二阶矩 $v_t$（尺度），加偏差校正：

  $$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$

  $$\hat m_t=\tfrac{m_t}{1-\beta_1^t},\ \hat v_t=\tfrac{v_t}{1-\beta_2^t},\quad \theta\leftarrow\theta-\eta\,\tfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

- **AdamW**：把 **weight decay 从梯度里解耦**出来，直接作用在权重上（$\theta\leftarrow\theta-\eta(\dots)-\eta\lambda\theta$）。Adam 里 L2 正则会被自适应分母缩放而失效，AdamW 修正了这点，**正则更干净、泛化更好**，是 LLM 标配。

**【权衡 / 追问】**
- 追问 **为什么大模型不用 SGD**：梯度噪声大、各参数尺度差异大，自适应方法收敛更稳更快；代价是 Adam 要存 m、v 两份状态，**显存约为参数量的 2 倍**（混合精度下更多），这也是 ZeRO/优化器状态切分要解决的问题。
- 追问学习率调度：大模型常用 **warmup + cosine 衰减**——warmup 防早期大梯度炸训，cosine 平滑退火。
- 追问新优化器：**Lion**（只存一阶动量、更省显存）、**Adafactor**（分解二阶矩省显存，T5 用）、**Muon/Shampoo**（二阶/矩阵预条件，近年大模型尝试）。

📖 参考：Adam — https://arxiv.org/abs/1412.6980 ｜ AdamW — https://arxiv.org/abs/1711.05101 ｜ GLU Variants — https://arxiv.org/abs/2002.05202

---

# 第一部分：LLM 基础与架构

### 1. Transformer 的核心是什么？为什么能取代 RNN？⭐ `#基础 #高频`
**【核心答案】** 核心是 self-attention：每个 token 能直接与序列中所有 token 交互，任意两个位置的路径长度为 O(1)，且整个序列可并行计算。

**【深入】**
- RNN 的两大问题：① 信息要逐步传递，长程依赖在反向传播中梯度消失/爆炸；② 时间步必须串行，无法并行，训练慢。
- Transformer 用注意力一步建立全局连接，并用残差连接 + LayerNorm 稳定深层训练，配合大规模数据和算力实现 scaling。

**【整体结构（自底向上的数据流）】**
- **输入层**：Token Embedding（把 id 查表成向量）+ Position Encoding（注入位置，见第 3 题）→ 得到序列表示 $X\in\mathbb{R}^{n\times d}$。
- **N 个堆叠的 Transformer Block**（GPT-3 96 层、LLaMA-3 数十层），每个 block 内两个子层：
  1. **多头自注意力（MHSA）子层**：`x → LN → MHSA → +x`（残差）
  2. **前馈网络（FFN/MLP）子层**：`x → LN → FFN → +x`（残差）
- **输出层**：最后一层 LN → 线性投影到词表大小（LM Head，常与输入 embedding **权重共享**）→ softmax 得到下一个 token 分布。
- 关键张量维度始终是 $n\times d$（n=序列长，d=hidden）：注意力做「token 间」混合（跨位置交互），FFN 做「token 内」逐位置的特征变换，两者交替，是 Transformer 的基本节奏。
- 一个 block 内：**多头自注意力 +（残差&归一化）+ 前馈网络 FFN +（残差&归一化）**。FFN 通常是 4×hidden 的两层 MLP（现代用 SwiGLU，见第 0.3 节），承担大部分参数量与「知识存储」。

**【残差连接：为什么不可或缺】**
- 形式：$\text{out} = x + \text{Sublayer}(\text{LN}(x))$（Pre-LN 写法）——子层只学「相对输入的增量 $F(x)$」，输出 = 输入 + 增量。
- **作用① 解决深层梯度消失**：反向传播时 $\frac{\partial \text{out}}{\partial x} = 1 + \frac{\partial F}{\partial x}$，那个 **「+1」让梯度有一条直达底层的「高速公路」**，即使 $\partial F/\partial x$ 很小，梯度也不会衰减到 0，几十上百层才训得动（源自 ResNet 思想）。
- **作用② 恒等映射易学**：如果某层最优是「什么都不做」，网络只需让 $F(x)\to 0$，比让一堆非线性层去拟合恒等函数容易得多 → 加深网络不会变差。
- **作用③ 信息保留**：每层都把原始信息原样带下去，再叠加新信息，避免深层把底层特征「冲刷」掉；可理解为对表示的**迭代式精炼（residual stream）**——这也是机理可解释性里把残差流看作「各层读写的公共总线」的由来。
- 配套的 **LayerNorm**：对每个 token 的特征维做归一化，稳定数值分布、加速收敛；LLM 常用更省的 **RMSNorm**（去掉均值中心化，只按均方根缩放）。

**【权衡 / 追问】**
- 代价是注意力的 O(n²) 复杂度（时间和显存都随序列长度平方增长），这是长序列优化的根源问题。
- 可能追问：**Pre-LN vs Post-LN**？Post-LN（原始 Transformer，`LN(x+Sublayer(x))`）表达力略强但深层梯度不稳、需 warmup；**Pre-LN**（`x+Sublayer(LN(x))`，归一化放子层输入端）让残差通路是干净的恒等映射、训练更稳、可去 warmup，是现在主流。
- 可能追问：FFN 为什么要先升维再降维？提供非线性容量、充当 key-value 记忆。
- 可能追问：**没有残差会怎样**？深层 Transformer 几乎无法收敛——这是 Pre-LN/残差/归一化「三件套」共同保证可训练性的核心原因。

📖 参考：Attention Is All You Need — https://arxiv.org/abs/1706.03762 ｜ ResNet（残差）— https://arxiv.org/abs/1512.03385

---

### 2. 手撕 self-attention，并解释每一步 ⭐ `#手撕 #高频`
**【核心答案】** Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V。Q/K/V 由输入分别乘三个可学习矩阵得到；QKᵀ 算相似度，缩放后 softmax 归一化成权重，再加权求和 V。

**【深入】**
- **为什么除以 √dₖ**：Q、K 各维度近似独立、均值 0 方差 1 时，点积的方差约等于 dₖ。维度越大点积越大，softmax 进入饱和区，梯度趋近 0。除以 √dₖ 把方差拉回 1 附近，保证梯度健康。
- **多头注意力**：把 d 维切成 h 个 d/h 维子空间并行做注意力，让不同头关注不同关系（语法、指代、位置等），最后拼接再线性映射。
- **因果掩码**：生成式模型在 softmax 前把「未来位置」置为 -∞，保证第 i 个 token 只能看到 ≤ i 的信息。

**【权衡 / 追问】**
- 追问 **复杂度**：序列长 n、维度 d，注意力是 O(n²·d)。
- 追问 **MHA / MQA / GQA 区别**：见第 5 题。
- 手撕时别忘了 mask 和 dropout 的位置，以及最后还有一层输出投影 Wₒ。

参考代码骨架（PyTorch 伪代码）：
```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, h, n, n]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = scores.softmax(dim=-1)
    return attn @ V
```

---

### 3. 位置编码有哪些？RoPE 为什么成为主流？⭐ `#基础 #高频`
**【核心答案】** 注意力本身对位置不敏感，需要显式注入位置信息。主流分三类：绝对位置编码、相对位置编码、旋转位置编码（RoPE）。RoPE 因为「统一了绝对与相对、外推性好、实现简单且不增加参数」成为 LLaMA / Qwen / 主流开源模型的标配。

**【深入】**
- **绝对位置编码**：原始 Transformer 用正弦函数（无参、可外推一点）或可学习位置向量（BERT，长度固定、不可外推）。直接加到 embedding 上。
- **RoPE（旋转位置编码）**：把每个 token 的 Q、K 按其位置 m 旋转一个角度（按维度成对做二维旋转）。数学上使得 ⟨q_m, k_n⟩ 只依赖相对距离 (m−n)，从而把相对位置信息天然编码进注意力内积。
- **ALiBi**：不改 Q/K，而是在注意力分数上加一个随距离线性递减的偏置（越远惩罚越大），无显式位置向量，外推性好。

**【权衡 / 追问】**
- 追问 **长上下文外推**：RoPE 直接外推到训练长度之外会退化，常用 **位置插值（PI）** 或 **NTK-aware / YaRN** 调整 base 频率来扩展上下文。
- 追问 RoPE 的 base（θ）超参影响：base 越大，长距离区分度越好，是长上下文调优的关键旋钮。

📖 参考：RoFormer (RoPE) — https://arxiv.org/abs/2104.09864

---

### 4. Decoder-only / Encoder-only / Encoder-Decoder 的区别与选型 `#基础`
**【核心答案】** 区别在注意力方向和结构：Encoder 双向、Decoder 单向因果、Encoder-Decoder 两者皆有。现代大模型几乎全是 Decoder-only。

**【深入】**
- **Encoder-only（BERT）**：双向注意力，每个 token 看全局，擅长理解类任务（分类、NER、检索）。不能直接生成。
- **Decoder-only（GPT/LLaMA）**：因果掩码、自回归，预训练目标统一为 next-token prediction，scaling 友好、few-shot 能力强，是目前主流。
- **Encoder-Decoder（T5/BART）**：编码器理解输入、解码器生成输出，擅长明确的 seq2seq（翻译、摘要）。

**【权衡 / 追问】**
- 追问 **为什么大模型都收敛到 Decoder-only**？训练目标简单统一、零样本/少样本泛化好、工程上推理范式一致；研究也表明在大规模下纯解码器架构的表现不输甚至更优。
- 追问 **PrefixLM**：一种折中，prompt 部分用双向注意力、生成部分用因果注意力。

📖 参考：BERT — https://arxiv.org/abs/1810.04805 ｜ GPT-3 — https://arxiv.org/abs/2005.14165

---

### 5. KV Cache 是什么？为什么是推理瓶颈？MQA / GQA 如何缓解？⭐ `#工程 #高频`
**【核心答案】** 自回归生成时历史 token 的 K、V 不变，缓存它们避免每步重算，把每步注意力从 O(n²) 降到 O(n)。代价是显存：KV cache ∝ batch × 层数 × 序列长 × head数 × head维。

**【深入】**
- 没有 KV cache，生成第 t 个 token 要重新计算前 t-1 个的全部注意力；有了它，只算当前 token 对历史的注意力。
- 长上下文 + 大 batch 下，KV cache 往往比模型权重还吃显存，是 decode 阶段的主要瓶颈。
- **MQA（Multi-Query Attention）**：所有 head 共享同一组 K/V，KV cache 缩小 head 倍，但质量有损。
- **GQA（Grouped-Query Attention）**：把 head 分成 g 组，每组共享一组 K/V，是 MHA 与 MQA 的折中，LLaMA-2/3 采用。

**【权衡 / 追问】**
- 追问 **怎么进一步省 KV cache**：量化 KV（KV cache int8）、PagedAttention 分页管理（见第 17 题）、滑动窗口注意力、把不重要的 KV 驱逐（H2O 等）。
- 追问 prefill 阶段需不需要 cache？prefill 一次性并行算完所有输入并把 K/V 写入 cache，供后续 decode 复用。

**【扩展：LLM 服务性能指标】** —— 面试常被追问「怎么衡量一个推理服务好不好」，分四类：

- **延迟（Latency，单请求体验）**
  - **TTFT（Time To First Token）**：从请求到首 token 的时间，主要由 **prefill** 阶段决定，受输入长度、是否命中 prefix cache 影响；决定「等待感」。
  - **TPOT / ITL（Time Per Output Token / Inter-Token Latency）**：相邻输出 token 的平均间隔，主要由 **decode** 阶段决定，决定「打字速度」。
  - **端到端延迟 ≈ TTFT + TPOT × 输出 token 数**；常用 P50/P90/P99 而非均值，长尾更能反映体验。

- **吞吐（Throughput，集群效率 / 成本）**
  - **Output tokens/s**（最常用）、**Total tokens/s**（含输入）、**Requests/s（QPS）**。
  - **吞吐 vs 延迟是一对矛盾**：增大 batch（continuous batching）提升吞吐，但单请求 TPOT 变差。常画「吞吐-延迟」曲线，在满足 SLO 的前提下追求最高吞吐。

- **资源 / 成本**
  - **GPU 利用率（MFU / 算力利用率）**、**显存占用**（权重 + KV cache，决定最大并发与最长上下文）。
  - **\$ / 1M tokens**、**单卡并发请求数**——直接对应部署成本。
  - 经验：prefill 是 **compute-bound**，decode 是 **memory-bandwidth-bound**，因此两阶段的瓶颈和优化手段（甚至 PD 分离部署）不同。

- **质量 / 稳定性**
  - **Goodput**：不是裸吞吐，而是「满足 SLO（如 TTFT<Xs、TPOT<Yms）的有效吞吐」，更贴近真实可用容量。
  - 准确率不应因量化/投机解码等优化而下降；还需关注 **超时率、错误率、抢占/排队时延**。

> 追问技巧：被问「服务慢了怎么排查」，按 **TTFT 高 → 看 prefill / 排队 / prefix cache 命中率**；**TPOT 高 → 看 batch、KV cache 显存是否打满、是否频繁抢占** 来分层定位。

📖 参考：GQA — https://arxiv.org/abs/2305.13245 ｜ vLLM/PagedAttention — https://arxiv.org/abs/2309.06180 ｜ DistServe（PD 分离）— https://arxiv.org/abs/2401.09670

---

### 6. MoE（混合专家）原理、优势与难点 ⭐ `#基础 #工程`
**【核心答案】** 用多个 FFN「专家」，由一个 router 网络为每个 token 选 top-k 个专家做稀疏激活。总参数量很大，但单次前向只激活一小部分，从而在固定算力下扩大模型容量。

**【深入】**
- 典型如 Mixtral 8×7B：8 个专家，每 token 选 top-2，激活参数约等于一个 13B 稠密模型，但总参数 47B。
- router 通常是一个线性层 + softmax，输出每个专家的权重，选 top-k 加权组合。
- **负载均衡损失**：防止 router 总是偏爱少数专家导致「专家塌缩」，强制 token 较均匀地分布到各专家。

**【权衡 / 追问】**
- 优点：相同推理算力下容量更大、效果更好。
- 难点：① 训练不稳定、对超参敏感；② **推理显存要装下全部专家**（省的是算力不是显存）；③ 负载不均、通信开销（专家分布在多卡时的 all-to-all）。
- 追问 **MoE 和模型并行的关系**：专家并行（expert parallelism）是分布式训练 MoE 的关键。

📖 参考：Mixtral of Experts — https://arxiv.org/abs/2401.04088

---

### 7. Tokenization：BPE 怎么工作？为什么数字/中文/代码容易出问题？ `#基础`
**【核心答案】** BPE（字节对编码）从字符（或字节）开始，反复把语料中出现频率最高的相邻 pair 合并成新 token，直到达到目标词表大小，得到一套子词词表，平衡词表规模与 OOV。

**【深入】**
- **BPE**：贪心按频率合并；**WordPiece**（BERT）：按合并后的语言模型似然增益来选；**SentencePiece**：直接在原始文本上训练、语言无关、能处理无空格语言；**byte-level BPE**（GPT-2）：在字节上做，理论上永不 OOV。
- 一个汉字常被切成多个 token（UTF-8 多字节），所以中文「字数 ≠ token 数」，中文文本的 token 消耗通常更高。
- 数字常被不规则切分（如 "12345" 可能切成 "123"+"45"），影响算术能力；代码的缩进、符号也会产生大量碎 token。

**【权衡 / 追问】**
- 追问 **词表大小的权衡**：词表大→序列短、embedding 矩阵大；词表小→序列长、计算多。
- 追问为什么 LLM 不擅长「数字母」「反转字符串」：因为它看到的是 token 不是字符。

---

# 第二部分：LLM 训练与对齐

### 8. 大模型训练分哪几个阶段？ `#对齐 #基础`
**【核心答案】** ① 预训练（海量无标注文本做 next-token prediction，学语言与世界知识）→ ② SFT 监督微调（用指令-回答对学会听指令）→ ③ 偏好对齐（RLHF / DPO，让输出更符合人类偏好：有用、无害、诚实 3H）。

**【深入】**
- 预训练消耗 99%+ 的算力，决定模型「知道什么」。
- SFT 改变的是「怎么表达/遵循格式」，数据量小但质量要求极高。
- 对齐阶段进一步精细调节行为边界与偏好。近年还常加 **持续预训练**（领域增强）和 **拒绝采样微调（RFT）**。

**【权衡 / 追问】**
- 追问「能力」主要来自哪个阶段？知识与核心能力来自预训练，SFT/对齐主要是「激发与对齐」，而非注入新知识——这也是「知识更新优先用 RAG 而非微调」的依据。

---

### 9. RLHF 完整流程，痛点是什么？⭐ `#对齐 #高频`
**【核心答案】** ① 用人类对多个回答的排序训练一个 **Reward Model（RM）**；② 用 **PPO** 优化策略模型最大化 reward，同时加 **KL 惩罚**约束它别偏离 SFT 模型太远。

**【深入】**
- 数据：让标注者对同一 prompt 的多个回答排序，RM 学习这个偏好打分。
- PPO 阶段同时存在 4 个模型：policy（在训）、reference（SFT 冻结，算 KL）、reward（打分）、critic/value（估计优势）。
- KL 惩罚是关键：没有它，policy 会为了刷高 reward 输出畸形文本（**reward hacking**）。

**【公式：RM 与 PPO】**
- **RM 损失**（Bradley-Terry 成对排序，y_w 优于 y_l）：

  $$\mathcal{L}_{RM} = -\,\mathbb{E}_{(x,y_w,y_l)}\big[\log \sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\big]$$

- **PPO 优化目标**（带 KL 惩罚的奖励 + clip 的策略梯度）：

  $$\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\Big[\,r_\phi(x,y) - \beta\,\mathrm{KL}\big(\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\big)\Big]$$

  实现上用 clip 形式（A 为 GAE 优势，r_t(θ)=π_θ/π_old 为重要性比）：

  $$\mathcal{L}_{PPO} = \mathbb{E}_t\big[\min(r_t(\theta)\,A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,A_t)\big]$$

**【权衡 / 追问】**
- 痛点：流程复杂、4 模型同时占显存、RL 训练不稳、对超参敏感、RM 容易被钻空子。
- 追问 **RLHF vs RLAIF**：RLAIF 用 AI（如更强模型/宪法）代替人类生成偏好标签，降本，Anthropic 的 Constitutional AI 是代表思路。
- 追问 **GRPO（DeepSeek 提出，DeepSeekMath/R1 采用）**：PPO 的简化，**去掉 critic/value 网络**，省一个与 policy 同规模的模型，显存与稳定性都更友好。做法是对同一 prompt 采样一组 G 个回答 {o_1..o_G}，用**组内归一化的 reward 当优势**：

  $$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$$

  再套 PPO 的 clip 目标，并把 KL 作为**独立正则项**直接加进 loss（而非塞进 reward）。优点：无需 critic、天然适配「可验证奖励 RLVR」（数学/代码用规则判对错）；局限：依赖组内采样多条、reward 方差大时不稳。

📖 参考：InstructGPT — https://arxiv.org/abs/2203.02155 ｜ GRPO/DeepSeekMath — https://arxiv.org/abs/2402.03300

---

### 10. DPO 相比 RLHF 的优势与局限 ⭐ `#对齐 #高频`
**【核心答案】** DPO 把 RLHF 的目标用数学推导成一个**直接在偏好数据上的分类损失**，不需要单独训练 reward model，也不需要 RL 采样循环，因此更稳定、更省资源、更易实现。

**【深入】**
- 关键洞察：RLHF 的最优策略与 reward 之间存在闭式关系，于是可以把「奖励最大化 + KL 约束」重参数化，直接用偏好对 (chosen, rejected) 做一个类似二分类的损失来优化策略，reward model 被「隐式」吸收进策略本身。
- 仍需要一个 reference 模型算 KL 项，但省掉了 RM 和 PPO 的复杂度。

**【公式：DPO】**
- 由 RLHF 最优解 $\pi^*(y|x) \propto \pi_{ref}(y|x)\exp\big(\tfrac{1}{\beta}r(x,y)\big)$ 反解出**隐式奖励** $r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$，代入 Bradley-Terry，配分项 Z(x) 在成对相减中抵消，得到：

  $$\mathcal{L}_{DPO} = -\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$

  直观理解：**抬高 chosen、压低 rejected 相对 reference 的对数概率比**，β 控制偏离 reference 的强度（等价于 RLHF 里的 KL 系数）。

**【权衡 / 追问】**
- 局限：① 是 **offline**（用固定偏好数据），不像 PPO 能在线采样，可能不如在线方法上限高；② 对偏好数据分布敏感、易过拟合。
- 追问 **衍生方法**：IPO（缓解过拟合）、KTO（用单条好/坏标签而非成对）、SimPO（去掉 reference 模型）、ORPO（把 SFT 和偏好对齐合一）。

**【三者对比：PPO vs DPO vs GRPO】**

| 维度 | PPO（RLHF） | DPO | GRPO |
|------|------------|-----|------|
| 范式 | Online RL | Offline，直接偏好优化 | Online RL（PPO 简化） |
| 需要 RM？ | ✅ 显式训练 | ❌ 隐式吸收进策略 | ✅（或规则/可验证奖励） |
| 需要 Critic/Value？ | ✅ | ❌ | ❌（用组内均值当 baseline） |
| 训练模型数 | 4（policy/ref/RM/critic） | 2（policy/ref） | 3（policy/ref/RM） |
| 优势估计 | GAE（critic 估计） | 无（直接分类损失） | 组内 reward 归一化 |
| 显存/复杂度 | 高 | 低 | 中 |
| 稳定性 | 对超参敏感、易不稳 | 稳定、好复现 | 比 PPO 稳，省 critic |
| 在线探索 | ✅ 上限高 | ❌ 受限于数据分布 | ✅ |
| 适用场景 | 通用对齐、追求上限 | 资源有限、快速对齐 | 推理/数学/代码（RLVR） |
| 代表 | InstructGPT/ChatGPT | Zephyr、众多开源对齐 | DeepSeekMath、DeepSeek-R1 |

> 一句话总结：**PPO 最通用但最重；DPO 用一个分类损失换掉整条 RL 流水线，简单稳定但是 offline；GRPO 介于两者之间——保留在线 RL 与探索，但用「组内相对优势」干掉 critic，特别适合有可验证奖励的推理任务。**

📖 参考：DPO — https://arxiv.org/abs/2305.18290 ｜ GRPO/DeepSeekMath — https://arxiv.org/abs/2402.03300

---

### 11. 为什么会幻觉（hallucination）？怎么缓解？⭐ `#对齐 #高频`
**【核心答案】** 根因是训练目标只优化「下一个 token 的流畅与高概率」，而非「事实正确」；当知识缺失/过时/长尾时，模型仍倾向流畅地编造而非承认不知道。

**【深入】**
- 来源细分：① 预训练数据本身有错/过时；② 长尾知识记不准；③ 解码随机性；④ SFT 阶段如果教模型回答它其实不知道的问题，会强化「自信地胡说」。
- 缓解分层：
  - **数据/训练**：提高数据质量；对齐时奖励「不确定就说不知道」与「引用来源」。
  - **推理时**：RAG 提供外部依据并溯源、降低温度、self-consistency（多次采样投票）、self-verification（让模型核查自己的答案）。
  - **系统层**：输出附带引用、置信度、可点击溯源。

**【权衡 / 追问】**
- 追问 RAG 能否消除幻觉？不能，只能降低——检索错/不全、或模型「无视」检索结果仍会幻觉。
- 追问怎么**评测**幻觉：TruthfulQA、FActScore、以及多模态的 POPE。

📖 参考：RAG — https://arxiv.org/abs/2005.11401

---

### 12. 灾难性遗忘怎么处理？ `#对齐`
**【核心答案】** 微调新任务/新领域时旧能力退化。常用：原始数据回放（replay/混合通用数据）、参数高效微调（LoRA/Adapter 只动少量参数）、较小学习率、正则化（如 EWC 约束重要参数）。

**【深入】**
- 全参微调最容易遗忘；只训练旁路（LoRA）天然保留原能力。
- 持续学习中常按比例混入通用语料（如 5–30%）来「锚定」原分布。

**【权衡 / 追问】** 追问数据配比怎么定？经验性，需实验；领域数据过多会遗忘通用能力，过少则领域提升不足。

---

### 13. LoRA 原理与优势？QLoRA 又是什么？⭐ `#工程 #高频`
**【核心答案】** 冻结原权重 W，旁路加一个低秩分解 ΔW = B·A（A 把维度降到秩 r，B 再升回去，r 远小于原维度），只训练 A、B。基于「权重更新本质是低秩的」这一假设。

**【深入】**
- 训练参数量从 d×d 降到 2×d×r，降几个数量级；显存和存储大幅下降；多个 LoRA 可即插即用切换。
- 推理时可把 B·A 合并回 W，**不增加推理延迟**。
- 初始化：A 用高斯、B 置零，保证训练开始时 ΔW=0 不破坏原模型。
- **QLoRA**：把基座权重量化到 4-bit（NF4）后再做 LoRA，单卡就能微调很大的模型，引入了 NF4、double quantization、paged optimizer 等技巧。

**【权衡 / 追问】**
- 追问 **秩 r 怎么选**：常见 8/16/32/64，越大表达力越强但越接近全参微调；任务越复杂用越大。
- 追问 LoRA 的局限：对需要大幅改变模型行为的任务，可能不如全参微调。

📖 参考：LoRA — https://arxiv.org/abs/2106.09685 ｜ QLoRA — https://arxiv.org/abs/2305.14314

---

# 第三部分：LLM 推理与工程优化

### 14. 量化是什么？PTQ 和 QAT 的区别？常见方案？ `#工程`
**【核心答案】** 把权重/激活从 FP16 降到 INT8/INT4 等低比特，省显存、提速。分训练后量化（PTQ，无需再训练，快）和量化感知训练（QAT，训练时模拟量化，精度好但成本高）。

**【深入】**
- **PTQ 主流**：GPTQ（逐层最小化量化误差，权重 4-bit）、AWQ（识别并保护「重要权重」对应的通道）、SmoothQuant（把激活的异常值平移到权重上，让激活更好量化）。
- 难点是**激活里的 outlier**：少数维度数值极大，直接量化会损失精度，上述方法都在处理它。
- 衍生概念：W4A16（权重 4-bit、激活 16-bit）、KV cache 量化。

**【权衡 / 追问】** 核心权衡是精度损失 vs 显存/速度收益。追问 INT4 一般损失多少？通常 4-bit 权重量化精度损失很小（1-2%），是性价比甜点。

📖 参考：GPTQ — https://arxiv.org/abs/2210.17323 ｜ AWQ — https://arxiv.org/abs/2306.00978

---

### 15. Flash Attention 到底解决了什么？⭐ `#工程 #高频`
**【核心答案】** 它**不减少计算量，而是减少显存读写**（IO-aware）。标准注意力要把 n×n 的注意力矩阵写回显存（HBM）再读出，IO 成 O(n²)；Flash Attention 用分块（tiling）在高速 SRAM 内计算、用 online softmax 增量更新，避免存下完整矩阵。

**【深入】**
- 关键观察：现代 GPU 计算快、访存慢，注意力是**访存受限**而非计算受限。
- online softmax：分块计算时维护running max 和 running sum，逐块更新结果，无需一次性看到整行。
- 结果：更快 + 更省显存，且是**精确注意力**（非近似，不损失精度）。
- 后续：FlashAttention-2（更好的并行与工作划分）、FlashAttention-3（针对 H100 的异步与 FP8）。

**【权衡 / 追问】** 追问它和稀疏/线性注意力的区别？后者通过近似降低计算复杂度（有损），Flash Attention 是无损的工程优化。

📖 参考：FlashAttention — https://arxiv.org/abs/2205.14135

---

### 16. 推测解码（Speculative Decoding）原理？⭐ `#工程`
**【核心答案】** 用一个**小模型（draft）快速自回归生成若干候选 token**，再用大模型**一次前向并行验证**这些 token，接受匹配的前缀、从第一个不匹配处重采。大模型一次能确认多个 token，提升吞吐。

**【深入】**
- 大模型对一段序列做一次前向就能拿到每个位置的概率分布，用它来「批量校验」小模型的猜测。
- 通过精心设计的接受/重采规则，保证最终输出分布与「只用大模型」**严格一致**（无损加速）。
- 变体：Medusa（给大模型加多个预测头自己做 draft，省去独立小模型）、EAGLE（在特征层做更准的 draft）、Lookahead Decoding。

**【权衡 / 追问】** 加速比取决于小模型的「命中率」和两者大小差距；draft 太弱则接受率低、收益小。追问适用场景：批量小、追求低延迟时收益明显。

📖 参考：Speculative Decoding — https://arxiv.org/abs/2211.17192

---

### 17. 提升推理吞吐的工程手段有哪些？ `#工程`
**【核心答案】** Continuous batching（动态拼接请求）、PagedAttention（KV cache 分页管理）、prefix caching（复用相同前缀的 KV）、张量/流水线并行、量化、推测解码。

**【深入】**
- **Continuous batching**：传统静态 batch 要等最慢的请求完成才能放新请求；continuous batching 在每步动态把已完成的换成新请求，大幅提高 GPU 利用率。
- **PagedAttention**（vLLM）：借鉴操作系统虚拟内存分页，把 KV cache 切成固定大小的块、用页表映射到非连续物理显存，几乎消除碎片，并支持请求间 KV 共享（如相同 system prompt）。
- **Prefix caching**：多个请求共享前缀（如同一 system prompt）时只算一次。

**【权衡 / 追问】** 追问 throughput vs latency 的权衡：大 batch 提吞吐但单请求延迟变高；在线服务要在两者间取舍。

📖 参考：vLLM / PagedAttention — https://arxiv.org/abs/2309.06180

---

### 18. Prefill 和 Decode 两个阶段有什么本质区别？ `#工程 #高频`
**【核心答案】** Prefill 处理输入 prompt，所有 token 可并行，是**计算密集（compute-bound）**；Decode 逐 token 生成，每步只算一个 token 对全部历史的注意力，是**访存密集（memory-bound）**，受 KV cache 读取带宽限制。

**【深入】**
- 这解释了为什么：长输入（prefill 重）和长输出（decode 步数多）的优化重点不同。
- 指标：prefill 影响「首 token 延迟 TTFT」，decode 影响「每 token 延迟 TPOT / 吞吐」。
- 工程上常把两者**分离调度**（如 disaggregated serving，prefill 和 decode 跑在不同节点）以分别优化。

**【权衡 / 追问】** 追问 chunked prefill：把长 prefill 切块，和 decode 混合调度，平衡 TTFT 与吞吐。

---

# 第六部分：高频开放题

> 这些没有标准答案，考的是判断力和「讲权衡」的能力。

- **上下文越长越好吗？** 不是。长上下文有「lost in the middle」（中段信息易被忽略）、推理成本随长度上升、注意力被稀释等问题。RAG 与长上下文互补而非替代：要溯源/省成本/知识常更新 → RAG；信息必须整体一起看（如长合同通读）→ 长上下文。

- **RAG 还是微调？** 知识频繁更新、需溯源、降低幻觉 → RAG；改变风格/格式/输出习惯、注入稳定的领域能力 → 微调；二者常结合（微调让模型更会用检索结果）。一句话：RAG 改「模型知道什么」，微调改「模型怎么表现」。

- **temperature / top-p 怎么调？** temperature 缩放 softmax 平滑度控制随机性（低→确定、高→发散）；top-p（核采样）只在累计概率达 p 的 token 集合里采样，截断长尾。事实/代码任务用低温，创意任务用高温。两者一般不同时大幅调。

- **Scaling Law 是什么？Chinchilla 的启示？** 模型性能随参数 N、数据 D、算力 C 呈幂律提升。Chinchilla 指出当时多数大模型「参数过大、数据不足」，在固定算力下应让 N 和 D 更均衡地增长（经验上 token 数约 20×参数量），同算力下小而「喂饱」的模型反而更强。

- **怎么把一个大模型压缩上线？** 组合拳：量化（4-bit 权重）+ 蒸馏（大模型教小模型）+ 剪枝 + 推理引擎优化（vLLM / TensorRT-LLM）。按延迟、成本、精度目标做取舍，先定 SLA 再选方案。

📖 参考：Scaling Laws — https://arxiv.org/abs/2001.08361 ｜ Chinchilla — https://arxiv.org/abs/2203.15556

---

# 第七部分：面试表达技巧

- **答题三段式**：先一句话给结论 → 再展开原理 → 最后讲权衡/局限。考官最在意「你是否理解 trade-off」。
- **遇到不会的**：诚实划定边界，再讲相邻你懂的部分，比硬编强。「这块我没深入做过，但据我理解……，相关的 X 我比较熟，可以展开。」
- **准备 1–2 个亲手做过的项目**：能讲清「为什么这么选、踩了什么坑、怎么解决、效果如何」——这是和「背题选手」拉开差距的地方。面试官最爱顺着项目深挖。
- **反问环节**备 1–2 题：团队的技术栈/当前最大技术挑战/这个岗位半年内要解决什么，显示你在认真思考是否合适。

---

# 附录 A：核心论文索引（LLM）

> 想系统学习时的精读清单。已逐一核对链接的标注 ✅。

**架构基础**
- ✅ Attention Is All You Need — https://arxiv.org/abs/1706.03762
- ✅ BERT — https://arxiv.org/abs/1810.04805
- ✅ GPT-3 (Few-Shot Learners) — https://arxiv.org/abs/2005.14165
- ✅ RoFormer / RoPE — https://arxiv.org/abs/2104.09864
- ✅ GQA — https://arxiv.org/abs/2305.13245
- ✅ Mixtral (MoE) — https://arxiv.org/abs/2401.04088

**训练与对齐**
- ✅ Scaling Laws (Kaplan) — https://arxiv.org/abs/2001.08361
- ✅ Chinchilla (Hoffmann) — https://arxiv.org/abs/2203.15556
- ✅ InstructGPT / RLHF — https://arxiv.org/abs/2203.02155
- ✅ DPO — https://arxiv.org/abs/2305.18290
- ✅ LoRA — https://arxiv.org/abs/2106.09685
- ✅ QLoRA — https://arxiv.org/abs/2305.14314
- ✅ Chain-of-Thought — https://arxiv.org/abs/2201.11903

**推理与工程**
- ✅ FlashAttention — https://arxiv.org/abs/2205.14135
- ✅ vLLM / PagedAttention — https://arxiv.org/abs/2309.06180
- ✅ Speculative Decoding — https://arxiv.org/abs/2211.17192
- ✅ GPTQ — https://arxiv.org/abs/2210.17323
- ✅ AWQ — https://arxiv.org/abs/2306.00978

---

# 附录 B：更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-04 | v2.0 | 33 题全部扩写为「核心答案/深入/权衡/参考」四段式；新增核心论文索引（链接已核对）；新增维护规范与标签体系 |
| 2026-06-04 | v1.0 | 初版，33 题速记要点 |

> 后续每次补充，请在此追加一行，并在对应题目下更新内容。

---
# 附录 C：待补充清单（LLM）

- [ ] State Space Models / Mamba（Transformer 的替代架构）
- [ ] 长上下文专题（YaRN、位置插值、Ring Attention）
- [ ] 推理模型 / test-time compute（o1 式长思考、强化学习提升推理）
- [ ] 评测体系（各 benchmark 的细节与局限）
