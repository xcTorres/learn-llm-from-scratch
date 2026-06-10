# LLaDA: Large Language Diffusion Models

> **Tech Sharing Draft** · Paper: *Large Language Diffusion Models* — Shen Nie, Fengqi Zhu, … Chongxuan Li (Renmin University of China & Ant Group), Feb 2025
> arXiv: [2502.09992](https://arxiv.org/abs/2502.09992) · [Project/demo](https://ml-gsai.github.io/LLaDA-demo/) · local PDF: [pdf/2502.09992v3.pdf](pdf/2502.09992v3.pdf)
>
> *Audience note: self-contained; Section 2 is a short primer. 🎤 marks speaker cues. Cola DLM uses LLaDA as its main baseline — see §11 for the side-by-side.*

---

## 1. TL;DR

LLaDA (**La**rge **La**nguage **D**iffusion with m**A**sking) asks a provocative question: **do LLM capabilities actually require left-to-right, token-by-token generation — or just good generative modeling?** It answers by training an **8B diffusion language model entirely from scratch** that **rivals LLaMA3 8B**.

The mechanism is deceptively simple — it's **masked prediction with a *random* mask ratio**:

- **Forward:** randomly mask tokens, with the mask ratio `t ~ Uniform[0,1]` (not BERT's fixed 15%).
- **Reverse (generation):** start from a **fully masked** sequence and **iteratively unmask** it over many steps, a Transformer predicting all masked tokens at once each step.

**Why it matters:**
1. First credible demonstration that **diffusion ≈ autoregressive at 8B scale** on standard benchmarks.
2. **Breaks the "reversal curse"** — bidirectional by construction, so it reasons backward almost as well as forward (where GPT-4o collapses).
3. Establishes the **discrete masked-diffusion** paradigm that later work (incl. **Cola DLM**) builds on or contrasts against.

🎤 *One-liner:* **"It's BERT-style masking turned into a real generative model — by making the mask ratio random."**

---

## 2. Background — Masked Diffusion in 5 Minutes

*Shared vocabulary for the room.*

- **Autoregressive LM (GPT/Llama):** factorize `p(x) = ∏ p(xᵢ | x_<ᵢ)` — predict next token left→right. Powerful, but **fixed order**, **sequential** (slow), and structurally biased.
- **BERT masked LM:** mask ~15% of tokens, predict them. Great for *representations*, but it's **not a generative model** — a fixed ratio can't define a full data distribution you can sample from.
- **LLaDA's insight:** if you let the **mask ratio vary across the whole range `[0,1]`**, masked prediction becomes a **principled generative process** — equivalent to a *discrete diffusion* whose forward process is "progressive masking" and whose reverse process is "progressive unmasking." The training loss then **upper-bounds the true negative log-likelihood** (a proper ELBO-style bound).

🎤 *Bridge line:* **"Fixed-ratio masking = BERT (representations). Random-ratio masking = a generative diffusion model (LLaDA)."**

### What AR limitation is it attacking?

| ARM weakness | Consequence |
|---|---|
| Strict left→right order | **Reversal curse**: knows "A→B" but fails "B→A" |
| Sequential decoding | No parallelism in token order; planning is implicit |
| Likelihood tied to one factorization | Capability assumed to *require* the AR form |

**LLaDA's thesis:** scalability, in-context learning, instruction-following come from **generative modeling principles** (maximize likelihood / minimize KL), **not** from the autoregressive formulation specifically.

---

## 3. The Method

### 3.1 Forward process — progressive masking
For `t ∈ (0,1)`, each token is **independently** masked with probability `t`, kept with probability `1−t`. At `t=1` everything is `[MASK]`; at `t=0` it's the clean text. During training, **`t` is sampled uniformly from `[0,1]`** — this is the one change that makes it generative rather than BERT.

### 3.2 Mask predictor & training objective
A **vanilla Transformer (no causal mask, full bidirectional attention)** takes the partially masked `x_t` and predicts **all masked tokens simultaneously**. Loss (only over masked positions, reweighted by `1/t`):

```
L(θ) = −E_{t, x₀, x_t} [ (1/t) · Σᵢ 1[x_t^i = MASK] · log p_θ(x₀^i | x_t) ]
```

This is a **likelihood lower bound**: `−E_data[log p_θ(x₀)] ≤ L(θ)` → a principled objective, unlike BERT's fixed-ratio loss.

🎤 *Key contrast to state on a slide:* **vs BERT** → random ratio + `1/t` reweighting makes it a valid generative model (Fisher-consistent), enabling in-context learning like an AR LLM.

### 3.3 Pretraining setup
- **8B parameters** (also a 1B variant), standard Transformer, **no causal masking**, vanilla MHA (not GQA), fixed length 4096 (1% sampled from `[1,4096]` for length robustness).
- **2.3 trillion tokens** (web + code + math + multilingual), **0.13M H800 GPU-hours**, AdamW, batch 1280.

🎤 *Punchline:* matches LLaMA3 8B **using only 2.3T tokens vs LLaMA3's 15T.**

### 3.4 Supervised fine-tuning (SFT)
- **4.5M prompt–response pairs.** Only **response tokens are masked**; the **prompt stays clean**. 3 epochs.
- EOS padding for equal lengths during training, removed at sampling. → gives instruction-following.

---

## 4. Inference — Generate by Unmasking

Start from a **fully masked** response and run the reverse process from `t=1 → t=0` over `N` steps:

1. Feed `prompt p₀` + current masked response `r_t` into the mask predictor.
2. Predict **all** masked tokens at once.
3. **Remask** a fraction `(s/t)` of them to stay aligned with the forward schedule, and repeat at the next (smaller) `t`.

**Remasking strategies (this is where quality comes from):**
- **Low-confidence remasking:** keep the high-confidence predictions, re-mask the least confident ones → commit to easy tokens first. (Analogous to confidence-ordered decoding.)
- **Semi-autoregressive remasking:** split the response into **blocks**, generate **block by block left-to-right**, run the diffusion reverse process **within** each block. Used for SFT models.

🎤 *Two knobs to mention:* **number of sampling steps** (quality↔speed trade-off) and **generation length** (a hyperparameter; results are insensitive thanks to variable-length training).

---

## 5. Headline Results

### 5.1 vs LLaMA (pretrained, Table 1)
| Task | LLaDA 8B | LLaMA3 8B | LLaMA2 7B |
|---|---|---|---|
| MMLU | **65.9** | 65.4 | 45.9 |
| GSM8K | **70.7** | 53.1 | 14.3 |
| C-Eval (中) | **70.5** | 51.7 | 34.0 |
| HumanEval | 33.5 | 34.2 | 12.8 |

→ **Competitive with LLaMA3 8B, far ahead of LLaMA2 7B**, especially strong on **math** — at a fraction of the tokens.

### 5.2 After SFT (no RL)
GSM8K **78.6** vs LLaMA3-Instruct 78.3; MMLU 65.5 vs 68.4; HumanEval 47.6 vs 59.8. Solid instruction-following from SFT alone.

### 5.3 Breaking the Reversal Curse (Table 3 — the showstopper)
496 Chinese poem pairs; predict the **next** line (forward) vs the **previous** line (reversal):

| Model | Forward | Reversal | Gap |
|---|---|---|---|
| GPT-4o | 82.7 | 34.3 | **−48.4** |
| Qwen2.5 7B | 75.9 | 38.0 | −37.9 |
| **LLaDA 8B** | 48.8 | **42.4** | **−6.4** |

🎤 *The slide that sells the talk:* AR models **collapse** on reversal; **LLaDA stays balanced** (only 6.4 pts) and **beats GPT-4o on the reversal direction** despite weaker forward — direct evidence that bidirectional generation removes a structural AR weakness.

### 5.4 Scaling
Scales smoothly to **~10²³ FLOPs**, tracking self-built ARM baselines across 6 tasks (and surpassing them on math). Counters the earlier "diffusion needs 16× compute" claim: that was measured on *likelihood*, an indirect metric; on downstream tasks the gap closes.

---

## 6. Limitations

- Comparisons capped at ~10²³ FLOPs; the ARM baseline isn't scaled identically.
- **No architecture specialization** — vanilla attention/position embeddings; no LLaDA-specific optimizations yet.
- **Inference is hyperparameter-sensitive** (steps, length, remasking, guidance).
- **SFT only, no RL** alignment.
- Still smaller than frontier models (GPT-4 / Gemini); **multimodal unexplored** (later addressed by LLaDA-V).
- Sampling is multi-step → **slower per sequence** than one AR pass unless distilled/accelerated.

---

## 7. Why It Matters

- **Existence proof:** diffusion LMs can **match AR at 8B** — capability comes from generative modeling, not the AR form.
- **Structural wins:** bidirectional → **no reversal curse**, parallel-friendly, plan-globally.
- **Foundation for a research line:** discrete masked diffusion → **LLaDA-V** (vision), and the baseline that **continuous** latent-diffusion LMs (**Cola DLM**) measure themselves against.

---

## 8. Cola DLM vs LLaDA (for your two-talk arc)

| | **LLaDA** | **Cola DLM** |
|---|---|---|
| Diffusion space | **Discrete tokens** (masking) | **Continuous latent** (via Text VAE) |
| Noise | Mask/categorical | Gaussian + Flow Matching |
| What's recovered | **Observation** (the tokens) | **Prior transport** in compressed space |
| Structure | Semi-AR by blocks (optional) | **Block-causal** by design |
| Hierarchy | Flat (token level) | **Hierarchical** (semantics vs wording) |
| Tooling reuse | Discrete-diffusion specific | **Image-diffusion tools transfer directly** |
| Scale shown | 8B from scratch, rivals LLaMA3 | ~2B matched study, beats AR/LLaDA on gen-quality |

🎤 *Narrative for the room:* **LLaDA proved discrete diffusion can rival AR; Cola DLM pushes diffusion into a *continuous, hierarchical latent* — trading token-level recovery for prior transport, and opening the door to unified multimodal generation.** They're two steps of the same story.

---

## 9. Suggested Talk Flow (≈25–30 min)

1. Hook: "Does an LLM *have* to write left-to-right?" → the reversal-curse table §5.3 (3 min)
2. Masked diffusion primer §2 — BERT vs LLaDA (random ratio) (5 min)
3. Method: forward masking, loss, why it's a real generative model §3 (6 min)
4. Inference by unmasking + remasking strategies §4 (4 min)
5. Results vs LLaMA3 + reversal curse §5 (5 min)
6. Limitations + why it matters §6–7 (3 min)
7. Bridge to Cola DLM §8 + Q&A (4 min)

**Three things to remember:**
1. **Random-ratio masking = generative diffusion** (the one idea that separates it from BERT).
2. **Generate by iteratively unmasking** a fully-masked sequence.
3. **Bidirectional → breaks the reversal curse**; matches LLaMA3 8B from scratch.

---

## 10. References

- **LLaDA** — Large Language Diffusion Models — [arXiv:2502.09992](https://arxiv.org/abs/2502.09992) · [HTML](https://arxiv.org/html/2502.09992v2) · demo: https://ml-gsai.github.io/LLaDA-demo/
- **LLaDA-V** (visual instruction tuning) — [arXiv:2505.16933](https://arxiv.org/abs/2505.16933)
- Related: **Cola DLM** [2605.06548] (continuous latent diffusion LM) · **SEDD** (score-entropy discrete diffusion) · survey *Discrete Diffusion in LLMs/MLLMs* [2506.13759]
- Background: DDPM [2006.11239] · Flow Matching [2210.02747]

---

*Draft v0 — generated from the paper (arXiv HTML) + demo page. Next: verify §5 numbers against the PDF tables, add the forward/reverse masking figure, and pull a real generation trace (sampling-stage visualization) for the slides.*
