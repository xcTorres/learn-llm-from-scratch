# Cola DLM: Continuous Latent Diffusion Language Model

> **Tech Sharing Draft** · Paper: *Continuous Latent Diffusion Language Model* — ByteDance Seed et al., May 2026
> arXiv: [2605.06548](https://arxiv.org/abs/2605.06548) · [Project page](https://hongcanguo.github.io/Cola-DLM/) · [Code](https://github.com/ByteDance-Seed/Cola-DLM) · License: Apache-2.0
>
> *Audience note: this draft assumes little diffusion background — Section 2 is a 5-minute primer. Speaker-cue callouts are marked with 🎤.*

---

## 1. TL;DR

Cola DLM generates text by **planning the whole passage in a continuous latent space first, then decoding it to words in (almost) one pass** — instead of predicting one token at a time, left to right.

It is a **hierarchical latent diffusion** model with three parts:

1. **Text VAE** — compresses text ⇄ continuous latent vectors (the "abstract sketch").
2. **Block-Causal DiT** — a Diffusion Transformer that *plans* by modeling the distribution over those latents (via Flow Matching).
3. **Conditional Decoder** — realizes the planned latents back into tokens.

**Why it's interesting:** it reframes language modeling as **prior transport in a compressed latent space** rather than token-level recovery. This (a) is non-autoregressive, (b) lets proven *image* diffusion tooling transfer directly to text, and (c) points toward a **unified discrete-text + continuous-modality** generative paradigm. At a matched ~2B scale it is competitive with / beats AR and discrete-diffusion (LLaDA) baselines on generation quality.

🎤 *One-liner for the room:* **"It separates *what to say* (global semantics) from *how to say it* (local wording)."**

---

## 2. Background — Diffusion in 5 Minutes (for the room)

*Skip/skim if the audience already knows diffusion. Keep this for shared vocabulary.*

- **Diffusion model (DDPM):** two processes. **Forward** gradually adds Gaussian noise to data until it's pure noise (fixed, no parameters). **Reverse** trains a network to *predict and remove* the noise step by step, so you can start from noise and denoise into a fresh sample. Core idea: *"learn to denoise."*
- **Latent diffusion (Stable Diffusion):** don't diffuse in raw pixel/token space — first compress with a **VAE** into a small **latent**, diffuse there, decode back. Massively cheaper. → *Cola does exactly this, but for text.*
- **DiT:** replace the U-Net denoiser with a **Transformer** over latent patches/tokens. Scales well. → *Cola's planner is a DiT.*
- **Flow Matching / Rectified Flow:** instead of DDPM's noise-prediction, learn a **velocity field** that transports noise → data along a near-straight path (an ODE). Fewer sampling steps. → *Cola trains its prior with Flow Matching.*

🎤 *Bridge line:* **"Everything here is image-diffusion machinery — Cola's contribution is making it work for discrete text."**

### The problem Cola is attacking

| Paradigm | Strength | Weakness |
|---|---|---|
| **Autoregressive (GPT/Llama)** | Excellent LM, simple | Fixed **left→right** order; sequential/slow; no explicit global "plan"; weak at non-monotonic tasks |
| **Discrete diffusion (LLaDA, SEDD)** | Parallel, non-AR | Operates on **discrete tokens** with categorical noise; slow; struggles to capture **global semantic structure** |

**Gap → Cola's bet:** combine *non-autoregressive generation* + *continuous representation* + *explicit probabilistic latent-variable modeling* into one framework.

---

## 3. The Core Idea: Plan, then Write

Standard LLMs hold no explicit plan — token *N* just conditions on tokens *1…N-1*. Cola adds an **outer diffusion loop over the entire latent sequence**, so it organizes meaning **globally** before committing to **local** words.

```
                 ┌─────────────────────────────────────────────┐
   text x  ──►   │  Text VAE encoder  q_φ(z₀|x)   (train only)  │ ──►  latent z₀
                 └─────────────────────────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────┐
   noise ε  ──►  │  Block-Causal DiT  (the "planner")   │   plans z₀ by
                 │  learns prior p_ψ(z₀) via Flow Match │   transporting noise→latent
                 └──────────────────┬──────────────────┘
                                    │  ẑ₀
                 ┌──────────────────▼──────────────────┐
                 │  Conditional Decoder  p_θ(x|z₀)      │ ──►  generated text x̂
                 │  (the "realizer", one pass)          │
                 └─────────────────────────────────────┘
```

**Generative factorization:**  `p(x) = ∫ p_θ(x | z₀) · p_ψ(z₀) dz₀`
- `p_ψ(z₀)` — the **learned latent prior** (the planner / DiT).
- `p_θ(x | z₀)` — the **conditional decoder** (the realizer / VAE decoder).
- `q_φ(z₀ | x)` — the **encoder**, used **only at training time**, never at generation.

🎤 *Mental model:* a writer who **sketches the outline in their head** (latent prior), then **writes the sentences** (decoder) — rather than blurting word-by-word.

---

## 4. Architecture & the "Block-Causal" Twist

### 4.1 Text VAE — the compression layer
Learns a stable, bidirectional text ⇄ latent mapping. Both encoder and decoder are kept **strictly causal** to prevent information leakage and to enable **streaming** generation later.

### 4.2 Block-Causal DiT — the planner
The latent isn't one monolithic blob — it's **factorized into blocks** with a **causal structure across blocks, bidirectional within a block**:

```
p_ψ(z₀) = p_ψ(z₀⁽¹⁾) · ∏_{b=2}^{B} p_ψ(z₀⁽ᵇ⁾ | z₀⁽<ᵇ⁾)
```

- **Within a block:** full (bidirectional) attention → rich local planning.
- **Across blocks:** causal → block *b* only sees **clean history** `z₀⁽<ᵇ⁾` (stop-gradient) plus its own **noisy** state.

🎤 *Why this matters:* it's a **hybrid** — diffusion's parallel refinement *inside* a block, autoregression's left-to-right coherence *across* blocks. Best of both, and it enables **KV-caching** + streaming.

### 4.3 Conditional Decoder — the realizer
Recovers tokens from the planned latents in essentially one pass (`p_θ(x | z₀)`).

---

## 5. Training — Two Stages

**Stage 1 · Text VAE pretraining.** Establish a stable text↔latent space:
```
L_VAE = −E[log p_θ(x|z₀)]  +  β·KL(q_φ(z₀|x) ‖ p_base)  +  λ_mask·L_mask
```
reconstruction + KL regularization + BERT-style masking.

**Stage 2 · Joint prior learning (DiT + VAE).** The DiT learns the latent prior with **Flow Matching**, while the VAE stays trainable:
```
L_stage2 = λ_VAE·L_VAE  +  λ_fm·L_FM  +  λ_ref·KL(q_φ ‖ q_φ_ref)
L_FM = E[ ‖ v_ψ(z_t⁽ᵇ⁾, t; z₀⁽<ᵇ⁾) − u_t⁽ᵇ⁾(z₀, z₁) ‖² ]   ← velocity-field regression
```

**ELBO view (the conceptual payoff):**
```
E[L_ELBO] = E[log p_θ(x|z₀)]  −  I_q(X; Z₀)  −  KL(q̄_φ(z₀) ‖ p_ψ(z₀))
              └ reconstruction ┘  └ compression ┘  └ prior matching ┘
```
Training cleanly decomposes into **reconstruct well**, **compress (don't store everything)**, and **match the prior**.

🎤 *Note:* Flow Matching here is the **solver for prior transport**, not the model's definition. Keep that distinction — it explains the likelihood quirk in §9.

---

## 6. Inference — Three Steps

1. **Encode the prompt:** `z^pre ~ q_φ(z^pre | x_pre)` → clean prefix latents.
2. **Generate response blocks** via block-causal flow transport (with classifier-free guidance):
   `ẑ₀⁽ᵇ⁾ = Φ⁰←¹_ψ(ε⁽ᵇ⁾; z^pre, ẑ₀⁽<ᵇ⁾)` — block by block, each refining noise → latent.
3. **Decode to text:** `x̂_res ~ p_θ(x_res | z^pre, ẑ₀⁽¹:ᴮ⁾)` in one pass.

Engineering niceties in the released code: **no-padding inference** (variable-length, concatenated with shape metadata), **KV-caching** in both DiT and VAE decoder, an **OpenAI-compatible Chat Completions** HTTP service, and HuggingFace `PreTrainedModel` integration.

---

## 7. A Unifying Lens: Markov-Path Framework

The paper frames prior methods as different "paths" under one Markov view — a great slide for positioning:

| Method | What the diffusion/chain operates on |
|---|---|
| **AR** | Token-level chain factorization — *observation path* |
| **LLaDA** | Discrete corruption→recovery — still *observation* focused |
| **Plaid** | Continuous, token-aligned recovery — no explicit latent |
| **Cola DLM** | **Prior-transport path in a compressed latent space**, with explicit **hierarchical** decomposition |

🎤 *Takeaway:* everyone else recovers *observations*; Cola transports a *prior* in a *compressed* space. That's the conceptual novelty.

---

## 8. Experiments & Findings

**Setup:** all methods at **~2B params**, OLMo-2 tokenizer, matched training. VAE ≈ 500M, DiT ≈ 1.8B, max sequence length 512. Baselines: **AR (Llama-style)** and **LLaDA (discrete diffusion)**. Benchmarks: LAMBADA, MMLU, SIQA, SQuAD, Story Cloze, OBQA, RACE, HellaSwag (8 tasks).

The study is organized around 4 research questions:

- **RQ1 — Global semantic structure:** the optimal diffusion timestep shifts systematically with latent dimension, with empirical peaks matching theory → evidence of shared cross-dimensional semantic structure.
- **RQ2 — Latent-space design:** **joint evolution from a pretrained init** beats both fixed latent spaces and training from scratch. Good initialization is what makes structured latents emerge.
- **RQ3 — Diffusion-process ablations:** block size, noise schedule, number of denoising steps, and CFG scale — paper identifies the effective configuration ranges.
- **RQ4 — Scaling:** favorable scaling curves up to **~2000 EFLOPs** vs. matched baselines on generation-quality metrics.

**Reference accuracies (≈2000 EFLOPs, from the repo):** LAMBADA 50.8%, MMLU 19.3%, Story Cloze 30.8%, 8-task average ≈ 26.75%.

🎤 *Honesty caveat for the room:* these are **research-scale** (2B, 512 ctx) numbers — the point is **matched-budget comparison and scaling trend vs AR/LLaDA**, not SOTA leaderboard numbers. Frame it as "method validation," not "production model."

---

## 9. Limitations (the genuinely interesting part)

1. **Likelihood ≠ generation quality.** Estimated **perplexity stays poor while generation quality stays strong.** Because Flow Matching optimizes *velocity-field regression*, not gold-token log-density, **PPL is a misleading metric here.** → The paper argues **generation quality + scaling reflect capability better than likelihood.** *(Great discussion hook.)*
2. **First-block conditioning** needs special handling — clean conditioning beats partial repaint, because errors accumulate along the flow trajectory.
3. **VAE reconstruction robustness** bounds everything — if the decoder can't faithfully reconstruct, generation suffers; stable VAE pretraining is essential.
4. **Applicability boundary** is explicit: the advantage holds only when rate–distortion is low at small rates, model-approximation error decreases, and the inference gap stays controllable.

---

## 10. Why It Matters / Future

- **A principled non-AR alternative:** hierarchical *continuous latent prior* modeling vs. strict token-level LM.
- **Tooling transfer:** continuous Gaussian latents mean **DDPM schedulers, CFG, ODE solvers** from image diffusion drop in directly.
- **Toward unified multimodal generation:** because the latent space is continuous, the same framework offers a **natural bridge from discrete text to continuous modalities (vision, audio)** — one generative paradigm for everything. *(This is the big-picture closer.)*

---

## 11. Suggested Talk Flow (≈30–40 min)

1. Hook: "What if an LLM planned the whole answer before writing the first word?" (2 min)
2. Diffusion in 5 minutes — §2 primer for the room (5 min)
3. AR vs discrete-diffusion: the gap (3 min)
4. Core idea: plan-then-write + the factorization diagram §3 (5 min)
5. Architecture & block-causal trick §4 (6 min)
6. Training 2 stages + ELBO intuition §5 (5 min)
7. Inference §6 (3 min)
8. Results + the PPL-vs-quality twist §8–9 (5 min)
9. Why it matters / unified multimodal §10 + Q&A (5 min)

**Three things the audience must remember:**
1. **Plan in latent, then decode** (not token-by-token).
2. **Block-causal = diffusion within a block + AR across blocks.**
3. **Prior transport in compressed space**, not token recovery — and *likelihood is the wrong yardstick*.

---

## 12. Open Questions to Raise

- How much does the **VAE bottleneck** cap ultimate quality vs. a strong AR model at scale?
- Is the **PPL-vs-generation** gap a measurement artifact or a deeper property of latent diffusion LMs?
- Latency in practice: block-causal + multi-step flow per block — does it actually beat AR decoding wall-clock?
- Does the "unified multimodal" promise hold, or is text-via-continuous-latent a detour vs. native discrete-diffusion?

---

## 13. References

- **Cola DLM** — Continuous Latent Diffusion Language Model — [arXiv:2605.06548](https://arxiv.org/abs/2605.06548) · [HTML](https://arxiv.org/html/2605.06548) · local PDF: [pdf/2605.06548v1.pdf](pdf/2605.06548v1.pdf)
- **Code** — [github.com/ByteDance-Seed/Cola-DLM](https://github.com/ByteDance-Seed/Cola-DLM) (Apache-2.0) · Weights: `ByteDance-Seed/Cola-DLM` on HuggingFace
- **Project page** — https://hongcanguo.github.io/Cola-DLM/
- Background: DDPM [2006.11239] · Latent Diffusion [2112.10752] · DiT [2212.09748] · Flow Matching [2210.02747] · Classifier-Free Guidance [2207.12598]
- Baselines/related: **LLaDA** (discrete diffusion LM) · **Plaid** (continuous token-aligned) · **SEDD**

---

*Draft v0 — generated from the paper, project page, and source repo. Next: verify the §8 numbers against the PDF tables, add real figures from the paper, and tighten the math notation for slides.*
