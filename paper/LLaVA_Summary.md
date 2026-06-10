# LLaVA: Visual Instruction Tuning

> **Tech Sharing Draft** · Paper: *Visual Instruction Tuning* — Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (UW-Madison & Microsoft), Apr 2023 · **NeurIPS 2023 Oral**
> arXiv: [2304.08485](https://arxiv.org/abs/2304.08485) · [Project](https://llava-vl.github.io/) · [Code](https://github.com/haotian-liu/LLaVA)
>
> *Audience note: self-contained. 🎤 marks speaker cues. **Track note:** LLaDA/Cola DLM are about **generating** sequences with diffusion; LLaVA is about **understanding** images with an LLM — a different branch of multimodal. See §8.*

---

## 1. TL;DR

LLaVA (**L**arge **L**anguage **a**nd **V**ision **A**ssistant) is the paper that **kicked off the open-source "see-and-chat" VLM era**. Two ideas, both simple:

1. **A data trick:** use **language-only GPT-4** to *manufacture* multimodal instruction-following data — feed it an image's **captions + bounding boxes** (text!), and have it write rich Q&A about the image it never saw.
2. **A minimal architecture:** **CLIP ViT** (frozen vision encoder) → **one linear projection** → **Vicuna** (LLM). Train in **two stages**. That's it.

Result: an assistant that **chats about images**, scoring **85.1% relative to GPT-4** on its own benchmark and **92.53%** on ScienceQA.

🎤 *One-liner:* **"Take instruction tuning — the thing that turned GPT-3 into ChatGPT — and do it for vision. The hard part was the data, and they got GPT-4 to write it."**

---

## 2. Background — Why This Paper Mattered

- **Instruction tuning** is what made LLMs *helpful*: fine-tune on `(instruction, response)` pairs so the model follows arbitrary user intent, not just continues text.
- In **2023, this was unexplored for multimodal.** Vision-language models existed (CLIP, BLIP-2, Flamingo) but they were classifiers/captioners, not **conversational assistants** that follow free-form instructions about an image.
- **The blocker was data:** there was no large corpus of *"image + open-ended instruction + good answer."* Human annotation is slow and expensive.

🎤 *The gap in one sentence:* **"We knew how to make LLMs follow instructions; nobody had the data to make a *vision* model do the same."**

---

## 3. The Core Contribution: a GPT-4 Data Engine

The cleverest part of the paper — and the most reusable idea.

**Problem:** GPT-4 (in 2023) was **text-only**; it couldn't see images.
**Trick:** represent the image **symbolically as text** and let GPT-4 reason over that:
- **Captions** — describe the scene in sentences.
- **Bounding boxes** — `(object, coordinates)` lists giving spatial layout.

Feed both to language-only GPT-4 and prompt it to produce **three types** of instruction data:

1. **Conversation** — multi-turn Q&A about objects, counts, actions, locations.
2. **Detailed description** — a thorough paragraph describing the whole image.
3. **Complex reasoning** — multi-step inferences ("why might this be dangerous?").

→ **158K** image-instruction samples, generated cheaply, no human labeling.

🎤 *Why it's a big deal:* this is **model-generated supervision** (a self-instruct / distillation idea applied to multimodal). It's the template the whole field copied.

---

## 4. Architecture — Deliberately Minimal

```
   image ──► CLIP ViT-L/14 ──► visual features  Z_v
                                      │
                                      ▼
                            Linear projection  W      ← the ONLY new module in stage 1
                                      │
                                      ▼
                          "visual tokens"  H_v  (in word-embedding space)
                                      │
   text instruction ─► tokenizer ─► H_q
                                      │
                       [ H_v ; H_q ] ─► Vicuna (LLM) ─► response
```

- **Vision encoder:** **CLIP ViT-L/14** (frozen) — extracts image features.
- **Connector:** **one linear layer `W`** — projects visual features into the **LLM's word-embedding space**, turning them into "visual tokens" the LLM treats like words.
- **LLM:** **Vicuna** (an instruction-tuned LLaMA).

🎤 *The whole bet:* **a single matrix `W` is enough to "translate" vision into the LLM's language** — no Q-Former, no cross-attention. Simplicity is the contribution. *(Your VLM doc Q20 contrasts this MLP-projector route vs BLIP-2's Q-Former vs Flamingo's cross-attention.)*

---

## 5. Two-Stage Training

**Stage 1 · Feature alignment pre-training.**
Freeze **both** the vision encoder **and** the LLM; train **only the projection `W`** on image–caption pairs. Goal: teach `W` to map visual features into the LLM's semantic space — i.e., learn the "translation," nothing else.

**Stage 2 · End-to-end instruction tuning.**
Unfreeze the **projector + LLM** (vision encoder stays frozen); train on the **158K GPT-4-generated instruction data**. Now it becomes a conversational assistant.

🎤 *Why freeze the LLM in stage 1?* A small amount of alignment data could damage the LLM's hard-won language ability. **First build the bridge, then fine-tune jointly.** *(= your VLM doc Q21.)*

---

## 6. Results

- **LLaVA-Bench** (their GPT-4-judged benchmark of conversation / description / reasoning): **85.1% relative score vs GPT-4.**
- **ScienceQA:** **90.92%** alone; **92.53%** when ensembled with GPT-4 — **SOTA at the time.**
- Qualitatively: coherent multimodal dialogue, OCR-ish reading, meme/joke explanation, reasoning about unusual scenes — emergent behaviors not explicitly trained.

🎤 *Headline:* **a 2-stage recipe on a frozen CLIP + Vicuna + one linear layer gets you 85% of GPT-4-level multimodal chat — open-source.**

---

## 7. Limitations

- **Single low-res image** (224/336px) → weak at **fine OCR, dense documents, small objects, counting**. *(Fixed later by LLaVA-1.5 / LLaVA-NeXT with higher res + tiling.)*
- **Object hallucination** — strong language prior can "describe" things not in the image.
- **Linear projector** is cheap but limited; LLaVA-1.5 upgraded it to an **MLP**.
- Data quality bounded by the GPT-4 generator and the caption/bbox annotations it was fed.
- No video / no multi-image in the original.

---

## 8. Where LLaVA Sits (vs your other two papers)

Your three papers form a nice map of multimodal + generative-LM space:

| | **LLaVA** | **LLaDA** | **Cola DLM** |
|---|---|---|---|
| Goal | **Understand** images, chat | **Generate** text | **Generate** text |
| Mechanism | CLIP→projector→LLM (AR decode) | **Discrete masked diffusion** | **Continuous latent diffusion** |
| Modality | Vision **in**, text out | Text only | Text only (→ multimodal future) |
| Key idea | **GPT-4-generated instruction data** + minimal projector | Random-ratio masking = generative | Plan in latent, then decode |
| Track | **Understanding-VLM** | **Generative diffusion-LM** | **Generative diffusion-LM** |

🎤 *Framing for the room:* **LLaVA = "how to make an LLM see." LLaDA/Cola = "how to make generation non-autoregressive."** Different axes — but Cola's endgame (unified continuous multimodal) is where these two tracks eventually meet.

---

## 9. Significance / Legacy

- Defined the **dominant open VLM recipe**: *frozen visual encoder → projector → LLM, two-stage train.*
- Spawned **LLaVA-1.5, LLaVA-NeXT**, and influenced Qwen-VL / InternVL lineage.
- Proved **synthetic, model-generated instruction data** is good enough to bootstrap multimodal assistants.

*(All of this maps to your [VLM知识总结.md](../VLM知识总结.md) Q20–Q21 — this Summary is the deep-dive behind those two questions.)*

---

## 10. Suggested Talk Flow (≈20 min)

1. Hook: "GPT-4 couldn't see images in 2023 — so how did they use it to teach a model to see?" (2 min)
2. Background: instruction tuning + the data gap §2 (3 min)
3. The data engine: caption+bbox → GPT-4 → 3 data types §3 (5 min)
4. Architecture: CLIP + one linear layer + Vicuna §4 (4 min)
5. Two-stage training + why freeze §5 (3 min)
6. Results + limitations §6–7 (3 min)
7. Legacy + where it sits §8–9 + Q&A

**Three things to remember:**
1. **The contribution is the *data engine*** (text-only GPT-4 writing multimodal instructions), not the architecture.
2. **One linear projection** translates CLIP features into the LLM's token space.
3. **Two stages:** align the projector first, then instruction-tune end-to-end.

---

## 11. References

- **LLaVA** — Visual Instruction Tuning — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) · [Project](https://llava-vl.github.io/) · [Code](https://github.com/haotian-liu/LLaVA)
- Follow-ups: **LLaVA-1.5** [2310.03744] · **LLaVA-NeXT** (blog)
- Contemporaries (your VLM doc Q20): **CLIP** [2103.00020] · **BLIP-2** [2301.12597] · **Flamingo** [2204.14198]
- Your notes: [VLM知识总结.md](../VLM知识总结.md) Q19–Q24

---

*Draft v0 — generated from the paper abstract/method + existing VLM notes. Next: add the data-generation prompt example, the LLaVA-Bench score table, and a real conversation trace for slides.*
