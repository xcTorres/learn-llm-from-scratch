# CLIP: Learning Transferable Visual Models From Natural Language Supervision

> **Tech Sharing Draft** · Paper: *Learning Transferable Visual Models From Natural Language Supervision* — Alec Radford, Jong Wook Kim, et al. (OpenAI), Feb 2021 · **ICML 2021**
> arXiv: [2103.00020](https://arxiv.org/abs/2103.00020) · [Blog](https://openai.com/research/clip) · [Code](https://github.com/openai/CLIP)
>
> *Audience note: self-contained. 🎤 marks speaker cues. **Why this is the root:** CLIP is the **vision encoder that LLaVA and most modern VLMs are built on.** Understand CLIP and the whole "understanding-VLM" track makes sense. See §8.*

---

## 1. TL;DR

CLIP (**C**ontrastive **L**anguage–**I**mage **P**re-training) learns vision **from raw text on the internet** instead of fixed labels. It trains **two encoders** — one for images, one for text — to agree: the embedding of an image and the embedding of its caption should be **close**, mismatched pairs **far apart**.

The payoff: a single pretrained model does **zero-shot classification** on tasks it was never trained for — just write the class names as text. It **matched a fully-supervised ResNet-50 on ImageNet with zero ImageNet training examples.**

🎤 *One-liner:* **"Stop predicting one of 1,000 fixed labels. Predict *which caption goes with this image* — and you get a model that recognizes almost anything you can describe in words."**

---

## 2. Background — The Problem with Old Vision Models

- Pre-CLIP vision models were trained on **fixed label sets** (ImageNet's 1,000 classes). Two costs:
  - **Expensive supervision** — every image hand-labeled into a closed vocabulary.
  - **Brittle/closed** — the model only knows those N classes; a new task means new labels + retraining.
- Meanwhile NLP had shown that **learning from raw web text** (GPT, BERT) scales beautifully without manual labels.

**CLIP's bet:** do the same for vision — **supervise images with the natural-language text that already accompanies them online** (alt-text, captions). Free, open-vocabulary supervision at internet scale.

🎤 *Gap in one sentence:* **"Labels are a bottleneck; the internet already pairs images with text — use that as the supervision."**

- **Data:** **400 million (image, text) pairs** scraped from the web (the "WIT" dataset).

---

## 3. The Core Idea: Contrastive Learning

Naively, you could train the model to *generate* the caption of each image — but that's slow and over-specifies (exact wording doesn't matter). CLIP instead solves an easier **matching** task.

**Setup — a batch of N image–text pairs:**
1. Image encoder embeds the N images → `I₁…I_N`.
2. Text encoder embeds the N captions → `T₁…T_N`.
3. Compute the **N×N cosine-similarity matrix** between every image and every text.
4. **The N correct pairs are on the diagonal (positives); the other N²−N are negatives.**
5. **Symmetric cross-entropy (InfoNCE):** for each row (image) pick its matching text, and for each column (text) pick its matching image. Maximize diagonal similarity, minimize off-diagonal.

```
         T₁   T₂   T₃  ...  T_N
   I₁  [ ✓  · · · ]      ← row softmax: image I₁ should pick text T₁
   I₂  [ ·  ✓ · · ]
   I₃  [ ·  · ✓ · ]      diagonal = positives
   ...                   everything off-diagonal = negatives
   I_N [ ·  · · ✓ ]
        ↑ column softmax: text T₁ should pick image I₁
```

A learned **temperature** scales the logits.

🎤 *Why contrastive, not generative (a likely question):* contrastive matching lets you learn from **massive, noisy web pairs** with **no fixed class set** — far more compute-efficient than generating exact captions, and it's exactly what yields the open-vocabulary, zero-shot power.

---

## 4. Architecture

- **Image encoder:** a **ResNet** (modified) *or* a **Vision Transformer (ViT)** — ViT-L/14 is the strong, widely-reused variant.
- **Text encoder:** a **Transformer** (GPT-style) reading the caption.
- Each maps to a shared embedding space; train with the contrastive loss above. Two separate towers, no cross-attention between them — a **dual-encoder** design.

🎤 *Note for the room:* **ViT-L/14 is literally the vision encoder LLaVA freezes and reuses.** This is where the two papers connect.

---

## 5. Zero-Shot Transfer — the Magic Trick

After pretraining, you classify **with no extra training**:

1. Take the class names of a new task, wrap each in a prompt: `"a photo of a {class}"`.
2. Encode all these prompts with the **text** encoder → a set of "class vectors."
3. Encode the image with the **image** encoder.
4. **Pick the class whose text vector is most similar** to the image.

That's it — the classifier is *built from words on the fly*. Change the task by changing the text, no retraining.

🎤 *The headline result:* **zero-shot CLIP matched a supervised ResNet-50 on ImageNet** — and generalized far better to distribution shift (ImageNet-R, -Sketch, -A), where standard models crumble.

- **Prompt engineering matters:** `"a photo of a {label}"` beats the bare label; ensembling multiple prompt templates adds a few more points (an early hint of prompt sensitivity).

---

## 6. Results & Findings

- Evaluated zero-shot across **30+ datasets**; competitive with fully-supervised baselines on many.
- **Robustness to distribution shift** is the standout — natural-language supervision learns more transferable features than fixed-label training.
- **Scales** smoothly with compute/data/model size.

---

## 7. Limitations

- **Not generative** — CLIP *matches* image↔text; it **cannot produce** text answers. (That's why VLMs bolt a CLIP encoder onto an LLM — see §8.)
- Weak at **fine-grained** distinctions, **counting**, **spatial/relational** reasoning, abstract/systematic tasks.
- **Prompt-sensitive** and inherits **web-scale social biases** from uncurated data.
- Zero-shot still **lags specialized supervised models** on hard fine-grained tasks.

---

## 8. Where CLIP Sits (vs your other papers)

CLIP is the **foundation of the understanding track** — the others build on it or contrast with it:

| | **CLIP** | **LLaVA** | **LLaDA / Cola DLM** |
|---|---|---|---|
| Job | **Align** image ↔ text | **Understand** image, chat | **Generate** text |
| How | **Contrastive** dual-encoder | CLIP→projector→LLM | **Diffusion** (discrete / latent) |
| Output | Embeddings (match score) | Free-form text answer | Free-form text |
| Generative? | **No** (representation) | Yes (AR decode) | Yes (non-AR diffusion) |
| Relation | **Provides the vision encoder** ▶ | **Uses CLIP ViT** as its eyes | Separate (generation track) |

🎤 *Narrative for the room:* **CLIP gives a model *eyes* (alignment). LLaVA gives those eyes a *mouth* (an LLM to talk). LLaDA/Cola rethink the *mouth itself* (diffusion instead of autoregression).** CLIP is step one of the whole story.

---

## 9. Significance / Legacy

- Launched **open-vocabulary, zero-shot** vision; killed the "fixed label set" assumption.
- Its **image encoder became the default visual backbone** for the entire VLM era (LLaVA, BLIP-2, Flamingo, Qwen-VL, InternVL…).
- The contrastive image-text objective underpins text-to-image generation too (Stable Diffusion's text conditioning uses a CLIP text encoder).

*(Maps to your [VLM知识总结.md](../VLM知识总结.md) Q19 — this Summary is the deep-dive behind that question.)*

---

## 10. Suggested Talk Flow (≈18 min)

1. Hook: "Classify any image into any category you can *describe* — with zero training examples." (2 min)
2. Background: the fixed-label bottleneck §2 (3 min)
3. Core idea: the N×N matrix + contrastive loss §3 (5 min)
4. Zero-shot trick: classifier-from-words §5 (4 min)
5. Results: ImageNet zero-shot + robustness §6 (2 min)
6. Limitations + where it sits §7–8 + Q&A (2 min)

**Three things to remember:**
1. **Supervision = web captions**, not labels → open-vocabulary.
2. **Contrastive matching** on an N×N similarity matrix (diagonal = positives).
3. **Zero-shot via "a photo of a {class}"** — build the classifier from text on the fly.

---

## 11. References

- **CLIP** — Learning Transferable Visual Models From Natural Language Supervision — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) · [Blog](https://openai.com/research/clip) · [Code](https://github.com/openai/CLIP)
- Builds toward: **LLaVA** [2304.08485] (uses CLIP ViT) · **BLIP-2** [2301.12597] · **Flamingo** [2204.14198]
- Related lineage: **ALIGN** (Google, noisy web pairs) · **SigLIP** (sigmoid loss, common modern replacement)
- Your notes: [VLM知识总结.md](../VLM知识总结.md) Q19

---

*Draft v0 — generated from the paper + existing VLM notes. Next: add the contrastive-loss pseudocode (the famous ~10-line snippet from the paper), the ImageNet zero-shot table, and the prompt-ensembling numbers for slides.*
