# learn-llm-from-scratch

Notes and paper deep-dives on LLM / VLM / multimodal generation.

## 📚 Knowledge Summaries (面试 / 复习)

Structured Q&A notes, each item as **核心答案 → 深入原理 → 权衡/追问 → 参考**:

- [LLM 知识总结](./LLM知识总结.md) — Transformer, attention, training, inference, alignment
- [VLM 知识总结](./VLM知识总结.md) — CLIP, LLaVA, diffusion, video understanding (multimodal)
- [Agent 知识总结](./Agent知识总结.md) — planning, tools, memory, multi-agent

## 📝 Paper Deep-Dives ([`paper/`](./paper))

Reading notes for each paper. PDFs in [`paper/pdf/`](./paper/pdf).

| Paper | Year | One-liner | Notes |
| ----- | ---- | --------- | ----- |
| CLIP — Learning Transferable Visual Models From Natural Language Supervision | 2021 | Contrastive web-scale image-text → zero-shot vision | [Notes](./paper/CLIP_Summary.md) · [arXiv](https://arxiv.org/abs/2103.00020) |
| LLaVA — Visual Instruction Tuning | 2023 | CLIP + 1 linear layer + Vicuna; GPT-4-generated instruction data | [Notes](./paper/LLaVA_Summary.md) · [arXiv](https://arxiv.org/abs/2304.08485) |
| DDPM — Denoising Diffusion Probabilistic Models | 2020 | Learn to denoise; forward noising / reverse denoising | [Notes](./paper/DDPM_Summary.md) · [arXiv](https://arxiv.org/abs/2006.11239) |
| LLaDA — Large Language Diffusion Models | 2025 | Random-ratio masking = generative; breaks the reversal curse | [Notes](./paper/LLaDA_Summary.md) · [arXiv](https://arxiv.org/abs/2502.09992) |
| Cola DLM — Continuous Latent Diffusion Language Model | 2026 | Plan in a latent space, then decode | [Notes](./paper/ColaDLM_Summary.md) · [arXiv](https://arxiv.org/abs/2605.06548) |
