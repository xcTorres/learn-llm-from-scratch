# AGENTS

## Purpose
Help AI coding agents work effectively in this repository by describing the main content, conventions, and current gaps.

## Repository overview
- This repo is a learning-focused collection of LLM, multimodal, and transformer examples.
- Primary code artifacts are under `source-code/transformer_translation/`.
- Training and inference are implemented as plain Python scripts, not as a packaged application.
- `fine-tune/` contains Jupyter notebooks for BERT, ViT, BLIP-2, and diffusion tasks.

## Key files and folders
- `README.md` — high-level project overview and references.
- `source-code/transformer_translation/README.md` — instructions for the transformer translation demo.
- `source-code/transformer_translation/train.py` — run training with `python train.py`.
- `source-code/transformer_translation/inference.py` — run inference with `python inference.py`.
- `source-code/transformer_translation/data.py` — dataset loading using Hugging Face datasets.
- `source-code/transformer_translation/transformer.py` — model definition and Transformer implementation.
- `fine-tune/` — experimental notebooks; treat as exploratory examples rather than production code.

## Agent guidance
- Prefer small, incremental documentation or code changes.
- Before editing, confirm the change is aligned with learning/educational intent.
- Use the existing `README.md` and `source-code/transformer_translation/README.md` as the primary source of truth.
- Avoid adding large scaffolding unless the user explicitly asks for environment setup or productionization.

## Environment and workflow
- There is no repository-level environment file, dependency manifest, or test harness.
- Assume Python and common ML libraries are required; infer exact dependencies from notebooks and the transformer example.
- For `source-code/transformer_translation/`, the documented workflow is:
  - `cd source-code/transformer_translation`
  - `python train.py`
  - `python inference.py`

## Known gap
- `README.md` references `source-code/graph/graphsage`, but that directory is not present in the repository listing. Verify whether this is an outdated reference or missing content.

## Best practices for agents
- Keep suggestions concise and directly relevant to the example-focused repo.
- Link to docs instead of copying them.
- If adding new instructions, place them in `AGENTS.md` rather than in a hidden dotfile.
