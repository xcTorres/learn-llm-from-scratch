# DDPM: Denoising Diffusion Probabilistic Models

> **Tech Sharing Draft** · Paper: *Denoising Diffusion Probabilistic Models* — Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley), Jun 2020 · **NeurIPS 2020**
> arXiv: [2006.11239](https://arxiv.org/abs/2006.11239) · [Code](https://github.com/hojonathanho/diffusion)
>
> *Audience note: this is the **theoretical root** of every other paper in this folder. 🎤 marks speaker cues. **Read this first** — LLaDA, Cola DLM, Stable Diffusion all reuse its forward/reverse-process machinery. See §8.*

---

## 1. TL;DR

DDPM is the paper that made **diffusion models actually work** for high-quality image generation. The idea is almost suspiciously simple:

- **Forward process:** take a real image and **gradually add Gaussian noise** over many steps until it's pure noise. This is **fixed, has no parameters** — you don't learn it.
- **Reverse process:** train a neural network to **undo one step of noising** — i.e., **denoise**. Run it repeatedly from pure noise and you generate a fresh image.

The decisive trick: instead of predicting complicated distributions, the network just **predicts the noise that was added**, trained with a **plain mean-squared-error loss**. That one simplification is what made diffusion trainable and high-quality.

🎤 *One-liner:* **"Learn to remove a little noise. Do it 1,000 times starting from static, and an image appears."**

---

## 2. Background — Why Diffusion?

By 2020, generative models had known trade-offs:
- **GANs:** sharp images but **unstable training**, mode collapse (miss parts of the data).
- **VAEs:** stable but **blurry**.
- **Autoregressive (PixelCNN):** good likelihood but **slow**, pixel-by-pixel.

**DDPM's pitch:** a generative model that is **stable to train** (no adversarial game) with **strong sample quality and mode coverage** — by trading compute (many denoising steps) for quality and stability.

🎤 *Frame it:* **"GANs fight a discriminator and break; diffusion just does regression — predict the noise. No adversary, no collapse."**

---

## 3. The Forward Process (Fixed — No Learning)

Define a Markov chain that adds a small amount of Gaussian noise at each of `T` steps (e.g. `T=1000`), with a variance schedule `β₁…β_T`:

```
q(x_t | x_{t-1}) = N( x_t ;  √(1−β_t)·x_{t-1} ,  β_t·I )
```

**The key convenience:** because each step is Gaussian, you can **jump to any step `t` in closed form** — no need to simulate the chain. With `α_t = 1−β_t` and `ᾱ_t = ∏ α_s`:

```
x_t = √(ᾱ_t)·x₀  +  √(1−ᾱ_t)·ε ,     ε ~ N(0, I)
```

So `x_t` is just the original image scaled down, plus a known amount of Gaussian noise. At `t=T`, `ᾱ→0` → pure noise.

🎤 *Why this matters:* **training can sample any `(x₀, t)` directly** — pick a random image, a random timestep, a random noise, and you instantly have a training example. No slow rollouts.

---

## 4. The Reverse Process (This Is What You Learn)

Generation reverses the chain: start at `x_T ~ N(0,I)` and denoise step by step. Each reverse step is modeled as a Gaussian whose mean is predicted by a network `θ`:

```
p_θ(x_{t-1} | x_t) = N( x_{t-1} ;  μ_θ(x_t, t) ,  Σ_t )
```

A naive objective would be a variational bound (ELBO) matching these reverse Gaussians to the true posteriors — messy. **DDPM's contribution is to simplify it.**

### The noise-prediction reparameterization
Instead of predicting the mean `μ_θ` directly, **predict the noise `ε` that was added** with a network `ε_θ(x_t, t)`. Ho et al. show this is both equivalent and far more stable, collapsing the whole objective into:

```
L_simple = E_{x₀, t, ε} [ ‖ ε − ε_θ( √ᾱ_t·x₀ + √(1−ᾱ_t)·ε ,  t ) ‖² ]
```

🎤 *Say this slowly — it's the whole paper:* **"Take an image, noise it to a random level `t`, and ask the network: *what noise did I just add?* Train with MSE. That's it."** Predicting noise beats predicting the mean directly.

### Sampling
Given the trained `ε_θ`, denoise iteratively from `t=T → 1`:
```
x_{t-1} = (1/√α_t) · ( x_t − (β_t/√(1−ᾱ_t))·ε_θ(x_t,t) )  +  σ_t·z ,   z~N(0,I)
```
Repeat ~1000 times → a sample.

---

## 5. Architecture & the Connection to Score Matching

- **Backbone:** a **U-Net** (convolutional encoder-decoder with skip connections) + **self-attention** at some resolutions. The timestep `t` is injected via a **sinusoidal time embedding** so the same network knows "how noisy" the input is.
- **Theory bonus:** predicting the noise `ε_θ` is **equivalent to estimating the score** (gradient of log-density, `∇ₓ log p(x)`). This links DDPM to **score-based models / Langevin dynamics** — the two lines (Ho's DDPM and Song's score-SDE) are the same thing in different language. *(This score/ODE view is what later enables fast samplers and Flow Matching — see §8.)*

🎤 *Bridge for later papers:* **"Predicting noise = estimating the score = a velocity field. That equivalence is the door to DDIM, DPM-Solver, and Cola's Flow Matching."**

---

## 6. Results

- **SOTA image quality at the time:** FID **3.17** on unconditional **CIFAR-10** (beating many GANs), high-quality **256×256 LSUN** samples.
- **Stable training, good mode coverage** — no adversarial instability.

---

## 7. Limitations

- **Slow sampling** — the headline weakness. Generating one image needs **hundreds-to-thousands of sequential network passes** (vs. a GAN's single pass). *(This single problem spawned a whole research line: DDIM, DPM-Solver, consistency/LCM models.)*
- **Operates in pixel space** → expensive at high resolution. *(Fixed by Latent Diffusion / Stable Diffusion — diffuse in a compressed VAE latent.)*
- Likelihood (in bits/dim) was not SOTA; the win was **sample quality**.

---

## 8. Where DDPM Sits — the Root of the Whole Folder

Everything else here is "DDPM, modified":

| Paper | What it changes about DDPM |
|---|---|
| **DDPM** (this) | The base: Gaussian forward/reverse, noise-prediction MSE, U-Net |
| **Stable Diffusion** | Run the *same* diffusion in a **VAE latent space** (cheap) + text conditioning |
| **DiT** | Swap the **U-Net for a Transformer** denoiser |
| **Flow Matching** | Replace noise-prediction with a **straight-path velocity field** (fewer steps) |
| **LLaDA** | Same *spirit*, but on **discrete text** — "noise" = **masking**, reverse = unmasking |
| **Cola DLM** | DDPM-style diffusion in a **continuous *text* latent** (Text VAE) + Flow Matching |

🎤 *Narrative for the room:* **"Learn these two processes — forward noising, reverse denoising — and you understand the engine inside Stable Diffusion, DiT, LLaDA, and Cola. They just change *what space* you diffuse in and *what network* does the denoising."**

*(Maps to your [VLM知识总结.md](../VLM知识总结.md) Q25–Q26.)*

---

## 9. Suggested Talk Flow (≈18 min)

1. Hook: "How do you turn random static into a photo? Learn to remove a tiny bit of noise — repeat." (2 min)
2. Background: GAN/VAE pain → why diffusion §2 (3 min)
3. Forward process + the closed-form jump §3 (4 min)
4. Reverse process + **the noise-prediction trick / L_simple** §4 (5 min)
5. U-Net + score connection §5, results §6 (2 min)
6. Limitation (slow) + where it leads §7–8 + Q&A (2 min)

**Three things to remember:**
1. **Forward = fixed Gaussian noising** (closed-form `x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε`).
2. **Reverse = a network that predicts the noise**, trained with **MSE** (`L_simple`).
3. **Quality & stability for the price of slow, many-step sampling** — the trade-off everything after tries to fix.

---

## 10. References

- **DDPM** — Denoising Diffusion Probabilistic Models — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) · [Code](https://github.com/hojonathanho/diffusion)
- Predecessor: Sohl-Dickstein et al. 2015 (*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*) · Parallel line: Song & Ermon (*score-based / NCSN*), Song et al. (*score SDE*)
- Builds toward (this folder): **Stable Diffusion** [2112.10752] · **DiT** [2212.09748] · **Flow Matching** [2210.02747] · **LLaDA** [2502.09992] · **Cola DLM** [2605.06548]
- Faster sampling: **DDIM** [2010.02502] · **DPM-Solver** [2206.00927] · Consistency/LCM
- Your notes: [VLM知识总结.md](../VLM知识总结.md) Q25–Q28

---

*Draft v0 — generated from the paper + existing VLM notes. Next: add the forward/reverse chain figure, the L_simple derivation in clean LaTeX, and the CIFAR-10 FID table for slides.*
