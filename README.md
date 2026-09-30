# 🧬 LoRA-Dataset-Coach

> The 2026-grade Swiss army knife for preparing, validating and evaluating LoRA training datasets — for **image (FLUX.2, Qwen-Image, Z-Image, Chroma1, FLUX.1, SDXL, Pony, Illustrious, NoobAI...)** and **video (Wan 2.2, HunyuanVideo 1.5, LTX-2.x, MiniMax-H3)** models.

[![License](https://img.shields.io/badge/license-Source--Available-orange.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Status](https://img.shields.io/badge/status-A--grade-brightgreen.svg)

---

## What it does

LoRA-Dataset-Coach is a **complete pipeline** that takes a folder of images and walks you through the full LoRA training workflow:

1. **Analyze** every image individually (face detection, identity match, quality, expression, pose, aesthetic, AI-generation detection, anatomical artifacts)
2. **Score** the dataset globally with a grade (A/B/C/D/F) and a **per-target-family rating** (SDXL classic, SDXL anime, FLUX.1, 2025-26 image models, Wan 2.2, 2025-26 video models, legacy video)
3. **Suggest** what's missing ("generate 3 more profile shots", "vary expressions", "too many white shirts → overfit risk")
4. **Auto-caption** with WD14 tags, Florence-2 or the 2026 standard **JoyCaption Beta One**
5. **Clean** the dataset (move rejects, recover blurry, deduplicate)
6. **Generate** masked training masks via BriaRMBG (for OneTrainer masked loss)
7. **Export** a ready-to-train folder for **30 LoRA targets** (Kohya, ai-toolkit, musubi-tuner, diffusion-pipe, OneTrainer...)
8. **Evaluate** the finished LoRA post-training with R-FaceSim, Copycat Detector, Black Hole Ranking (MirrorMetrics-inspired)

The whole thing runs in a single Tkinter GUI with a live preview that scrolls through every photo as it's analyzed.

---

## Screenshots

> *(Screenshots coming soon. The interface uses Catppuccin Mocha dark theme.)*

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/akalavol/LoRA-Dataset-Coach.git
cd LoRA-Dataset-Coach

# 2. Install dependencies
#    (Python 3.11 or 3.12 required for tkinter + PIL.ImageTk)
pip install -r requirements.txt

# 3. Launch (Windows)
run.bat

# OR direct:
python manager.py
```

On first analysis run, the tool will auto-download the needed models from HuggingFace:
- **insightface antelopev2** (~250 MB, face embedding)
- **WD14-MOAT tagger** (~330 MB, booru tags) — only if you pick WD14 mode
- **Florence-2-base** (~540 MB, natural captions fallback) — only if you pick Florence mode
- **JoyCaption Beta One** (~4 GB INT4 / 8 GB BF16, modern captions) — only if you pick JoyCaption mode
- **sdxl-detector** (~350 MB, AI image detection)
- **BriaRMBG-1.4** (~176 MB, subject masks) — only if you generate masks

---

## Feature matrix

| Feature | Status | Notes |
|---|---|---|
| Face detection & identity sim | ✅ | InsightFace antelopev2 (512-D embeddings) |
| Reference photo identity check | ✅ | Cosine sim vs ground-truth ref |
| Quality metrics | ✅ | Laplacian sharpness, brightness, contrast, megapixels |
| Pose estimation | ✅ | Yaw via 5 keypoints, flags profile/back shots |
| Expression detection | ✅ | CLIP zero-shot (7 expressions) |
| Shot type classification | ✅ | face_only / both / body_only via face proportion |
| Duplicate detection | ✅ | dHash (hamming < 5) + face sim > 0.96 |
| WD14 tagging | ✅ | Standalone ONNX reuse of ComfyUI tagger |
| Florence-2 natural captions | ✅ | CPU/CUDA fallback |
| **JoyCaption Beta One** | ✅ | **2026 community standard** for persona LoRAs |
| **AI-generated detection** | ✅ | Organika/sdxl-detector ViT (99.6% accuracy) |
| **Anatomical artifact detection** | ✅ | HADM-light via WD14 tags + caption regex |
| **C2PA / EXIF / PNG metadata** | ✅ | Detects already-tagged AI images |
| **Aspect-ratio bucket analysis** | ✅ | Flags imbalanced datasets |
| **Diversity scoring (CLIP clustering)** | ✅ | Union-Find clusters, big-cluster detection |
| **Per-target family scoring** | ✅ | 5 grades (SDXL/Flux/Wan/...) per dataset |
| **Overfit attribute alerts** | ✅ | "white shirt @ 87% → overfit risk" |
| Analysis cache | ✅ | Skip already-analyzed images by mtime |
| Move rejects / recover blurry | ✅ | Auto-folder + SUPIR/UltraSharp README |
| Caption inline editor | ✅ | Edit WD14/Florence/JoyCaption in popup, saves to .txt + cache |
| **Subject masks (BriaRMBG)** | ✅ | OneTrainer masked training |
| Multi-target LoRA prep | ✅ | 30 targets, auto-crops, configs, launch scripts, READMEs |
| **Targeted prompt generator** | ✅ | Exports ComfyUI workflows to fill missing shot types |
| **Post-train LoRA evaluator** | ✅ | R-FaceSim, Copycat, Black Hole Ranking, ComfyUI auto-generation + A/B gain |
| PDF report export | ✅ | Landscape A4 with Catppuccin theme |
| GitHub auto-update | ✅ | Built-in updater checks releases |

---

## Supported LoRA training targets

The **🧬 Prepare LoRA** action exports a ready-to-train folder for any of these.
Commands and `arch` names for the 2025-2026 targets were checked against the
musubi-tuner / ai-toolkit source on **2026-09-28**.

### ✨ Image 2025-2026 (LLM text encoders → long natural captions)
- `flux2_dev_aitoolkit` — FLUX.2 [dev] 32B via ai-toolkit (`arch: flux2`)
- `flux2_klein_aitoolkit` — FLUX.2 [klein] 9B base via ai-toolkit (`arch: flux2_klein_9b`)
- `flux2_musubi` — FLUX.2 dev / klein-base via musubi-tuner (`.bat` with cache + train)
- `qwen_image_musubi` / `qwen_image_aitoolkit` — Qwen-Image
- `zimage_musubi` / `zimage_aitoolkit` — Z-Image (Base; Turbo needs the de-turbo adapter)
- `chroma_aitoolkit` — Chroma1 Base

### 🎬 Video 2025-2026
- `wan22_musubi` — Wan 2.2 A14B T2V, high + low noise experts in one run
- `wan22_aitoolkit` — Wan 2.2 A14B T2V, 24 GB recipe
- `hunyuan15_musubi` — HunyuanVideo 1.5 T2V
- `ltx2_aitoolkit` — LTX-2 / 2.3 / 2.5 (audio+video) via ai-toolkit
- `minimax_h3_musubi` — MiniMax-H3 image LoRA (experimental, de-distillation adapter, **check the license for your country**)

> A photo dataset teaches a video model **appearance**, not motion. For motion, train on clips.

### 📸 Photo / Realistic
- `sdxl_kohya`, `onetrainer_sdxl` — SDXL
- `flux_aitoolkit`, `flux_kohya` — FLUX.1 dev

### 🎨 Anime / Style (SDXL forks)
- `pony_kohya`, `illustrious_kohya`, `noobai_kohya` (auto quality prefixes)

### 🗄 Legacy (kept for compatibility)
`sd15_kohya`, `sd35_kohya`, `hunyuan_dit_kohya`, `sana_diffpipe`, `wan21_musubi`,
`hunyuan_diffpipe` (HunyuanVideo 1), `ltx_video_diffpipe` (LTX 0.9), `cogvideox_diffpipe`,
`mochi_diffpipe`, `open_sora_diffpipe`

### ❌ Not exportable from a photo dataset (shown in the GUI)
- **FramePack / FramePack-F1** — musubi-tuner trains it on videos only (one-frame mode needs start→target image pairs)
- **Wan 2.5 / 2.6 / 2.7**, **MiniMax Hailuo** — API only, no weights
- **HunyuanImage 3.0** — open weights but no LoRA trainer supports it yet

For each target, the tool generates the correct:
- **Crop strategy** (square_face for SDXL, multi-bucket for Flux/Wan)
- **Resolution(s)**
- **Captioner choice** (WD14 for SDXL, JoyCaption natural for Flux/Wan)
- **Quality prefix** (Pony/Illustrious/NoobAI need specific tags)
- **Trainer config file** (.toml for Kohya/diffusion-pipe, .yaml for ai-toolkit)
- **README with launch instructions**

---

## The post-training evaluator (📊 Evaluate LoRA)

Once your LoRA is trained, generate ~30 test images, then provide:
1. **Generated folder** — your test outputs
2. **Reference folder** — real photos of the subject (≠ training photos)
3. **Training folder** (optional) — for copycat detection

The tool computes the **2026 community standard** metrics:

| Metric | What it tells you |
|---|---|
| **R-FaceSim** | Mean cosine similarity of generations vs real photos. >0.7 = identity learned. |
| **Copycat ratio** | % of generations >0.95 similar to a training image. >0 = LoRA is memorizing not generalizing. |
| **Black Hole Ranking** | Which training image attracts the most generations. >40% to a single image = mode collapse. |
| **Mode collapse signal** | Std deviation of R-FaceSim. <0.03 = single mode learned. |

Final verdict: **A** (excellent) to **F** (failed) with specific advice.

### Automatic test set with your local ComfyUI
Instead of generating the ~30 test images by hand, open **🎨 Generate test images with
ComfyUI** in the Evaluate tab:
1. In ComfyUI, export your usual workflow (the one for your model — FLUX.2, Qwen-Image, Wan…)
   with **Workflow → Export (API)**. It needs a `LoraLoader`/`LoraLoaderModelOnly` node and
   a `SaveImage` node; put `{prompt}` in the positive text (otherwise it is found through the sampler).
2. Click **🔄 List** to get the LoRAs ComfyUI sees, choose yours, set the trigger word.
3. **Generate then evaluate**: N images with the LoRA + the same seeds without it (A/B
   baseline), then R-FaceSim / Copycat / Black Hole and the **identity gain** of the LoRA.

---

## Architecture

```
manager.py                   ← Tkinter GUI (system Python with tkinter)
├── analyze_dataset.py        ← Main analysis engine (ComfyUI-future Python)
├── wd14_local.py             ← WD14-MOAT ONNX tagger
├── florence_local.py         ← Florence-2 captioner
├── joycaption_local.py       ← JoyCaption Beta One (2026 standard)
├── ai_detector_local.py      ← sdxl-detector ViT
├── artifact_detector_local.py ← HADM-light artifact detector
├── metadata_ai.py            ← C2PA / EXIF / PNG / filename heuristics
├── mask_generator_local.py   ← BriaRMBG subject masks
├── lora_prep.py              ← 19-target LoRA dataset preparation
├── lora_evaluator.py         ← Post-train R-FaceSim + Copycat
├── prompt_generator.py       ← Targeted ComfyUI workflow exporter
├── updater.py                ← GitHub release auto-updater
└── export_pdf.py             ← Landscape A4 PDF reporter
```

Two Python environments are used:
- **System Python (3.11/3.12)** — runs `manager.py` (needs tkinter + PIL.ImageTk)
- **ComfyUI's `python_embeded`** — runs `analyze_dataset.py` and `lora_evaluator.py` (has CUDA torch + insightface + transformers)

You can adapt the path to `python_embeded.exe` in `manager.py` (`COMFYUI_FUTURE_PY` constant) if you use a different ComfyUI install.

---

## Documentation

For the full, in-depth documentation see [**DOCUMENTATION.md**](DOCUMENTATION.md).

It covers every UI section, every JSON field, every training target's specifics, performance benchmarks, the cache format, and a complete troubleshooting guide.

---

## Updating

The tool has a built-in updater under the **⚙ Config** tab:

1. Click `🔄 Check for updates`
2. If a newer GitHub release exists, click `⬇ Install update`
3. The current `.py` files are backed up to `_backup_before_update/`
4. New files are pulled (via `git pull` if installed via git, otherwise via release zip)
5. Restart the app

---

## Roadmap

| Phase | Features | Status |
|---|---|---|
| **A** | JoyCaption + tag frequency overfit alerts | ✅ |
| **B** | sdxl-detector + HADM-light + C2PA/EXIF metadata | ✅ |
| **C** | CLIP diversity + per-target scoring + aspect-ratio | ✅ |
| **D** | Inline caption editor + BriaRMBG subject masks | ✅ |
| **E** | Targeted prompt generator + post-train evaluator + GitHub auto-update | ✅ |
| Future | Native InstantID auto-fill (no ComfyUI roundtrip), benchmarks vs known-good datasets | 🟡 ideas welcome |

---

## Contributing

PRs welcome. Major directions where help is appreciated:
- Adding new LoRA training targets (the catalog in `lora_prep.py` is structured for easy addition)
- Improving HADM artifact detection (HADM-full integration with Detectron2)
- ComfyUI API integration for auto-fill (instead of exporting JSON workflows)
- Screenshots / demo GIFs for the README

---

## Credits & inspiration

- **InsightFace** — antelopev2 face analysis
- **WD14 Tagger** (SmilingWolf) — booru-tags captioning
- **Florence-2** (Microsoft) — natural captioning
- **JoyCaption Beta One** (fancyfeast) — modern LLaVA-based persona captioning
- **Organika/sdxl-detector** — AI image classifier
- **BriaRMBG-1.4** (BRIA AI) — background removal
- **HADM paper** ([arXiv 2411.13842](https://arxiv.org/abs/2411.13842)) — anatomical artifact detection
- **MirrorMetrics** ([AndyLone22](https://github.com/AndyLone22/MirrorMetrics)) — post-training LoRA evaluation methodology
- **Meta-PHD** ([arXiv 2503.22352](https://arxiv.org/abs/2503.22352)) — R-FaceSim metric

---

## License

This project uses a **source-available license** with a small-entity commercial
tier and a paid commercial tier — see [LICENSE](LICENSE) for the full text.

**Quick summary** (non-binding):

| Use case | Free under this license |
|---|---|
| Personal / hobby / research / educational | ✅ Yes |
| Freelancers and sole proprietors | ✅ Yes |
| Companies with **fewer than 5 employees** | ✅ Yes |
| Companies with **5 or more employees** (commercial use) | ❌ Needs a paid commercial license |
| Repackaging as a competing product | ❌ Not allowed |

For a commercial license inquiry, open an issue tagged
[`commercial-license-inquiry`](https://github.com/akalavol/LoRA-Dataset-Coach/issues/new?labels=commercial-license-inquiry).

Third-party models downloaded at runtime (Florence-2, JoyCaption, WD14,
sdxl-detector, BriaRMBG, InsightFace) are licensed under their own terms,
which you must comply with independently.
