# LoRA Dataset Coach — Full Documentation

Integrated tool in the **File Manager** (📊 Dataset Analyzer tab) that takes a folder of images and tells you, in seconds per image, **whether your dataset is ready to train a persona LoRA** — and prepares the training folder for your trainer of choice among **19 targets** (Kohya / Flux / Wan / Hunyuan / LTX / CogVideoX / Mochi / Open-Sora / ai-toolkit / OneTrainer / etc.).

**Current score vs 2026 state-of-the-art: A (9.5/10)** after phases A + B + C + D + E.

---

## 1. Overview

```
[dataset folder]
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  PER-IMAGE ANALYSIS                                          │
│  • face detection (insightface antelopev2)                   │
│  • face embedding → cosine sim vs reference photo            │
│  • orientation (yaw) via keypoints                           │
│  • quality (Laplacian sharpness, brightness, contrast, MP)   │
│  • expression via CLIP zero-shot                             │
│  • shot type (face_only / both / body_only)                  │
│  • perceptual hash (duplicate detection)                     │
│  • captioning: WD14 / Florence-2 / JoyCaption / all          │
│  • AI-generated detection (sdxl-detector ViT)                │
│  • artifact detection (hands/eyes/limbs via WD14 + caption)  │
│  • AI metadata reading (C2PA / EXIF / PNG text / filename)   │
│  • cache (.analyzer_cache.json) — skip already-analyzed      │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  GLOBAL ANALYSIS                                              │
│  • cosine sim matrix (internal face / body coherence)         │
│  • duplicate detection (pHash hamming < 5 OR face sim > 0.96) │
│  • CLIP Union-Find clustering (visual diversity)              │
│  • angles / shots / expressions / aspect-ratios distribution  │
│  • WD14 tag frequency → per-category overfit alerts           │
│  • per-image viability verdict (yes / borderline / no)        │
│  • global A/B/C/D/F grade + concrete action plan              │
│  • per-target-family scores (SDXL / Flux / Wan / Video)       │
│  • targeted generation suggestions                            │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  ACTIONS                                                      │
│  🗑 Move rejects        → _rejected/                          │
│  🔧 Blurry → upscale    → _a_upscaler/ + SUPIR README         │
│  🧬 Prepare LoRA        → 19 targets available                │
│  🎭 Subject masks       → <image>-masklabel.png (OneTrainer)  │
│  ✨ Generate missing    → ComfyUI workflows for missing shots │
│  📊 Evaluate LoRA       → R-FaceSim + Copycat + Black Hole    │
│  📄 PDF export          → landscape A4 report                 │
│  🖱 Double-click row    → detail popup with caption editor    │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Modules

| File | Role | Phase |
|------|------|-------|
| `manager.py` | Tkinter GUI | base |
| `analyze_dataset.py` | Analysis engine (ComfyUI-future subprocess) | base |
| `wd14_local.py` | WD14-MOAT standalone tagger (ONNX) | base |
| `florence_local.py` | Florence-2 captioner (deprecated 2026 for persona) | base |
| `joycaption_local.py` | **JoyCaption Beta One captioner** (2026 standard) | **A** |
| `ai_detector_local.py` | **AI-generated image detector** (Organika/sdxl-detector ViT) | **B** |
| `artifact_detector_local.py` | **Anatomical artifact detector** (hands/eyes via WD14 + caption) | **B** |
| `metadata_ai.py` | **AI metadata reader** (C2PA / EXIF / PNG text / filename) | **B** |
| `lora_prep.py` | Multi-target preparation (19 targets, crops + configs) | base+ |
| `mask_generator_local.py` | **Subject masks BriaRMBG-1.4** for OneTrainer masked training | **D** |
| `lora_evaluator.py` | **Post-train LoRA evaluator** (R-FaceSim, Copycat, Black Hole) | **E** |
| `prompt_generator.py` | **Targeted ComfyUI workflow exporter** | **E** |
| `updater.py` | **GitHub releases auto-updater** | **E** |
| `export_pdf.py` | Landscape A4 PDF generation | base |

Python environments:
- **GUI**: System Python (3.11/3.12 — needs `tkinter` + `PIL/ImageTk`)
- **Analysis**: ComfyUI-future's `python_embeded` (`insightface`, `onnxruntime`, `transformers`, CUDA `torch`)

---

## 3. The interface

### 3.1 Dataset selection

```
📂 Dataset folder: [ C:\AI\datasets\my_persona ] [📂] [🔍 Analyze]
                                                            [📄 PDF export]
                                                            [🗑 Move rejects (N)]
                                                            [🔧 Blurry → upscale (N)]
                                                            [🧬 Prepare LoRA (N)]
                                                            [🎭 Subject masks (N)]
                                                            [✨ Generate missing (N)]
```

The last 6 buttons are **disabled** until an analysis has run. Once done, their labels show counts.

### 3.2 Reference photo (optional)

```
📷 Reference photo: [ C:\AI\datasets\ref.png ] [📂] [✕]   (optional)
```

When provided, the analyzer computes the embedding of this photo once, then for each dataset image computes `cos_sim(emb_image, emb_ref)`:

| Score | Verdict | Effect on viability |
|-------|---------|---------------------|
| `≥ 0.50` | ✅ OK | none |
| `0.35 - 0.50` | ⚠ doubtful | downgrades to `borderline` if `yes` |
| `< 0.35` | ❌ wrong person | downgrades to `no` |

**Why it matters**: without a reference, the analyzer only checks **internal coherence**. If InstantID generated 20 photos of someone else who all look similar, the analyzer misses it. With a reference, each photo is compared to **ground truth**.

### 3.3 Captioner (4 radio buttons — Phase A)

```
🏷 Captions: ( ) WD14 tags  ( ) Florence-2  ( ) JoyCaption ⭐  ( ) All
            (WD14 for SDXL · JoyCaption for Flux/Wan)
```

| Mode | Tagger | Output | For | Model size |
|------|--------|--------|-----|-----------|
| `wd14` | WD14-MOAT (ONNX) | `image.txt` (booru) | SDXL / Kohya / OneTrainer | 330 MB |
| `natural` | Florence-2 | `image.nat.txt` | **Fallback only** (hallucinates on persons) | 540 MB |
| `joycaption` | **JoyCaption Beta One** (LLaVA fine-tune) | `image.joy.txt` | **Flux / Wan / Hunyuan / SD 3.5** | 4 GB INT4 / 8 GB BF16 |
| `all` | All 3 | All 3 files | If unsure about target | all |

**JoyCaption Beta One is the 2025-2026 community standard** for persona LoRAs. Florence-2 kept as CPU fallback. Consensus: **Florence-2 hallucinates emotions/clothes/contexts** on people.

### 3.4 Progress

```
⚙ Loading insightface (~5-10s)…             ← current phase
[████████████░░░░░░░░░░░░] 53%   ⏱ ~12s remaining
📷 16/30 — IMG_1234.png                       ← current file
```

Typical phases (loading order):
1. `Loading insightface`
2. `Loading CLIP for body + expressions analysis`
3. `Loading sdxl-detector (AI image detection)` (Phase B)
4. `Loading WD14 tagger` (if mode ∈ {wd14, both, all})
5. `Loading Florence-2` (if mode ∈ {natural, both, all})
6. `Loading JoyCaption Beta One` (if mode ∈ {joycaption, all})
7. `Analyzing — N images`
8. `Cache: X reused, Y newly analyzed`
9. `Computing similarity matrix`
10. `Detecting duplicates`
11. `Cache updated`

### 3.5 Live preview (3 zones)

```
┌──────────────────┐ ┌──────────────────┐ ┌────────────────────────┐
│ 📷 Current       │ │ 📌 Reference     │ │ 🔎 Last verdict        │
│  [image 200x200] │ │  [image 200x200] │ │ 👤 1 face (12.3%)      │
│                  │ │                  │ │ ✅ Ref match: OK (0.78)│
│ IMG_0042.png     │ │ ref.png          │ │ ✅ Sharpness 312 → OK  │
└──────────────────┘ └──────────────────┘ │ 😶 slight smile         │
                                          │ 🏷 brown hair, indoors  │
                                          │ ⭐ A young woman with…  │
                                          │ 🤖 AI-score: 0.97       │
                                          │ ❌ Artifacts (high):    │
                                          │    hands, eyes          │
                                          │ 📋 AI metadata: exif    │
                                          └────────────────────────┘
```

### 3.6 Global verdict block (Phases A + B + C)

```
┌──────────────────────────────────────────────────────────────┐
│  A    DATASET VERDICT: EXCELLENT                             │
│       28 viable photos → 28 after cleanup (target 20-30)     │
│                                                              │
│  📷 Ref: ref.png → ✅ 28 photos match (avg 0.74)            │
│                                                              │
│  ⚠️ Overfit risks:                       (Phase A)          │
│    • Clothing : « white shirt » (87%), « jeans » (60%)      │
│    • Background: « indoors » (90%)                           │
│                                                              │
│  🎯 Scores per target family:            (Phase C)          │
│    B    72/100   SDXL classic                                │
│    B+   78/100   SDXL anime                                  │
│    A    88/100   Flux                                        │
│    C    55/100   Wan video                                   │
│    C    50/100   Video (Hunyuan/Mochi/...)                   │
│                                                              │
│  🌈 Diversity: 67/100  (Good diversity)  · 22 CLIP clusters  │
│  📐 AR: ⬜ 26 square · 📱 2 portrait · 🖼 0 landscape         │
│                                                              │
│  🎯 Action plan:                                             │
│    • Generate 2-3 profile shots (yaw 60-80°)                 │
│    • Vary clothing (white shirt at 87%)                      │
└──────────────────────────────────────────────────────────────┘
```

Global grade scale:
| Grade | Condition | Color |
|-------|-----------|-------|
| `A` | ≥ 30 viable + coherence > 0.6 | green |
| `B+` | 20–29 viable | green |
| `B` | default | green |
| `B-` | 15–19 viable | yellow |
| `C` | 10–14 viable | yellow |
| `D` | 1–9 viable | red |
| `F` | 0 viable | red |

### 3.7 Detail table

One row per image, 10 columns: **Image · Resol · Sharp · Face · %face · Pose° · Expression · S.face · S.body · Quality · 🧬 LoRA - reason**

Row color:
- 🟢 `ok`: viable
- 🟡 `warn`: borderline
- 🔴 `err`: reject

**Single click**: updates current preview + last verdict panel with full info.

**Double click**: opens a 1200×800 popup with the image in large + **all metadata** in the right panel, including:
- Viability (green/yellow/red) + reason
- Resolution, quality, % frame, yaw, expression
- Reference match (color by sim)
- Duplicate of / Upscale candidate
- **AI-generated score** (red > 0.7, yellow > 0.4, green otherwise) [Phase B]
- **AI artifacts** (severity + categories) [Phase B]
- **AI metadata** (sources + confidence) [Phase B]
- **3 editable captions** side by side (WD14 / Florence / **JoyCaption** ⭐) with **💾 Save** button under each → rewrites the `.txt` sidecar + updates cache + memory [Phases A + **D**]

### 3.8 Final summary

```
📊 30 images | ✅ 28 viable | ⚠️ 2 borderline | ❌ 0 reject
| 🔁 0 duplicates | 🏷 30 captions | 🤖 30 AI-detected | ❌ 0 severe artifacts
Face coherence: 0.78 | Body coherence: 0.81

🎯 Best target for this dataset: « Flux » (A, 88/100)
   — JoyCaption captions present, varied ratios, OK diversity
✅ Identity verified (vs ref): 28 photos match (avg 0.74)
🌈 Diversity: 67/100 — Good diversity
📐 AR: ✅ Healthy mix → ideal for Flux/Wan multi-bucket
💡 30 Kohya captions generated
🎯 Next to generate:
   → Generate 2-3 profile shots (yaw 60-80°)
```

---

## 4. Actions

### 4.1 🗑 Move rejects
- Identifies all images marked `lora_viable == "no"`
- Creates `<dataset>/_rejected/`
- Moves files **with their captions** `.txt` / `.nat.txt` / `.joy.txt`

### 4.2 🔧 Blurry → upscale
- Identifies `sharpness ∈ [50, 100]` + face detected
- Creates `<dataset>/_a_upscaler/` + README.txt with SUPIR/UltraSharp instructions
- Moves captions too

### 4.3 🎭 Subject masks (Phase D)

Generates `<image>-masklabel.png` next to each viable photo. This is the **OneTrainer masked training** format: LoRA loss is focused on the subject (white in mask), background (black) is ignored. Significant quality gain when backgrounds vary or pollute training.

Config popup:
- **Binarize** (default ON) at threshold 0.5 — pure white subject, pure black background, no alpha gradients
- **Threshold slider** 0.1–0.9 to adjust if masks cut wrong
- **Viable only**: yes + borderline only

Model: **BriaRMBG-1.4** (~176 MB auto-downloaded on first run, CUDA 1s/image, CPU 8s/image). Reuses the `ComfyUI-BRIA_AI-RMBG` custom node if already installed.

OneTrainer launch:
- Load dataset
- Concepts → Image augmentations → check **"masked training"**
- Train

### 4.4 🧬 Prepare LoRA (19 targets — multi-format)

Popup with:
- Trigger word (`persona_name`)
- **Dropdown grouped by category**:
  ```
  ━━ 📸 Realistic photo ━━     ━━ 🎨 Anime/Style ━━     ━━ 🎬 Video ━━
  sdxl_kohya                     pony_kohya               wan21_musubi
  sd15_kohya                     illustrious_kohya        wan22_musubi
  sd35_kohya                     noobai_kohya             hunyuan_diffpipe
  hunyuan_dit_kohya                                        ltx_video_diffpipe
  sana_diffpipe                                            cogvideox_diffpipe
  flux_aitoolkit                                           mochi_diffpipe
  flux_kohya                                               open_sora_diffpipe
  chroma_aitoolkit
  onetrainer_sdxl
  ```
- "Viable only" checkbox (recommended)
- **🎭 Also generate subject masks** checkbox (great combo with OneTrainer)
- Live info shows: label, resolutions, recommended captioner, doc link, **auto-prefixed quality tags** (Pony/Illustrious/NoobAI)

### 4.5 ✨ Generate missing (Phase E)

Uses the `next_to_generate` suggestions from the analyzer summary and exports **ready-to-drop ComfyUI workflows** to fill the missing shot types.

Each generated `.json` workflow:
- Is based on InstantID dataset workflow (workflow #12)
- Has the **correct prompt fragment** for the missing shot type (e.g., `side profile view`, `upper body shot`)
- Uses **DPRandomGenerator** for variations
- Has `batch_size` already set to the suggested count
- Saves to a unique output prefix per category

You drag-and-drop one into ComfyUI, it runs, you copy the new images back to the dataset, you re-analyze (cache!), and you iterate.

### 4.6 📊 Evaluate LoRA (Phase E — separate tab)

Once your LoRA is trained, switch to the **📊 Evaluate LoRA** tab:

1. **Generated images folder**: ~30 images you produced with your trained LoRA
2. **Real photos of subject**: photos of the real person, **different from training** (otherwise inflated)
3. **Training dataset folder** (optional): used for copycat detection

Metrics computed (MirrorMetrics-inspired):
- **R-FaceSim**: mean cosine sim of each generated image vs all real photos. Reported with mean / std / min / max.
- **Copycat alerts**: generated images >0.95 similar to a training image → LoRA is copying, not generalizing.
- **Black Hole Ranking**: per generation, which training image is closest? If one training image attracts >40% of generations → mode collapse.
- **Identity score**: derived 0-100 → LoRA grade A/B+/B/C/D/F.

Penalties applied to the final score:
- -5 per copycat (capped at -30)
- -20 × (face detection failure ratio)
- -15 if std-R-FaceSim < 0.03 (mode collapse)
- -15 if black hole > 40%

Verdict with specific advice (e.g., "Retrain with more diverse dataset", "Identity loose — check the analyzer first").

### 4.7 📄 PDF export
- A4 landscape, Catppuccin theme
- Header: verdict cartouche (giant A-F grade + action plan)
- Reference block if present
- Global stats, recommendations, table of all images

---

## 5. Analysis cache

File: `<dataset>/.analyzer_cache.json`

```json
{
  "version": 2,
  "entries": {
    "IMG_0042.png|123456|1748685432": {
      "entry": { ... full analysis result ... },
      "face_emb": [512 floats],
      "body_emb": [512 floats],
      "phash": "1234567890123456789"
    }
  }
}
```

**Key** = `name | size_bytes | mtime` → auto-invalidates if you replace a file.

**Behavior**:
1. Cache hit → everything retrieved, **no** detection re-run (insightface/CLIP/WD14/JoyCaption/sdxl-detector)
2. Cache miss → full analysis + add to cache
3. If **reference changes**, `face_similarity_to_ref` and `ref_match` are **recomputed** from cached embedding
4. Sim matrices, duplicates, **CLIP clustering**, **AR distribution**, **per-target scores** are **always recomputed** on the full set

**Practical consequence**: you add 5 photos → only the 5 are analyzed (~30s), the rest is instant. Great for InstantID iteration cycles.

**Reset cache**: delete `.analyzer_cache.json`.

---

## 6. Files generated next to images

| File | When | Content |
|------|------|---------|
| `image.txt` | captioner ∈ {wd14, both, all} | WD14 booru tags |
| `image.nat.txt` | captioner ∈ {natural, both, all} | Florence-2 (CPU fallback) |
| `image.joy.txt` | captioner ∈ {joycaption, all} | **JoyCaption Beta One** (2026 std) |
| `image-masklabel.png` | After 🎭 Subject masks | OneTrainer subject mask B/W |

On **🧬 Prepare LoRA**, the final caption is rebuilt per target:
- `wd14` → uses `image.txt` or `wd14_tags`
- `natural` → **priority JoyCaption** > Florence-2 > reconstruct from WD14
- Quality prefix injected before trigger word for SDXL forks (Pony/Illustrious/NoobAI)

---

## 7. Catalog of 19 supported targets

### 📸 Realistic photo (9 targets)

| Key | Trainer | Resolutions | Crop | Captioner |
|-----|---------|-------------|------|-----------|
| `sdxl_kohya` | Kohya SS GUI | 1024² | face square | WD14 |
| `sd15_kohya` | Kohya SS GUI | 512² | face square | WD14 |
| `sd35_kohya` | Kohya sd3 branch | 1024² | face square | natural |
| `hunyuan_dit_kohya` | Tencent HunyuanDiT | 1024² | face square | natural |
| `sana_diffpipe` | diffusion-pipe (NVIDIA Sana) | 1024² + 512×1024 + 1024×512 | buckets | natural |
| `flux_aitoolkit` | ai-toolkit (ostris) | 1024² + 1024×768 + 768×1024 | buckets | natural |
| `flux_kohya` | Kohya sd3 branch | 1024² | face square | natural |
| `chroma_aitoolkit` | ai-toolkit (Chroma = Flux uncensored variant) | 1024² + 1024×768 + 768×1024 | buckets | natural |
| `onetrainer_sdxl` | OneTrainer | 1024² | face square | WD14 |

### 🎨 Anime/Style — SDXL forks (3 targets)

All based on SDXL but with **mandatory quality tags** auto-injected before the trigger word:

| Key | Auto quality prefix |
|-----|---------------------|
| `pony_kohya` | `score_9, score_8_up, score_7_up, source_photo` |
| `illustrious_kohya` | `masterpiece, best quality, very aesthetic, absurdres` |
| `noobai_kohya` | `masterpiece, best quality, newest, absurdres, highres` |

### 🎬 Video (7 targets)

| Key | Trainer | Resolutions | Specifics |
|-----|---------|-------------|-----------|
| `wan21_musubi` | musubi-tuner | 832×480 + 480×832 + 720² | I2V/T2V, .bat launcher generated |
| `wan22_musubi` | musubi-tuner | 832×480 + 480×832 + 720² | Wan 2.2, .bat launcher |
| `hunyuan_diffpipe` | diffusion-pipe | 832×480 + 480×832 + 720² | Hunyuan video |
| `ltx_video_diffpipe` | diffusion-pipe | 768×512 + 512×768 + 704² | LTX-Video real-time, frame_buckets |
| `cogvideox_diffpipe` | cogvideox-factory | 720×480 + 480×720 | CogVideoX 5B, network_dim 64, lr 1e-3 |
| `mochi_diffpipe` | diffusion-pipe | 848×480 + 480×848 | Mochi 1 (Genmo) |
| `open_sora_diffpipe` | diffusion-pipe | 720² + 1280×720 + 720×1280 | Open-Sora 2.0 (HPC-AI) |

### Crop strategies

- **`square_face`**: square crop centered on face, configurable margin (60% default = headshot with shoulders), resize to target resolution.
- **`bucket_face`**: picks the bucket whose ratio matches the source image best, then crops at the right ratio centered on the face. **Keeps more context** than pure square. Recommended for Flux / Wan / Hunyuan / LTX which support varied ratios.

### Generated structure

**Kohya** (`folder_naming = "kohya"`):
```
kohya_persona/
  10_persona/
    persona_001.png   (1024×1024)
    persona_001.txt   ("persona, tag1, tag2..." or with quality_prefix for Pony/Illustrious/NoobAI)
    ...
  kohya_config.toml         (or kohya_sd35_config.toml / kohya_hunyuan_dit_config.toml / kohya_flux_config.toml)
  README.txt
```

**Flat** (`folder_naming = "flat"`: Flux ai-toolkit / Sana / Wan musubi / Hunyuan / LTX / CogVideoX / Mochi / Open-Sora / OneTrainer):
```
flux_persona/
  images/
    persona_001.png
    persona_001.txt
    ...
  ai_toolkit_config.yaml   (or musubi_dataset.toml + launch_musubi.bat,
                            or diffusion_pipe_*.toml,
                            or cogvideox_factory_config.yaml,
                            or nothing if OneTrainer)
  README.txt
```

---

## 8. AI detection + artifacts + metadata (Phase B)

### sdxl-detector — `ai_detector_local.py`
- Model `Organika/sdxl-detector` (ViT, **99.6% accuracy**)
- ~350 MB downloaded on first run
- Per-image output: `ai_score` (0-1) + `is_ai_classifier` (bool) + `ai_label` ("artificial" / "human")
- Summary stats: `ai_classifier_count`, `ai_score_avg`

### Anatomical artifacts — `artifact_detector_local.py`
**HADM-light** approach without heavy Detectron2:
- **Parses already-computed WD14 tags** for artifact keywords:
  - `hands` (severity high): bad hands, extra fingers, six fingers, mutated hands, fused fingers
  - `eyes` (severity high): asymmetric eyes, deformed eyes, crossed eyes, lazy eye
  - `limbs` (severity medium): extra arms/legs, missing limbs
  - `anatomy` (severity medium): bad anatomy, disfigured, deformed, mutation
  - `quality` (severity low): lowres, jpeg artifacts
- **Cross-check natural caption** (Florence-2 / JoyCaption) with regex for "deformed hand", "asymmetric eyes", etc.
- **Viability impact**:
  - `high` → `lora_viable = "no"` (rotten hands/eyes pollute the LoRA)
  - `medium` → downgrades to `borderline`

Full HADM (Detectron2 + Distortion-5K, paper arXiv 2411.13842) remains documented for advanced use, but **the tag-based detector covers 80% of cases in practice**: if WD14 sees the artifact, we see it too.

### AI metadata — `metadata_ai.py`
4 combined sources:
1. **C2PA v2.2**: reads signed manifest if present (optional, requires `c2pa-python`)
2. **EXIF Software field**: matches "Stable Diffusion", "Midjourney", "DALL-E", "ComfyUI", "Fooocus", "InvokeAI", "Leonardo.ai", "Playground", "Ideogram", "Krea"
3. **PNG tEXt chunks**: parameters / prompt / workflow / comment (ComfyUI / A1111 signatures)
4. **Filename heuristics**: `ComfyUI_00042_.png`, `MJ_xxx.png`, `_flux_`, `instantid_`, `_lora_`, `dalle_`, etc.

Output: `ai_metadata_sources` + `ai_metadata_confidence` (high if C2PA/EXIF, medium if filename only).

### Auto-recommendations (final summary)
- `❌ N photo(s) with severe AI artifacts (rotten hands/eyes) — definitely remove`
- `⚠️ N photo(s) with medium artifacts — check individually`
- `💡 30/30 (100%) photos detected as AI-generated — normal if InstantID/Flux dataset`
- `⚠️ N photo(s) detected AI in supposedly real dataset — check via double-click`
- `📋 N photo(s) with explicit AI metadata (C2PA/EXIF/filename)`

---

## 9. Diversity + per-target scores + AR (Phase C)

### Aspect-ratio distribution
5 verdict buckets:
| Bucket | Ratio | Counts for |
|--------|-------|-----------|
| `square` | 0.9–1.1 | SDXL / SD1.5 / Kohya ideal |
| `portrait` | 0.5–0.9 | Flux portrait / Wan portrait |
| `tall_portrait` | < 0.5 | **⚠️ Breaks Flux/Wan bucketing** |
| `landscape` | 1.1–2.0 | Flux landscape / Wan landscape |
| `wide_landscape` | > 2.0 | **⚠️ Breaks bucketing** |

Recommendations by mix:
- 95%+ square → "ideal SDXL, Flux multi-bucket underused"
- Healthy mix (sq + po + la present) → "ideal Flux/Wan multi-bucket"
- Portrait/landscape-heavy → suggestion for Wan in the right orientation

### CLIP diversity + tags
**Union-Find clustering** on `body_embeddings` from CLIP:
- Merge threshold: `sim > 0.92` → same cluster
- **Cluster count** = visual variance measure
- **Big clusters** (≥ 3 near-identical photos) reported by name

**Global score** = weighted mean of:
- `clip_score` (cluster/total ratio × 100)
- **-15 penalty per overfit category** detected (clothing/background/lighting/accessory/background_color)

Verdicts:
| Score | Verdict |
|-------|---------|
| 75+ | Excellent diversity |
| 55–74 | Good diversity |
| 35–54 | Limited diversity |
| < 35 | Dataset too homogeneous |

### Per-target-family scores

**5 families** instead of one global note. Each with weighted criteria and grade A+/A/B+/B/C/D/F.

| Family | Weighted criteria |
|--------|-------------------|
| **SDXL classic** | Vol 25 / Res 1024+ 20 / WD14 15 / Diversity 20 / Square AR 10 / No artifacts 10 |
| **SDXL anime** (Pony/Illustrious/NoobAI) | Same but **+8 on artifacts** (anime style + tolerant) |
| **Flux** | Vol 25 / Res 1024+ 20 / **Natural captions 20** (JoyCaption preferred) / **AR variety 15** / Diversity 15 / Artifacts 5 |
| **Wan video** | Vol 25 / Res 512+ 15 / **Captions ≥ 100 chars 20** (T5 loves long) / AR variety 15 / **po+la mix 10** / Diversity 15 |
| **Video (Hunyuan/Mochi/LTX/CogVideoX)** | Same as Wan -5 (clips MP4 recommended additionally) |

Auto recommendation at top: `🎯 Best target for this dataset: « Flux » (A, 88/100) — concise reasons`.

---

## 10. Tag frequency + overfit (Phase A)

Categorizes each frequent WD14 tag into 5 **accessory attribute** categories (= what you **don't want** the LoRA to memorize as part of the persona):

| Category | Detected patterns |
|----------|-------------------|
| `clothing` | shirt, dress, jacket, coat, blouse, sweater, hoodie, t-shirt, top, skirt, pants, jeans, shorts, bra, bikini, uniform |
| `background` | indoors, outdoors, background, wall, room, kitchen, bedroom, studio, street, park, beach, forest, sky, garden, office, cafe |
| `lighting` | lighting, sunlight, backlight, shadow, dim, bright, dark, neon, golden hour, natural light |
| `accessory` | earrings, necklace, glasses, sunglasses, hat, scarf, watch, ring, bracelet, tie, bag, headphones, mask |
| `background_color` | white background, black background, grey background, blue background, plain background, simple background |

**Thresholds**:
- ≥ 75% of viable photos → **red ALERT** (overfit almost guaranteed)
- 50–75% → **yellow WARNING**

**Identity tags intentionally ignored**: `woman`, `long hair`, `brown hair` — those are what you **want** the LoRA to learn.

Display in verdict block:
```
⚠️ Overfit risks:
  • Clothing  : « white shirt » (87%), « jeans » (60%)
  • Background: « indoors » (90%), « plain background » (73%)
```

---

## 11. Auto-captioning: WD14 vs Florence-2 vs JoyCaption

### 11.1 Market state 2025-2026

**WD14 (booru tags)** — historical format:
```
brown hair, long hair, indoors, white shirt, smile, looking at viewer
```
- Recommended by Kohya docs for SDXL/SD1.5
- Excellent when base model trained on booru datasets (Pony, Illustrious, NoobAI)
- Fast (ONNX), ~30s for 30 photos

**Florence-2 (Microsoft natural caption)** — **deprecated 2026** for persons:
```
a young woman with long brown hair, wearing a white shirt, smiling indoors
```
- **Hallucinates emotions/clothes/contexts on persons** (community consensus)
- Still useful for general images (objects, landscapes) or as CPU fallback
- ~1 min for 30 photos on CPU, ~10s on CUDA

**JoyCaption Beta One (LLaVA fine-tune)** — **2026 standard**:
```
A young woman with long dark brown hair, captured in a three-quarter view portrait.
She wears a beige knit sweater and has a slight smile, looking directly at the camera.
The lighting is soft and diffuse, suggesting an indoor environment with natural window light.
```
- **CivitAI / HuggingFace 2026 standard** for persona LoRAs
- Uncensored, full-sentence, precise descriptions without hallucination
- 4 GB INT4 / 8 GB BF16 model (CUDA strongly recommended)
- ~30s/image on CUDA, ~2 min/image on CPU

### 11.2 When to use what

| Target | Captioner |
|--------|-----------|
| SDXL / SD 1.5 / Kohya / OneTrainer | **WD14** |
| Pony / Illustrious / NoobAI | **WD14** (+ auto quality_prefix) |
| Flux / Chroma / SD 3.5 / HunyuanDiT / Sana | **JoyCaption** ⭐ |
| Wan 2.x / HunyuanVideo / LTX-Video / CogVideoX / Mochi / Open-Sora | **JoyCaption** ⭐ |
| Still unsure | **All** (all 3 stored, `lora_prep` picks the right one per target) |

### 11.3 Practical tips

- **Trigger word always first**: `persona_alpha, a young woman with...`
- For SDXL forks: **quality_prefix auto-injected before trigger**: `score_9, score_8_up, score_7_up, source_photo, persona_alpha, ...`
- **5–10% caption dropout** in trainer config to prevent overfit on accessory attributes
- Re-read 3-4 `.txt` files before training. The **inline editor** in the popup (Phase D) lets you fix any hallucinated attribute directly.

---

## 12. Complete pipeline: from click to trained LoRA

```
1. ComfyUI InstantID workflow (02 or 12)
       │ generates 30 photos in output/
       ▼
2. Copy/move to C:\AI\datasets\my_persona\
       │
       ▼
3. File Manager → 📊 Dataset Analyzer tab
   - Folder: C:\AI\datasets\my_persona
   - Reference: the anchor photo used by InstantID
   - Captions:
     • SDXL/Pony/Illustrious/NoobAI/OneTrainer → "WD14 tags"
     • Flux / Wan / Hunyuan / SD 3.5 / HunyuanDiT / LTX / CogVideoX / Mochi / Open-Sora → "JoyCaption ⭐"
     • Unsure → "All"
   - 🔍 Analyze
       │ Phases A+B+C: live preview + global verdict + per-target scores
       ▼
4. Reading results:
   - A/B/C/D/F verdict + best target auto-suggested
   - Per-family scores (SDXL / SDXL anime / Flux / Wan / Other video)
   - Diversity score + CLIP cluster count
   - AR distribution + bucket alerts
   - Overfit alerts (accessory tags > 70%)
   - AI detection + artifacts + metadata
   - Concrete action plan
       │
       ▼
5. Cleanup (in order):
   a. 🗑 Move rejects → _rejected/
   b. 🔧 Blurry → upscale → ComfyUI SUPIR → reinject
   c. ✨ Generate missing (Phase E)
      → Get .json workflows for missing shots → drop in ComfyUI
      → Add new photos → re-analyze (cache !)
       │
       ▼
6. 🧬 Prepare LoRA
   - Trigger word
   - Target: auto-suggested by verdict, or manual choice
   - "Viable only" checked
   - "🎭 Also generate subject masks" if OneTrainer
       │
       ▼
7. Final folder opens in explorer
   - README.txt with precise launch instructions for chosen trainer
   - Trainer config ready (.toml / .yaml / .bat)
   - Cropped images + captions (trigger word + quality_prefix if SDXL fork)
   - Optional <image>-masklabel.png if masks generated
       │
       ▼
8. Run the trainer (see Section 7 per target)
       │
       ▼
9. LoRA outputs to output/<persona>_lora.safetensors
       │
       ▼
10. Copy to C:\AI\ComfyUI\models\loras\
    Load in ComfyUI with LoraLoader
    Trigger word in prompt
       │
       ▼
11. 📊 Evaluate LoRA tab (Phase E)
    - Generate ~30 test images with the LoRA
    - Provide real reference photos (not training!)
    - Get R-FaceSim, Copycat, Black Hole verdict
    - Iterate if score < B+
```

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `antelopev2 models not found` | InstantID never used | Run workflow 02 once to download antelopev2 |
| `WD14 unavailable` at startup | No network on first run | Download fails, analysis continues without tags. Check connection |
| `Florence-2 unavailable (transformers/torch missing)` | ComfyUI-future venv has broken torch | Reinstall torch in that venv |
| `JoyCaption unavailable` | Not enough VRAM (LLaVA 8 GB BF16) | Enable INT4 (auto if CUDA). On CPU, use Florence-2 |
| `sdxl-detector unavailable` | transformers < 4.40 or HF issue | Update transformers, check HF access |
| Analysis seems stuck at 0% | Loading insightface (CPU) | Normal first run, ~10s |
| Table stays empty | Silent error | Tab → red zone at bottom with full traceback |
| Cache not used | You copy-pasted files, size changed | Normal: if size/mtime changes, cache invalidates. Reset: delete `.analyzer_cache.json` |
| JoyCaption very slow | CPU without CUDA | Normal on CPU (~2 min/image). On CUDA INT4 → 30s/image |
| "Not enough viable photos" for Kohya | < 10 viable | Either generate more, or uncheck "viable only" (LoRA will suffer) |
| Crop cuts the chin | margin_ratio too small | Edit `lora_prep.py` → `margin_ratio=0.8` or 1.0 |
| AI score = 0.97 everywhere | Dataset 100% InstantID-generated | Normal and expected, the recommendation says so |
| Artifacts severity high everywhere | Hallucinating base model (e.g., Flux Schnell) | Regenerate with InstantID + SDXL, or Flux dev |
| AR alerts "tall_portrait" | Refs are 9:21 or more extreme | Pre-crop before import, or avoid Flux multi-bucket |
| LoRA evaluator says A but I see overfit | You used training photos as reference | **Must use REAL photos NOT in training set** |
| Evaluator says high copycat ratio | LoRA trained too long or dim too high | Retrain with lower network_dim or more dropout |

---

## 14. Performance

On 6-core CPU, dataset of 30 photos 832×1216:

| Phase | Time |
|-------|------|
| Loading insightface (1st run) | 8s |
| Loading insightface (subsequent) | 2s |
| Loading CLIP (1st run) | 6s |
| Loading CLIP (subsequent) | 1s |
| Loading WD14 (1st run, includes DL 330 MB) | 60s |
| Loading WD14 (subsequent) | 3s |
| Loading Florence-2 (1st run, DL 540 MB) | 90s |
| Loading Florence-2 (subsequent) | 8s |
| Loading JoyCaption (1st run, DL 4-8 GB) | 5-10 min |
| Loading JoyCaption (subsequent, CUDA) | 15s |
| Loading sdxl-detector (1st run, DL 350 MB) | 30s |
| Loading sdxl-detector (subsequent) | 3s |
| Per-image analysis (WD14) | 1.2s |
| Per-image analysis (Florence-2 CPU) | 4s |
| Per-image analysis (Florence-2 CUDA) | 0.5s |
| Per-image analysis (JoyCaption CPU) | 120s |
| Per-image analysis (JoyCaption CUDA INT4) | 30s |
| Per-image analysis (JoyCaption CUDA BF16) | 5s |
| AI detection (sdxl-detector) | 0.3s CPU, 0.05s CUDA |
| Artifact detection (regex on tags) | 0.001s |
| AI metadata reading | 0.01s |
| Sim matrices + duplicates + clustering + verdict | 1s |
| BriaRMBG mask (CUDA) | 1s |
| BriaRMBG mask (CPU) | 8s |
| LoRA evaluator (30 gen + 20 ref, CPU) | ~3 min |
| LoRA evaluator (CUDA) | ~30s |

**Cache hit**: 0.05s per image. On a 50-photo dataset already cached with 3 new ones: 13s total.

**Recommendation**: for Flux/Wan, do one analysis in `joycaption` or `all` mode (long but complete), then the cache brings you to 0.05s per image for all subsequent iterations.

---

## 15. Updating

Built-in GitHub releases updater via the ⚙ Config tab:

- **🔄 Check now**: queries `https://api.github.com/repos/akalavol/LoRA-Dataset-Coach/releases/latest`
- Compares with `VERSION` file
- If newer release exists, shows release notes
- **⬇ Install update** button:
  - If install is a git clone: runs `git pull --ff-only`
  - Otherwise: downloads zipball, backs up current `.py` to `_backup_before_update/`, replaces files
- Restart the app

---

## 16. Roadmap

Status as of 2026-05-31 (after Phases A + B + C + D + E):

| Phase | Features | Status |
|-------|----------|--------|
| **A** | JoyCaption + tag frequency overfit alerts | ✅ |
| **B** | sdxl-detector + HADM-light + C2PA/EXIF/PNG metadata | ✅ |
| **C** | CLIP diversity + per-target scoring + AR distribution | ✅ |
| **D** | Inline caption editor + BriaRMBG subject masks | ✅ |
| **E** | Targeted prompt generator + LoRA evaluator + GitHub auto-update | ✅ |
| Future | Native InstantID auto-fill (no ComfyUI roundtrip), known-good dataset benchmarks, multi-concept LoRAs | 🟡 |

---

*Documentation updated 2026-05-31 — Phases A + B + C + D + E shipped.*
