# Changelog

All notable changes to LoRA-Dataset-Coach will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.0.1] — 2026-05-31

### Added
- **Source-available license** replacing MIT: free for personal use, freelancers,
  and companies with fewer than 5 employees. Larger commercial use requires a
  separate commercial license.
- **CHANGELOG.md** following Keep a Changelog format.
- **In-app changelog viewer** in the ⚙ Config tab. Reads CHANGELOG.md locally
  and can refresh against the GitHub `main` branch for the latest entries.

### Changed
- README updated with the new license summary table.

---

## [v1.0.0] — 2026-05-31 — Initial public release

### Added — Phase A (Modern captioner + overfit detection)
- **JoyCaption Beta One** standalone module (`joycaption_local.py`). LLaVA fine-tune,
  2026 community standard for persona LoRAs. INT4 quantization support.
- New `joycaption` and `all` captioner modes in the analyzer (`analyze_dataset.py`).
- 4th radio button in the GUI (WD14 / Florence-2 / JoyCaption ⭐ / All).
- **Tag frequency analysis** with overfit alerts grouped by 5 categories:
  clothing, background, lighting, accessory, background_color.
- Thresholds: ≥75% = red alert, 50–75% = yellow warning.
- Identity tags (`woman`, `long hair`...) intentionally ignored.
- `.joy.txt` sidecar file written next to each image.
- `lora_prep.py` priority chain: JoyCaption > Florence-2 > WD14 reconstruction.

### Added — Phase B (AI detection + artifact detection + metadata)
- **`ai_detector_local.py`** — wraps `Organika/sdxl-detector` ViT (350 MB,
  99.6% accuracy). Per-image `ai_score`, `is_ai_classifier`, `ai_label`.
- **`artifact_detector_local.py`** — HADM-light detector using WD14 tags +
  natural-caption regex. Detects:
  - `hands` (high severity): bad hands, extra fingers, six fingers, mutated hands
  - `eyes` (high): asymmetric, deformed, crossed, lazy eye
  - `limbs` (medium): extra arms/legs, missing limbs
  - `anatomy` (medium): bad anatomy, disfigured, deformed, mutation
  - `quality` (low): lowres, jpeg artifacts
- **`metadata_ai.py`** — reads C2PA v2.2 manifests, EXIF Software field,
  PNG tEXt chunks, filename heuristics. Detects AI-generated images.
- Viability impact: `high` artifacts → `lora_viable=no`,
  `medium` → `borderline`.
- New summary fields: `ai_classifier_count`, `ai_score_avg`,
  `artifacts_high_count`, `artifacts_medium_count`, `ai_metadata_count`.

### Added — Phase C (Diversity + per-target scoring + AR distribution)
- **Aspect-ratio distribution** in 5 buckets (`square`, `portrait`,
  `tall_portrait`, `landscape`, `wide_landscape`) with bucket-cassure alerts.
- Target recommendations based on AR mix.
- **CLIP diversity clustering** via Union-Find on body embeddings
  (similarity > 0.92 = same cluster). Detects big clusters (≥3 near-identical photos).
- Diversity score with overfit category penalty (-15 per category).
- 4-grade verdict: Excellent / Good / Limited / Too homogeneous.
- **5 per-target-family scores** with distinct weighted criteria:
  - SDXL classic (Vol 25 / Res 1024 20 / WD14 15 / Diversity 20 / Square AR 10 / No artifacts 10)
  - SDXL anime (same +8 artifact tolerance)
  - Flux (Vol 25 / Res 20 / Natural captions 20 / AR variety 15 / Diversity 15 / Artifacts 5)
  - Wan video (long captions 20, po+la mix 10, +diversity)
  - Video other (Wan -5, MP4 clips recommended)
- Auto-recommendation: "🎯 Best target for this dataset: « X » (A, 88/100)"

### Added — Phase D (Inline editor + subject masks)
- **Inline caption editor** in the double-click popup. Edit any of the 3 captions
  (WD14, Florence-2, JoyCaption), 💾 Save button rewrites the sidecar `.txt` +
  updates cache + memory.
- **`mask_generator_local.py`** — reuses BriaRMBG-1.4 from ComfyUI-BRIA_AI-RMBG.
  Auto-downloads model (176 MB) if missing. CUDA 1s/image, CPU 8s/image.
- Stand-alone 🎭 Subject masks button + integrated checkbox in 🧬 Prepare LoRA.
- Binarization threshold control (default 0.5, OneTrainer-compatible).
- Output format: `<image>-masklabel.png` next to each viable image.

### Added — Phase E (LoRA evaluator + targeted generation + auto-updater)
- **`lora_evaluator.py`** — new 📊 Evaluate LoRA tab. MirrorMetrics-inspired
  post-training evaluation:
  - R-FaceSim (mean cosine sim vs REAL reference photos)
  - Copycat Detector (sim > 0.95 vs training image = memorization)
  - Black Hole Ranking (which training image attracts most generations)
  - Mode collapse signal (std < 0.03)
  - A/B+/B/C/D/F final grade with specific advice
- **`prompt_generator.py`** — ✨ Generate missing button. Turns
  `next_to_generate` suggestions into drop-ready ComfyUI workflows. Smart
  mapping: profile → side profile view, expression → multi-emotion DP variation.
- Workflows use InstantID + DPRandomGenerator + batch_size auto-set.
- **`updater.py`** — built-in GitHub releases auto-updater in ⚙ Config tab.
  Two modes: `git pull --ff-only` if git install, otherwise zipball download
  with backup of current `.py` files to `_backup_before_update/`.

### Added — Multi-target LoRA preparation (19 targets total)
**Photo (9):** sdxl_kohya, sd15_kohya, sd35_kohya, hunyuan_dit_kohya,
sana_diffpipe, flux_aitoolkit, flux_kohya, chroma_aitoolkit, onetrainer_sdxl.

**Anime / SDXL forks (3):** pony_kohya, illustrious_kohya, noobai_kohya.
Auto quality prefix injected before trigger word:
- Pony: `score_9, score_8_up, score_7_up, source_photo`
- Illustrious: `masterpiece, best quality, very aesthetic, absurdres`
- NoobAI: `masterpiece, best quality, newest, absurdres, highres`

**Video (7):** wan21_musubi, wan22_musubi, hunyuan_diffpipe,
ltx_video_diffpipe, cogvideox_diffpipe, mochi_diffpipe, open_sora_diffpipe.

Each target auto-generates the right crop strategy (`square_face` or
`bucket_face`), captioner choice (WD14 vs natural), config file (.toml/.yaml),
launch instructions (README + .bat for musubi-tuner).

GUI dropdown is grouped by category with separators.

### Added — Base features
- Tkinter GUI with Catppuccin Mocha dark theme.
- Face detection via InsightFace antelopev2 + 512-D embeddings.
- Reference photo identity check (cosine sim, threshold 0.5 / 0.35).
- Quality: Laplacian sharpness, brightness, contrast, megapixels.
- Pose estimation: yaw via 5 keypoints.
- Expression detection: CLIP zero-shot, 7 expression labels.
- Shot type: face_only / both / body_only via face proportion.
- Duplicate detection: dHash (Hamming < 5) + face sim > 0.96.
- Analysis cache (`.analyzer_cache.json`) — skip already-analyzed images
  by name+size+mtime key.
- 🗑 Move rejects → `_rejected/` folder.
- 🔧 Blurry → upscale → `_a_upscaler/` folder with SUPIR/UltraSharp README.
- 📄 PDF export (A4 landscape, Catppuccin theme, full table + verdict).
- Live preview during analysis (3 zones: current / reference / live verdict).

### Released
- GitHub repository: https://github.com/akalavol/coach/LoRA-Dataset-Coach
- License: Source-Available v1.0 (free for small entities < 5 employees).

---

## How to add an entry to this file

When releasing a new version:

1. Move the contents of `## [Unreleased]` to a new versioned section
   `## [vX.Y.Z] — YYYY-MM-DD`.
2. Group changes under: `### Added`, `### Changed`, `### Deprecated`,
   `### Removed`, `### Fixed`, `### Security`.
3. Bump the `VERSION` file.
4. Push, then run `gh release create vX.Y.Z --notes-file <(sed -n ...)`.

---

## [Unreleased]

### Planned
- Native ComfyUI API integration (no JSON workflow roundtrip)
- Multi-concept LoRA support
- Known-good dataset benchmark comparisons
- HADM-full integration (Detectron2-based, replaces the tag-only detector
  for users who want maximum precision)
