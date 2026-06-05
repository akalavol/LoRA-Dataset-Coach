# Changelog

All notable changes to LoRA-Dataset-Coach will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.1.0] — 2026-06-05

### Added — Audio dataset analysis (Voice & Music)
Two new tabs extend the Coach beyond images, to audio LoRA / voice training:

- **🎤 Voice RVC tab** — analyze a folder of voice recordings for RVC training
  (Applio). Per-file metrics: duration, **SNR** (signal-to-noise), RMS level,
  sample rate, fundamental frequency (F0). A→F grade per file + global summary
  (total duration, sample-rate consistency, top issues) and a "ready for RVC"
  verdict with recommendations. One-click launch of Applio.

- **🎵 Music LoRA tab** — prepare a music collection for **ACE-Step 1.5** LoRA
  training. Checks conformance to the ACE-Step format (`song.mp3` +
  `song.lyrics.txt` + `song.json` with bpm/keyscale/caption/timesignature/
  language), flags missing lyrics/annotations, reports BPM & key diversity, and
  can **auto-generate the missing `.json` templates**. One-click launch of the
  ACE-Step UI.

New modules: `analyze_voice.py`, `analyze_music.py`.
New deps: `librosa`, `soundfile`, `mutagen` (audio analysis only).

---

## [v1.0.13] — 2026-06-01

### Added — multiple reference photos (robust identity)
The reference can now be a **single photo OR a folder of photos** (face,
3/4, full-body…). When a folder is given, the analyzer builds an **identity
centroid** (mean of the detected face embeddings, re-normalized) instead of
relying on one possibly-bad shot. This makes the "right person / wrong person"
check much more robust and cuts false "wrong person" flags.
- New 📁 button in the analyzer to pick a reference folder (the 📄 button still
  picks a single photo).
- Preview shows the first ref image + the count (`📁 refs (4 réfs)`).
- Photos in the ref folder with no detectable face are skipped (and reported).
- The verdict shows e.g. `📷 Réf : 4 photos → ✅ 28 correspondent (avg 0.79)`.

Note: face identity uses InsightFace (very reliable). A full-body ref helps
mainly the face centroid; true body-shape matching isn't possible with the
current models, so body similarity stays "indicative" via CLIP.

---

## [v1.0.12] — 2026-06-01

### Fixed — Kohya config rejected ("extra keys not allowed @ data['model']")
The generated `kohya_config.toml` used an invalid structure (`[model]` /
`[folders]` / `[training]`) that Kohya's **Dataset Config** field rejects
(sd-scripts dataset configs must be `[general]` + `[[datasets]]` + subsets,
with no `[model]` section). Now the Kohya targets produce:
- **`dataset_config.toml`** — a valid sd-scripts dataset config (this is the
  file that goes in Kohya's "Dataset Config" field). It pins resolution,
  repeats, captions to the `N_persona/` image folder.
- **`kohya_GUI_SETTINGS.txt`** — the recommended model / LR / network settings
  to copy into the GUI (plain text, so it can't be mistaken for a loadable
  config). Also documents the no-toml "folder method".

README/launch instructions updated accordingly.

---

## [v1.0.11] — 2026-05-31

### Fixed — cache write failed on numpy values
`Echec ecriture cache : Object of type float32 is not JSON serializable`.
Some analysis fields are numpy `float32`/`int64`, which plain `json.dumps`
can't write — so the cache was **never saved**, and every run re-analyzed
everything (no duplicate-skipping). `save_cache` and the live `IMGINFO`
messages now use a numpy-aware serializer (`float32→float`, `int64→int`,
`ndarray→list`). The cache now persists and is read back at the start of the
next run, so unchanged photos are skipped.

To answer the question "shouldn't it read the JSONs first to avoid doing
duplicates?" — it already does (`load_cache` at the start of every analysis);
the write was just silently failing, so there was never a cache to read.

---

## [v1.0.10] — 2026-05-31

### Fixed (critical) — the in-app updater did nothing
The zip-based updater downloaded the new release but **copied 0 files**, so
after "updating" and restarting you were still on the old version. Two bugs:
- The file-exclusion check tested the *absolute* path, which always contains
  the temp folder name `_update_tmp` → **every** file was skipped.
- The extracted folder was detected with `next(iterdir())`, which could return
  the downloaded `release.zip` instead of the source folder.

Both fixed: extraction now goes to a dedicated subfolder, exclusion is checked
on the *relative* path, the new VERSION is read back and shown in the success
message, and the updater reports a real failure if nothing was copied.

> ⚠️ Because the **broken** updater ships in ≤ 1.0.9, those versions can't
> self-update to this fix. Update **once manually** (re-download the latest
> release zip), then the in-app updater works for all future versions.

---

## [v1.0.9] — 2026-05-31

### Fixed / Improved — analysis cache
- **The cache now actually persists.** Root cause of "it re-scans everything":
  on versions ≤ 1.0.5 the analysis crashed at the very end (the `summary_verdict`
  bug, fixed in 1.0.6) *before* the cache was written, so `.analyzer_cache.json`
  was never created. Now that analysis completes, the cache is saved.
- **Incremental cache saving.** The cache is now flushed every 10 images during
  the analysis (and during phase-2 captioning), not only at the end. If a run
  is interrupted (app closed, crash, power loss), already-analyzed photos are
  remembered and won't be redone on the next scan.
- **Explicit cache log.** The start of each analysis now reports
  `Cache trouvé : N photo(s) déjà analysée(s)` or `Aucun cache (1er scan)`, so
  you can see the cache working.

The cache logic itself (key = name|size|mtime) was verified correct: an
unchanged file is reused, a modified/replaced file is re-analyzed.

---

## [v1.0.8] — 2026-05-31

### Fixed
- **PDF export crash** on Unicode characters (e.g. the `→` arrow, `≥`, `°`,
  emoji) that the Helvetica core font can't encode. All text written to the
  PDF is now sanitized centrally (cell/multi_cell overrides): known symbols
  are mapped to ASCII, anything else outside latin-1 is replaced. No more
  `FPDFUnicodeEncodingException`.

### Added — clearer "what's missing"
- **Dataset coverage panel** in the verdict block, showing exactly which shot
  types are present vs missing:
  - `Angles : face 70 · 3/4 8 · profil 0  ❗ aucun profil`
  - `Cadrage : gros plan 75 · mi-corps 5 · plein pied 0  ❗ aucun plein pied`
  - `Expressions : sourire (60), neutre (20)  ❗ peu varié`
  Each count is colour-coded (green = well covered, yellow = thin, red = absent).
- **"Photos à générer pour compléter"** section listing the concrete missing
  shot types right under the coverage panel (in yellow).
- The coverage + photos-to-generate are also written to `_analysis_report.txt`.

---

## [v1.0.7] — 2026-05-31

### Fixed (regression)
- **The results table and bottom summary were never filled after an analysis.**
  That display code had ended up inside `_move_rejected_photos` by mistake, so
  it only ran if you moved rejects. Moved it into its own
  `_populate_results_table` method, called at the end of every analysis.

### Added
- **Auto-save after every analysis.** Two files are now written to the dataset
  folder automatically:
  - `_analysis_report.json` — the full result (re-loadable, shareable)
  - `_analysis_report.txt` — a readable summary (grade, action plan, per-target
    scores, recommendations, per-image table)
  Nothing is lost anymore, even if you close the app.
- **The global grade is now shown at the top of the bottom summary**
  (`🏆 NOTE : A (EXCELLENT)`), in addition to the big verdict block above.

---

## [v1.0.6] — 2026-05-31

### Fixed (critical)
- **Crash at the very end of every analysis** (`UnboundLocalError: summary_verdict`).
  The `summary` dict referenced the global verdict before it was computed, so
  the whole run crashed after processing all images — losing the entire result.
  The verdict is now injected into the summary after the grade is computed.
- **JoyCaption was fully disabled when `bitsandbytes` wasn't installed.**
  The INT4 path failed at model load (not at config creation, so the old
  try/except missed it). Now bitsandbytes is detected up-front; if absent,
  JoyCaption loads in **BF16 (~8 GB VRAM)** instead of being skipped. A
  hard INT4 failure also falls back to BF16 automatically.

### Note
If you downloaded a zip (not a git clone), update via ⚙ Config → 🔄 Check now
→ ⬇ Install update (zip mode), or re-download the latest release. This release
also includes the GPU acceleration from v1.0.4 — insightface + CLIP will use
your GPU instead of CPU.

---

## [v1.0.5] — 2026-05-31

### Changed (major efficiency win)
- **Two-phase captioning pipeline.** Natural captions (Florence-2 / JoyCaption)
  are the slowest step. They are no longer run inline on every image. Instead:
  - **Phase 1** runs the fast analysis on all images (face, CLIP, quality,
    WD14 tags, AI detection, artifacts) and computes the viability verdict.
  - **Phase 2** runs Florence-2 / JoyCaption **only on the photos judged viable
    or borderline** — the rejects (no face, blurry, wrong person, duplicates,
    severe artifacts) are skipped entirely.
  - On a 200-photo set with ~80 viable, JoyCaption now runs on 80 images
    instead of 200 — roughly **60% less time** on the slow step.
- Phase 2 has its own progress bar and live preview, so you see the fast
  results (and the dataset verdict) before the slow captioning starts.
- The cache persists phase-2 captions, so a later run that keeps more photos
  only captions the newly-kept ones.
- Per-target scores now correctly reflect the phase-2 captions.
- Pre-flight warning updated to explain that only viable photos get captioned.

---

## [v1.0.4] — 2026-05-31

### Fixed (big performance win)
- **insightface face detection and CLIP (body + expression) were pinned to
  CPU** even when a CUDA GPU was available. They now run on the GPU when one
  is present. Since both run on *every* image, this is a ~3-5× speedup on
  those stages — meaningful for every analysis, large or small.

### Added
- **Device selector in the ⚙ Config tab**: `Auto (GPU if available)` /
  `Force GPU (CUDA)` / `Force CPU`.
  - `auto` (default) uses the GPU when present, falls back to CPU otherwise.
  - `cpu` is useful when the GPU is busy (e.g. ComfyUI is generating) — it
    hides the GPU from the whole subprocess (torch + onnxruntime + all
    captioners) via `CUDA_VISIBLE_DEVICES`.
  - `cuda` forces GPU but safely falls back to CPU if no GPU is actually
    detected (no crash).
- The chosen device is shown in the progress log (`⚙ Device : CUDA …`).
- Device preference is honored by the analyzer AND the LoRA evaluator.

### Note on CPU+GPU "coupling"
True data-parallel CPU+GPU splitting was considered and rejected: the CPU is
~4× slower than the GPU on these models, so it would only add ~25% throughput
while doubling memory use and adding a lot of fragile scheduling code. Putting
the always-run models on the GPU (this release) is the far better lever.

---

## [v1.0.3] — 2026-05-31

### Fixed
- **Removed the hard 10-minute timeout** that killed analyses of large
  datasets (e.g. 200 photos) even while they were still progressing.
  Replaced with an **inactivity watchdog**: the subprocess is only killed
  after 30 minutes of complete silence (downloads emit progress to stderr,
  so they keep it alive). Applied to both the analyzer and the LoRA evaluator.
- Fixed a latent race where `communicate()` and the progress thread both
  read the subprocess stderr. stdout and stderr are now each read by their
  own dedicated thread.

### Added
- **Live activity ticker during model loading.** The phase line now shows
  elapsed time (`⚙ Loading… — 12s`) and, after 25s, a hint that first-run
  model downloads can take several minutes. A secondary line shows the time
  since the last sign of activity, so the app never looks frozen.
- More granular `STEP` messages in the engine: image count, library import,
  face-model loading, "model ready".
- **Pre-flight warning** when launching JoyCaption / All captioning on 40+
  images: estimates the cost and suggests running WD14 first (fast) then
  JoyCaption on the kept photos (cache avoids redoing work).

---

## [v1.0.2] — 2026-05-31

### Removed
- **Stripped all non-LoRA features.** The app was originally a general
  "AI File Manager" with extra tabs. Removed the **LoRAs management**,
  **Outputs browser**, **Inputs browser**, and **Prompt optimizer (Ollama)**
  tabs and all their code. The tool is now a focused 3-tab app:
  📊 Dataset Analyzer · 📊 Evaluate LoRA · ⚙ Config.
- Removed orphaned module-level helpers (`PROMPT_SYSTEMS`, `list_files`,
  `fmt_size`, `LORA_EXTS`, `IMAGE_EXTS`, `MEDIA_EXTS`).

### Changed
- **ComfyUI Python path is now configurable** in the ⚙ Config tab
  (previously hard-coded). Essential for anyone whose ComfyUI install
  differs from the original author's. Applied hot, no restart needed.
- Config tab now exposes two settings: **ComfyUI python.exe** (file picker)
  and **default datasets folder** (folder picker), instead of the old
  loras/outputs/inputs paths.
- Window title and header renamed to "LoRA Dataset Coach".
- Documentation updated to describe the 3-tab layout.

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
