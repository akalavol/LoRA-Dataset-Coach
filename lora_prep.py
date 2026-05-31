"""
Preparation finale pour entrainement LoRA - MULTI-TARGET.

Supporte plusieurs trainers et modeles cibles :
  - SDXL via Kohya SS GUI (le classique)
  - SD 1.5 via Kohya
  - Flux via ai-toolkit (ostris) ou Kohya
  - Wan 2.2 via musubi-tuner
  - HunyuanVideo via diffusion-pipe
  - OneTrainer (SDXL ou Flux)

Chaque target a sa propre :
  - resolution / bucket strategy
  - structure de dossier
  - format de config (.toml, .yaml, .json)
  - captioner prefere (WD14 tags booru vs caption naturelle)
"""
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image


# ============================================================
# CATALOGUE DES TARGETS
# ============================================================

TARGETS = {
    "sdxl_kohya": {
        "label": "SDXL (Kohya SS GUI)",
        "base_model": "RealVisXL_V5.0_fp16.safetensors",
        "resolutions": [(1024, 1024)],         # carre, c'est l'optimum SDXL
        "crop_strategy": "square_face",        # crop 1:1 centre visage
        "captioner": "wd14",                   # tags booru
        "config_format": "kohya_toml",
        "folder_naming": "kohya",              # 10_persona/
        "default_repeats": 10,
        "default_epochs": 10,
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "trainer_doc_url": "https://github.com/bmaltais/kohya_ss",
    },
    "sd15_kohya": {
        "label": "SD 1.5 (Kohya SS GUI)",
        "base_model": "v1-5-pruned-emaonly.safetensors",
        "resolutions": [(512, 512)],
        "crop_strategy": "square_face",
        "captioner": "wd14",
        "config_format": "kohya_toml",
        "folder_naming": "kohya",
        "default_repeats": 10,
        "default_epochs": 8,
        "network_dim": 16,
        "network_alpha": 8,
        "learning_rate": 1e-4,
        "trainer_doc_url": "https://github.com/bmaltais/kohya_ss",
    },
    "flux_aitoolkit": {
        "label": "Flux (ai-toolkit / ostris)",
        "base_model": "flux1-dev-Q8_0.gguf",
        "resolutions": [(1024, 1024), (1024, 768), (768, 1024)],  # buckets ratio libres
        "crop_strategy": "bucket_face",        # garde des ratios variés
        "captioner": "natural",                # T5 comprend le langage naturel
        "config_format": "aitoolkit_yaml",
        "folder_naming": "flat",               # images/ direct, pas de N_persona
        "default_repeats": None,
        "default_epochs": 12,
        "network_dim": 16,
        "network_alpha": 16,
        "learning_rate": 4e-4,
        "trainer_doc_url": "https://github.com/ostris/ai-toolkit",
    },
    "flux_kohya": {
        "label": "Flux (Kohya, branche sd3)",
        "base_model": "flux1-dev-Q8_0.gguf",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "natural",
        "config_format": "kohya_toml_flux",
        "folder_naming": "kohya",
        "default_repeats": 4,
        "default_epochs": 10,
        "network_dim": 16,
        "network_alpha": 16,
        "learning_rate": 5e-5,
        "trainer_doc_url": "https://github.com/bmaltais/kohya_ss",
    },
    "wan22_musubi": {
        "label": "Wan 2.2 vidéo (musubi-tuner)",
        "base_model": "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        # Wan I2V/T2V utilise des ratios vidéo. Pour un LoRA persona on peut
        # garder du carré 720x720, mais buckets 832x480 (paysage) + 480x832
        # (portrait) couvrent mieux les usages réels.
        "resolutions": [(720, 720), (832, 480), (480, 832)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",                # Wan T5 = langage naturel
        "config_format": "musubi_toml",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 16,
        "network_dim": 32,
        "network_alpha": 32,
        "learning_rate": 2e-4,
        "trainer_doc_url": "https://github.com/kohya-ss/musubi-tuner",
    },
    "hunyuan_diffpipe": {
        "label": "HunyuanVideo (diffusion-pipe)",
        "base_model": "hunyuan-video-t2v-720p",
        "resolutions": [(720, 720), (832, 480), (480, 832)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "diffpipe_toml",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 20,
        "network_dim": 32,
        "network_alpha": 32,
        "learning_rate": 5e-4,
        "trainer_doc_url": "https://github.com/tdrussell/diffusion-pipe",
    },
    "onetrainer_sdxl": {
        "label": "SDXL (OneTrainer)",
        "base_model": "RealVisXL_V5.0_fp16.safetensors",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "wd14",
        "config_format": "onetrainer_hint",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 10,
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "trainer_doc_url": "https://github.com/Nerogar/OneTrainer",
    },

    # ===========================================================
    # SDXL forks (très populaires 2025-2026, base SDXL mais
    # quality boosters obligatoires en début de caption)
    # ===========================================================
    "pony_kohya": {
        "label": "Pony Diffusion XL (Kohya)",
        "base_model": "ponyDiffusionV6XL.safetensors",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "wd14",
        "quality_prefix": "score_9, score_8_up, score_7_up, source_photo",
        "config_format": "kohya_toml",
        "folder_naming": "kohya",
        "default_repeats": 8,
        "default_epochs": 10,
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "category": "image_anime",
        "trainer_doc_url": "https://civitai.com/articles/4348",
    },
    "illustrious_kohya": {
        "label": "Illustrious XL (Kohya)",
        "base_model": "Illustrious-XL-v1.0.safetensors",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "wd14",
        "quality_prefix": "masterpiece, best quality, very aesthetic, absurdres",
        "config_format": "kohya_toml",
        "folder_naming": "kohya",
        "default_repeats": 6,
        "default_epochs": 12,
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "category": "image_anime",
        "trainer_doc_url": "https://civitai.com/models/795765",
    },
    "noobai_kohya": {
        "label": "NoobAI XL (Kohya)",
        "base_model": "noobaiXLNAIXL_vPred10Version.safetensors",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "wd14",
        "quality_prefix": "masterpiece, best quality, newest, absurdres, highres",
        "config_format": "kohya_toml",
        "folder_naming": "kohya",
        "default_repeats": 6,
        "default_epochs": 10,
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "category": "image_anime",
        "trainer_doc_url": "https://civitai.com/models/833294",
    },

    # ===========================================================
    # Modeles DiT/T5 photo modernes
    # ===========================================================
    "sd35_kohya": {
        "label": "SD 3.5 Large (Kohya sd3 branche)",
        "base_model": "sd3.5_large.safetensors",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "natural",
        "config_format": "kohya_toml_sd35",
        "folder_naming": "kohya",
        "default_repeats": 4,
        "default_epochs": 10,
        "network_dim": 16,
        "network_alpha": 16,
        "learning_rate": 4e-5,
        "category": "image_photo",
        "trainer_doc_url": "https://github.com/bmaltais/kohya_ss",
    },
    "hunyuan_dit_kohya": {
        "label": "HunyuanDiT (Kohya)",
        "base_model": "hunyuan-dit-1.2.safetensors",
        "resolutions": [(1024, 1024)],
        "crop_strategy": "square_face",
        "captioner": "natural",
        "config_format": "kohya_toml_hunyuan_dit",
        "folder_naming": "kohya",
        "default_repeats": 6,
        "default_epochs": 12,
        "network_dim": 16,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "category": "image_photo",
        "trainer_doc_url": "https://github.com/Tencent/HunyuanDiT",
    },
    "sana_diffpipe": {
        "label": "Sana (NVIDIA) via diffusion-pipe",
        "base_model": "Sana_1600M_1024px.pth",
        "resolutions": [(1024, 1024), (512, 1024), (1024, 512)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "diffpipe_toml_sana",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 10,
        "network_dim": 16,
        "network_alpha": 16,
        "learning_rate": 1e-4,
        "category": "image_photo",
        "trainer_doc_url": "https://github.com/tdrussell/diffusion-pipe",
    },
    "chroma_aitoolkit": {
        "label": "Chroma (Flux variant) ai-toolkit",
        "base_model": "chroma-unlocked-v37.safetensors",
        "resolutions": [(1024, 1024), (1024, 768), (768, 1024)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "aitoolkit_yaml_chroma",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 10,
        "network_dim": 16,
        "network_alpha": 16,
        "learning_rate": 3e-4,
        "category": "image_photo",
        "trainer_doc_url": "https://github.com/ostris/ai-toolkit",
    },

    # ===========================================================
    # VIDEO (suite)
    # ===========================================================
    "wan21_musubi": {
        "label": "Wan 2.1 vidéo (musubi-tuner)",
        "base_model": "Wan2.1-I2V-14B.safetensors",
        "resolutions": [(720, 720), (832, 480), (480, 832)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "musubi_toml_wan21",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 16,
        "network_dim": 32,
        "network_alpha": 32,
        "learning_rate": 2e-4,
        "category": "video",
        "trainer_doc_url": "https://github.com/kohya-ss/musubi-tuner",
    },
    "ltx_video_diffpipe": {
        "label": "LTX-Video (Lightricks) diffusion-pipe",
        "base_model": "ltx-video-2b-v0.9.safetensors",
        # LTX optimal : 768x512 paysage ou 512x768 portrait
        "resolutions": [(768, 512), (512, 768), (704, 704)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "diffpipe_toml_ltx",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 20,
        "network_dim": 32,
        "network_alpha": 32,
        "learning_rate": 3e-4,
        "category": "video",
        "trainer_doc_url": "https://github.com/Lightricks/LTX-Video",
    },
    "cogvideox_diffpipe": {
        "label": "CogVideoX 5B (Tsinghua) diffusion-pipe",
        "base_model": "CogVideoX-5b",
        # CogVideoX cible : 720x480 (5B) ou 1360x768 (1.5)
        "resolutions": [(720, 480), (480, 720)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "diffpipe_toml_cogvideox",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 15,
        "network_dim": 64,
        "network_alpha": 64,
        "learning_rate": 1e-3,
        "category": "video",
        "trainer_doc_url": "https://github.com/a-r-r-o-w/cogvideox-factory",
    },
    "mochi_diffpipe": {
        "label": "Mochi 1 (Genmo) diffusion-pipe",
        "base_model": "mochi-1-preview",
        "resolutions": [(848, 480), (480, 848)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "diffpipe_toml_mochi",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 20,
        "network_dim": 32,
        "network_alpha": 32,
        "learning_rate": 2e-4,
        "category": "video",
        "trainer_doc_url": "https://github.com/genmoai/mochi",
    },
    "open_sora_diffpipe": {
        "label": "Open-Sora 2.0 (HPC-AI) diffusion-pipe",
        "base_model": "Open-Sora-v2.0",
        "resolutions": [(720, 720), (1280, 720), (720, 1280)],
        "crop_strategy": "bucket_face",
        "captioner": "natural",
        "config_format": "diffpipe_toml_opensora",
        "folder_naming": "flat",
        "default_repeats": None,
        "default_epochs": 18,
        "network_dim": 32,
        "network_alpha": 32,
        "learning_rate": 2e-4,
        "category": "video",
        "trainer_doc_url": "https://github.com/hpcaitech/Open-Sora",
    },
}


# Categories pour grouper dans la GUI
TARGET_CATEGORIES = {
    "image_photo":     "📸 Photo réaliste",
    "image_anime":     "🎨 Anime/Style (SDXL forks)",
    "video":           "🎬 Vidéo",
    "default":         "🖼  Autre",
}


def get_target_category(target_key):
    cfg = TARGETS.get(target_key, {})
    cat = cfg.get("category")
    if cat:
        return cat
    # Heuristique pour les targets historiques sans clef category
    label = cfg.get("label", "").lower()
    if "video" in label or "wan" in label or "hunyuan" in target_key.lower():
        return "video"
    if "anime" in label:
        return "image_anime"
    return "image_photo"


def list_targets():
    """Retourne [(key, label), ...] pour le dropdown GUI."""
    return [(k, v["label"]) for k, v in TARGETS.items()]


# ============================================================
# CROP STRATEGIES
# ============================================================

def crop_face_square(pil_img, face_bbox, target_size=1024, margin_ratio=0.6):
    """Crop carre centre sur le visage."""
    x1, y1, x2, y2 = face_bbox
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    side = max(fw, fh) * (1 + margin_ratio * 2)
    side = min(side, min(pil_img.width, pil_img.height))

    half = side / 2
    cx = max(half, min(pil_img.width - half, cx))
    cy = max(half, min(pil_img.height - half, cy))

    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    crop = pil_img.crop(box)
    if crop.size != (target_size, target_size):
        crop = crop.resize((target_size, target_size), Image.LANCZOS)
    return crop


def crop_face_bucket(pil_img, face_bbox, allowed_resolutions, margin_ratio=0.7):
    """Crop vers la resolution du bucket dont le ratio correspond le mieux a l'image.
    Garde plus de contexte qu'un crop carre pur."""
    w, h = pil_img.size
    img_ratio = w / h

    # Choisit le bucket dont le ratio est le plus proche
    best = min(allowed_resolutions, key=lambda r: abs((r[0] / r[1]) - img_ratio))
    tw, th = best
    target_ratio = tw / th

    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    else:
        cx, cy = w / 2, h / 2
        fw = fh = min(w, h) / 4

    # Cote du crop dans l'image source
    face_max = max(fw, fh) * (1 + margin_ratio * 2)
    # On veut un crop au bon ratio cible
    if target_ratio >= 1:
        crop_w = max(face_max, face_max * target_ratio)
        crop_h = crop_w / target_ratio
    else:
        crop_h = max(face_max, face_max / target_ratio)
        crop_w = crop_h * target_ratio

    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    if crop_w / crop_h > target_ratio:
        crop_w = crop_h * target_ratio
    else:
        crop_h = crop_w / target_ratio

    hw, hh = crop_w / 2, crop_h / 2
    cx = max(hw, min(w - hw, cx))
    cy = max(hh, min(h - hh, cy))
    box = (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))
    crop = pil_img.crop(box)
    if crop.size != (tw, th):
        crop = crop.resize((tw, th), Image.LANCZOS)
    return crop


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def prepare_lora_folder(analysis_data, source_folder, output_folder,
                         persona_name="persona",
                         target="sdxl_kohya",
                         repeats=None,
                         viable_only=True,
                         captioner=None,   # None = celui du target
                         progress_cb=None,
                         quality_prefix_override=None):
    """
    Cree le dossier final pour le trainer choisi.
    Retourne {written, skipped, errors, output_path, target, config_path}.
    """
    if target not in TARGETS:
        raise ValueError(f"Target inconnu : {target}. Liste : {list(TARGETS.keys())}")

    cfg = TARGETS[target]
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    resolutions = cfg["resolutions"]
    crop_strategy = cfg["crop_strategy"]
    use_captioner = captioner or cfg["captioner"]
    repeats = repeats or cfg.get("default_repeats") or 10

    # Structure du dossier images selon folder_naming
    if cfg["folder_naming"] == "kohya":
        images_folder = output_folder / f"{repeats}_{persona_name}"
    else:
        images_folder = output_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    written, skipped, errors = 0, 0, []
    imgs = analysis_data.get("images", []) or []
    total = len(imgs)

    for idx, img in enumerate(imgs):
        if progress_cb:
            try:
                progress_cb(idx + 1, total, img.get("name", "?"))
            except Exception:
                pass

        if viable_only and img.get("lora_viable") == "no":
            skipped += 1
            continue

        src_path = Path(img.get("path", ""))
        if not src_path.is_file():
            errors.append(f"introuvable : {src_path.name}")
            continue

        try:
            pil = Image.open(src_path).convert("RGB")
        except Exception as e:
            errors.append(f"{src_path.name}: open fail ({e})")
            continue

        face_bbox = img.get("_face_bbox")
        try:
            if crop_strategy == "square_face":
                target_size = resolutions[0][0]
                cropped = crop_face_square(pil, face_bbox, target_size=target_size) if face_bbox \
                    else _center_crop_resize(pil, resolutions[0])
            elif crop_strategy == "bucket_face":
                cropped = crop_face_bucket(pil, face_bbox, resolutions)
            else:
                cropped = _center_crop_resize(pil, resolutions[0])
        except Exception as e:
            errors.append(f"{src_path.name}: crop fail ({e})")
            continue

        out_name = f"{persona_name}_{written + 1:03d}.png"
        out_path = images_folder / out_name
        try:
            cropped.save(out_path, "PNG")
        except Exception as e:
            errors.append(f"{src_path.name}: save fail ({e})")
            continue

        # Caption : selon use_captioner + quality prefix du target
        q_prefix = quality_prefix_override
        if q_prefix is None:
            q_prefix = cfg.get("quality_prefix")
        caption = _build_caption(img, persona_name, use_captioner,
                                  quality_prefix=q_prefix)
        (images_folder / f"{out_name[:-4]}.txt").write_text(caption, encoding="utf-8")
        written += 1

    # Genere le fichier de config selon le trainer
    config_path = _write_trainer_config(
        target=target, cfg=cfg, persona_name=persona_name,
        repeats=repeats, output_folder=output_folder,
        images_folder=images_folder, written=written,
    )

    # README
    readme = _generate_readme(target=target, cfg=cfg, persona_name=persona_name,
                               repeats=repeats, written=written, skipped=skipped,
                               captioner=use_captioner,
                               output_folder=output_folder, images_folder=images_folder)
    (output_folder / "README.txt").write_text(readme, encoding="utf-8")

    return {
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "output_path": str(output_folder),
        "target": target,
        "config_path": str(config_path) if config_path else None,
    }


def _center_crop_resize(pil, target_size):
    """Fallback : crop centre + resize."""
    if isinstance(target_size, tuple):
        tw, th = target_size
    else:
        tw = th = target_size
    w, h = pil.size
    target_ratio = tw / th
    img_ratio = w / h
    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        pil = pil.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        pil = pil.crop((0, top, w, top + new_h))
    return pil.resize((tw, th), Image.LANCZOS)


def _build_caption(img, persona_name, use_captioner, quality_prefix=None):
    """Construit la caption finale.

    Ordre du caption :
        [quality_prefix,] persona_name, [caption_body]

    quality_prefix (ex Pony : "score_9, score_8_up, score_7_up, source_photo")
    s'insere AVANT le trigger word — c'est la convention des SDXL forks.
    """
    if use_captioner == "wd14":
        body = img.get("wd14_tags") or ""
    elif use_captioner == "natural":
        body = (img.get("joycaption")
                 or img.get("natural_caption")
                 or "")
        if not body:
            tags = img.get("wd14_tags") or ""
            if tags:
                body = f"a photo of {persona_name}, " + tags.replace(",", " and").strip()
    else:
        body = ""

    # Assemblage : quality_prefix, persona_name, body
    parts = []
    if quality_prefix:
        parts.append(quality_prefix.strip().rstrip(","))
    # Persona name (trigger word) -- seulement si pas deja dans le body
    if not body or persona_name.lower() not in body.lower():
        parts.append(persona_name)
    if body:
        parts.append(body)

    return ", ".join(p for p in parts if p) or persona_name


# ============================================================
# GENERATEURS DE CONFIGS PAR TRAINER
# ============================================================

def _write_trainer_config(target, cfg, persona_name, repeats, output_folder, images_folder, written):
    """Genere le fichier config approprie."""
    fmt = cfg["config_format"]
    if fmt == "kohya_toml":
        path = output_folder / "kohya_config.toml"
        path.write_text(_kohya_toml(cfg, persona_name, repeats, output_folder, sd15=("sd15" in target)),
                        encoding="utf-8")
        return path
    elif fmt == "kohya_toml_flux":
        path = output_folder / "kohya_flux_config.toml"
        path.write_text(_kohya_flux_toml(cfg, persona_name, repeats, output_folder),
                        encoding="utf-8")
        return path
    elif fmt == "aitoolkit_yaml":
        path = output_folder / "ai_toolkit_config.yaml"
        path.write_text(_aitoolkit_yaml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "musubi_toml":
        path = output_folder / "musubi_dataset.toml"
        path.write_text(_musubi_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        # Et le shell de lancement
        sh = output_folder / "launch_musubi.bat"
        sh.write_text(_musubi_launch_bat(persona_name, output_folder), encoding="utf-8")
        return path
    elif fmt == "diffpipe_toml":
        path = output_folder / "diffusion_pipe_config.toml"
        path.write_text(_diffpipe_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "onetrainer_hint":
        return None
    # ===== SDXL forks (Pony / Illustrious / NoobAI) =====
    # Ils utilisent le meme format Kohya SDXL standard, deja gere par kohya_toml
    # ===== Nouveaux formats =====
    elif fmt == "kohya_toml_sd35":
        path = output_folder / "kohya_sd35_config.toml"
        path.write_text(_kohya_sd35_toml(cfg, persona_name, output_folder),
                        encoding="utf-8")
        return path
    elif fmt == "kohya_toml_hunyuan_dit":
        path = output_folder / "kohya_hunyuan_dit_config.toml"
        path.write_text(_kohya_hunyuan_dit_toml(cfg, persona_name, output_folder),
                        encoding="utf-8")
        return path
    elif fmt == "aitoolkit_yaml_chroma":
        path = output_folder / "ai_toolkit_chroma_config.yaml"
        path.write_text(_aitoolkit_chroma_yaml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "diffpipe_toml_sana":
        path = output_folder / "diffusion_pipe_sana.toml"
        path.write_text(_diffpipe_sana_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "musubi_toml_wan21":
        path = output_folder / "musubi_wan21_dataset.toml"
        path.write_text(_musubi_wan21_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        sh = output_folder / "launch_musubi_wan21.bat"
        sh.write_text(_musubi_wan21_launch_bat(persona_name, output_folder), encoding="utf-8")
        return path
    elif fmt == "diffpipe_toml_ltx":
        path = output_folder / "diffusion_pipe_ltx.toml"
        path.write_text(_diffpipe_ltx_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "diffpipe_toml_cogvideox":
        path = output_folder / "cogvideox_factory_config.yaml"
        path.write_text(_cogvideox_factory_yaml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "diffpipe_toml_mochi":
        path = output_folder / "diffusion_pipe_mochi.toml"
        path.write_text(_diffpipe_mochi_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    elif fmt == "diffpipe_toml_opensora":
        path = output_folder / "diffusion_pipe_opensora.toml"
        path.write_text(_diffpipe_opensora_toml(cfg, persona_name, output_folder, images_folder),
                        encoding="utf-8")
        return path
    return None


def _kohya_toml(cfg, persona_name, repeats, output_folder, sd15=False):
    res = cfg["resolutions"][0]
    base = cfg["base_model"]
    ckpt_dir = "C:/AI/ComfyUI-future/ComfyUI_windows_portable/ComfyUI/models/checkpoints"
    return f"""# Config Kohya {('SD 1.5' if sd15 else 'SDXL')} - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Charger via Kohya SS GUI : LoRA > Tools > Load config
[model]
v2 = false
v_parameterization = false
pretrained_model_name_or_path = "{ckpt_dir}/{base}"

[folders]
output_dir = "{str(output_folder / 'output').replace(chr(92), '/')}"
logging_dir = "{str(output_folder / 'logs').replace(chr(92), '/')}"
train_data_dir = "{str(output_folder).replace(chr(92), '/')}"

[training]
output_name = "{persona_name}_lora"
save_model_as = "safetensors"
resolution = "{res[0]},{res[1]}"
batch_size = 1
max_train_epochs = {cfg['default_epochs']}
save_every_n_epochs = 1
network_module = "networks.lora"
network_dim = {cfg['network_dim']}
network_alpha = {cfg['network_alpha']}
learning_rate = {cfg['learning_rate']}
unet_lr = {cfg['learning_rate']}
text_encoder_lr = {cfg['learning_rate'] / 2}
lr_scheduler = "cosine_with_restarts"
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
save_precision = "bf16"
seed = 42
cache_latents = true
gradient_checkpointing = true
xformers = true
"""


def _kohya_flux_toml(cfg, persona_name, repeats, output_folder):
    res = cfg["resolutions"][0]
    return f"""# Config Kohya Flux LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Branche kohya-ss/sd-scripts sd3 - voir trainer_doc_url
[model]
pretrained_model_name_or_path = "flux1-dev.safetensors"  # ADAPTE
ae = "ae.safetensors"
clip_l = "clip_l.safetensors"
t5xxl = "t5xxl_fp16.safetensors"

[folders]
output_dir = "{str(output_folder / 'output').replace(chr(92), '/')}"
train_data_dir = "{str(output_folder).replace(chr(92), '/')}"

[training]
output_name = "{persona_name}_flux_lora"
save_model_as = "safetensors"
resolution = "{res[0]},{res[1]}"
batch_size = 1
max_train_epochs = {cfg['default_epochs']}
network_module = "networks.lora_flux"
network_dim = {cfg['network_dim']}
network_alpha = {cfg['network_alpha']}
learning_rate = {cfg['learning_rate']}
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
save_precision = "bf16"
gradient_checkpointing = true
"""


def _aitoolkit_yaml(cfg, persona_name, output_folder, images_folder):
    res = cfg["resolutions"][0]
    return f"""# ai-toolkit (ostris) Flux LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Lance via : python run.py {persona_name}.yaml
job: extension
config:
  name: "{persona_name}_flux_lora"
  process:
    - type: 'sd_trainer'
      training_folder: "{str(output_folder / 'output').replace(chr(92), '/')}"
      device: cuda:0
      trigger_word: "{persona_name}"
      network:
        type: "lora"
        linear: {cfg['network_dim']}
        linear_alpha: {cfg['network_alpha']}
      save:
        dtype: bf16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "{str(images_folder).replace(chr(92), '/')}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [{res[0]}, {res[1]}]
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: {cfg['learning_rate']}
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
"""


def _musubi_toml(cfg, persona_name, output_folder, images_folder):
    return f"""# Dataset config pour musubi-tuner (Wan 2.2 LoRA)
# Lance via : accelerate launch wan_train_network.py --dataset_config dataset.toml ...
# Genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
[general]
resolution = [832, 480]    # bucket par defaut, le tuner pioche automatiquement
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{str(images_folder).replace(chr(92), '/')}"
cache_directory = "{str(output_folder / 'cache').replace(chr(92), '/')}"
num_repeats = 4
"""


def _musubi_launch_bat(persona_name, output_folder):
    return f"""@echo off
:: Lancement musubi-tuner pour {persona_name} - ADAPTE le chemin musubi
:: Doc : https://github.com/kohya-ss/musubi-tuner

set MUSUBI_DIR=C:\\AI\\musubi-tuner

cd /d %MUSUBI_DIR%
accelerate launch wan_train_network.py ^
  --dataset_config "{output_folder}\\musubi_dataset.toml" ^
  --task t2v-14B ^
  --dit "C:\\AI\\ComfyUI-future\\ComfyUI_windows_portable\\ComfyUI\\models\\unet\\Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf" ^
  --output_dir "{output_folder}\\output" ^
  --output_name {persona_name}_wan_lora ^
  --save_every_n_epochs 1 ^
  --max_train_epochs 16 ^
  --network_module networks.lora_wan ^
  --network_dim 32 ^
  --network_alpha 32 ^
  --learning_rate 2e-4 ^
  --optimizer_type AdamW8bit ^
  --mixed_precision bf16 ^
  --gradient_checkpointing
pause
"""


def _diffpipe_toml(cfg, persona_name, output_folder, images_folder):
    return f"""# diffusion-pipe config HunyuanVideo LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# https://github.com/tdrussell/diffusion-pipe
output_dir = '{str(output_folder / 'output').replace(chr(92), '/')}'
dataset = 'dataset.toml'  # cree ce fichier dans le dossier diffusion-pipe
epochs = {cfg['default_epochs']}
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 4
gradient_clipping = 1.0
warmup_steps = 100

[model]
type = 'hunyuan-video'
transformer_path = 'C:/AI/hunyuan/transformer'
vae_path = 'C:/AI/hunyuan/vae'
llm_path = 'C:/AI/hunyuan/llm'
clip_path = 'C:/AI/hunyuan/clip'
dtype = 'bfloat16'
transformer_dtype = 'float8'
timestep_sample_method = 'logit_normal'

[adapter]
type = 'lora'
rank = {cfg['network_dim']}
dtype = 'bfloat16'

[optimizer]
type = 'adamw_optimi'
lr = {cfg['learning_rate']}
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8
"""


def _generate_readme(target, cfg, persona_name, repeats, written, skipped, captioner,
                      output_folder, images_folder):
    return f"""LoRA TRAINING DATASET - {persona_name}
{'='*60}
Target : {cfg['label']}
Genere le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Photos exportees : {written}   |   Exclues : {skipped}
Resolution(s) : {cfg['resolutions']}
Captioner : {captioner}
Repeats : {repeats}
Trainer doc : {cfg['trainer_doc_url']}

STRUCTURE
---------
{Path(images_folder).name}/
    {persona_name}_001.png
    {persona_name}_001.txt   (caption avec trigger word « {persona_name} »)
    ...
{('kohya_config.toml' if 'kohya' in cfg['config_format'] else
  'ai_toolkit_config.yaml' if 'aitoolkit' in cfg['config_format'] else
  'musubi_dataset.toml + launch_musubi.bat' if 'musubi' in cfg['config_format'] else
  'diffusion_pipe_config.toml' if 'diffpipe' in cfg['config_format'] else
  '(pas de config auto - se configure dans la GUI OneTrainer)')}

TRIGGER WORD : {persona_name}
Pour invoquer le LoRA, ajoute « {persona_name} » dans tes prompts.

LANCEMENT
---------
{_launch_instructions(target, cfg, persona_name, output_folder)}

CONSEILS
--------
- Epochs : commence aux defaults ({cfg['default_epochs']}), regarde les samples chaque epoch
- Overfit (LoRA = photocopie) : baisse network_dim ou augmente caption_dropout_rate
- Underfit (visage approximatif) : monte epochs ou network_dim ({cfg['network_dim']} -> {cfg['network_dim']*2})
- Resolution(s) : {cfg['resolutions']} (multi-bucket si plusieurs)
"""


# ============================================================
# NOUVEAUX FORMATS DE CONFIG (SDXL forks + DiT/T5 photo + video)
# ============================================================

def _kohya_sd35_toml(cfg, persona_name, output_folder):
    """SD 3.5 Large via Kohya branche sd3."""
    res = cfg["resolutions"][0]
    return f"""# Config Kohya SD 3.5 Large LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# IMPORTANT : utilise la branche sd3 de kohya-ss/sd-scripts
# git clone -b sd3 https://github.com/kohya-ss/sd-scripts
[model]
pretrained_model_name_or_path = "sd3.5_large.safetensors"  # ADAPTE chemin
clip_l = "clip_l.safetensors"
clip_g = "clip_g.safetensors"
t5xxl = "t5xxl_fp16.safetensors"

[folders]
output_dir = "{str(output_folder / 'output').replace(chr(92), '/')}"
train_data_dir = "{str(output_folder).replace(chr(92), '/')}"

[training]
output_name = "{persona_name}_sd35_lora"
save_model_as = "safetensors"
resolution = "{res[0]},{res[1]}"
batch_size = 1
max_train_epochs = {cfg['default_epochs']}
network_module = "networks.lora_sd3"
network_dim = {cfg['network_dim']}
network_alpha = {cfg['network_alpha']}
learning_rate = {cfg['learning_rate']}
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
save_precision = "bf16"
gradient_checkpointing = true
cache_latents = true
weighting_scheme = "logit_normal"
"""


def _kohya_hunyuan_dit_toml(cfg, persona_name, output_folder):
    """HunyuanDiT via Kohya branche dedie."""
    res = cfg["resolutions"][0]
    return f"""# Config Kohya HunyuanDiT LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Voir https://github.com/Tencent/HunyuanDiT pour les chemins exacts
[model]
pretrained_model_name_or_path = "C:/AI/models/hunyuan-dit-1.2.safetensors"

[folders]
output_dir = "{str(output_folder / 'output').replace(chr(92), '/')}"
train_data_dir = "{str(output_folder).replace(chr(92), '/')}"

[training]
output_name = "{persona_name}_hunyuan_dit_lora"
save_model_as = "safetensors"
resolution = "{res[0]},{res[1]}"
batch_size = 1
max_train_epochs = {cfg['default_epochs']}
network_module = "networks.lora_hunyuan_dit"
network_dim = {cfg['network_dim']}
network_alpha = {cfg['network_alpha']}
learning_rate = {cfg['learning_rate']}
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
gradient_checkpointing = true
"""


def _aitoolkit_chroma_yaml(cfg, persona_name, output_folder, images_folder):
    """Chroma (Flux variant uncensored) via ai-toolkit."""
    res = cfg["resolutions"][0]
    return f"""# ai-toolkit Chroma LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Chroma = Flux-Schnell variant uncensored, base FLUX.1
job: extension
config:
  name: "{persona_name}_chroma_lora"
  process:
    - type: 'sd_trainer'
      training_folder: "{str(output_folder / 'output').replace(chr(92), '/')}"
      device: cuda:0
      trigger_word: "{persona_name}"
      network:
        type: "lora"
        linear: {cfg['network_dim']}
        linear_alpha: {cfg['network_alpha']}
      save:
        dtype: bf16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "{str(images_folder).replace(chr(92), '/')}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [{res[0]}, {res[1]}]
      train:
        batch_size: 1
        steps: 2500
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: {cfg['learning_rate']}
        dtype: bf16
      model:
        name_or_path: "lodestones/Chroma"
        is_flux: true
        quantize: true
"""


def _diffpipe_sana_toml(cfg, persona_name, output_folder, images_folder):
    """Sana (NVIDIA) via diffusion-pipe."""
    return f"""# diffusion-pipe Sana LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Sana = transformer image NVIDIA, super rapide grace au Linear DiT
output_dir = '{str(output_folder / 'output').replace(chr(92), '/')}'
dataset = '{str(output_folder / 'dataset_sana.toml').replace(chr(92), '/')}'
epochs = {cfg['default_epochs']}
micro_batch_size_per_gpu = 2
gradient_accumulation_steps = 1
pipeline_stages = 1

[model]
type = 'sana'
transformer_path = 'C:/AI/models/sana_1600M_1024px'
text_encoder_path = 'gemma-2b-it'  # Sana utilise Gemma
vae_path = 'dc_ae_f32c32_sana_1.0'
dtype = 'bfloat16'
transformer_dtype = 'bfloat16'

[adapter]
type = 'lora'
rank = {cfg['network_dim']}
dtype = 'bfloat16'

[optimizer]
type = 'adamw_optimi'
lr = {cfg['learning_rate']}
betas = [0.9, 0.99]
weight_decay = 0.01

# Cree aussi dataset_sana.toml avec :
# [[directory]]
# path = '{str(images_folder).replace(chr(92), '/')}'
# resolutions = [1024]
# enable_ar_bucket = true
# min_ar = 0.5
# max_ar = 2.0
"""


def _musubi_wan21_toml(cfg, persona_name, output_folder, images_folder):
    """Wan 2.1 via musubi-tuner."""
    return f"""# Dataset config musubi-tuner Wan 2.1 LoRA
# Genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
[general]
resolution = [832, 480]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{str(images_folder).replace(chr(92), '/')}"
cache_directory = "{str(output_folder / 'cache').replace(chr(92), '/')}"
num_repeats = 4
"""


def _musubi_wan21_launch_bat(persona_name, output_folder):
    return f"""@echo off
:: Lancement musubi-tuner Wan 2.1 pour {persona_name}
set MUSUBI_DIR=C:\\AI\\musubi-tuner
cd /d %MUSUBI_DIR%
accelerate launch wan_train_network.py ^
  --dataset_config "{output_folder}\\musubi_wan21_dataset.toml" ^
  --task i2v-14B ^
  --dit "C:\\AI\\models\\Wan2.1-I2V-14B.safetensors" ^
  --output_dir "{output_folder}\\output" ^
  --output_name {persona_name}_wan21_lora ^
  --save_every_n_epochs 1 ^
  --max_train_epochs 16 ^
  --network_module networks.lora_wan ^
  --network_dim 32 ^
  --network_alpha 32 ^
  --learning_rate 2e-4 ^
  --optimizer_type AdamW8bit ^
  --mixed_precision bf16 ^
  --gradient_checkpointing
pause
"""


def _diffpipe_ltx_toml(cfg, persona_name, output_folder, images_folder):
    """LTX-Video via diffusion-pipe."""
    return f"""# diffusion-pipe LTX-Video LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# LTX-Video = generation video temps reel (Lightricks)
output_dir = '{str(output_folder / 'output').replace(chr(92), '/')}'
dataset = '{str(output_folder / 'dataset_ltx.toml').replace(chr(92), '/')}'
epochs = {cfg['default_epochs']}
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4
pipeline_stages = 1

[model]
type = 'ltx-video'
diffusers_path = 'C:/AI/models/LTX-Video'
dtype = 'bfloat16'
transformer_dtype = 'float8'
load_dtype = 'bfloat16'

[adapter]
type = 'lora'
rank = {cfg['network_dim']}
dtype = 'bfloat16'

[optimizer]
type = 'adamw_optimi'
lr = {cfg['learning_rate']}
betas = [0.9, 0.99]
weight_decay = 0.01

# Cree dataset_ltx.toml avec :
# [[directory]]
# path = '{str(images_folder).replace(chr(92), '/')}'
# num_repeats = 4
# resolutions = [[768, 512]]
# frame_buckets = [1, 17, 33]  # video frames si tu as des clips
# enable_ar_bucket = true
"""


def _cogvideox_factory_yaml(cfg, persona_name, output_folder, images_folder):
    """CogVideoX via cogvideox-factory (a-r-r-o-w)."""
    return f"""# CogVideoX 5B LoRA training - cogvideox-factory
# Genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Repo : https://github.com/a-r-r-o-w/cogvideox-factory
#
# Pas un fichier yaml standalone mais un set d'arguments :
# bash train_text_to_video_lora.sh
#
# Variables a definir :
MODEL_PATH="THUDM/CogVideoX-5b"
DATASET_PATH="{str(images_folder).replace(chr(92), '/')}"
OUTPUT_PATH="{str(output_folder / 'output').replace(chr(92), '/')}"
ID_TOKEN="{persona_name}"
LR={cfg['learning_rate']}
EPOCHS={cfg['default_epochs']}
LORA_RANK={cfg['network_dim']}
LORA_ALPHA={cfg['network_alpha']}

# Lance :
# accelerate launch --config_file accelerate_configs/uncompiled_1.yaml \\
#   training/cogvideox_text_to_video_lora.py \\
#   --pretrained_model_name_or_path $MODEL_PATH \\
#   --instance_data_root $DATASET_PATH \\
#   --output_dir $OUTPUT_PATH \\
#   --id_token $ID_TOKEN \\
#   --lora_rank $LORA_RANK \\
#   --lora_alpha $LORA_ALPHA \\
#   --learning_rate $LR \\
#   --max_train_epochs $EPOCHS \\
#   --train_batch_size 1 \\
#   --mixed_precision bf16 \\
#   --gradient_checkpointing \\
#   --enable_tiling \\
#   --enable_slicing
"""


def _diffpipe_mochi_toml(cfg, persona_name, output_folder, images_folder):
    """Mochi 1 via diffusion-pipe."""
    return f"""# diffusion-pipe Mochi 1 LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
output_dir = '{str(output_folder / 'output').replace(chr(92), '/')}'
dataset = '{str(output_folder / 'dataset_mochi.toml').replace(chr(92), '/')}'
epochs = {cfg['default_epochs']}
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4
pipeline_stages = 1

[model]
type = 'mochi'
diffusers_path = 'C:/AI/models/mochi-1-preview'
dtype = 'bfloat16'
transformer_dtype = 'float8'

[adapter]
type = 'lora'
rank = {cfg['network_dim']}
dtype = 'bfloat16'

[optimizer]
type = 'adamw_optimi'
lr = {cfg['learning_rate']}
betas = [0.9, 0.99]
weight_decay = 0.01

# dataset_mochi.toml :
# [[directory]]
# path = '{str(images_folder).replace(chr(92), '/')}'
# resolutions = [[848, 480]]
# frame_buckets = [1, 31, 61]
"""


def _diffpipe_opensora_toml(cfg, persona_name, output_folder, images_folder):
    """Open-Sora 2.0 via diffusion-pipe."""
    return f"""# diffusion-pipe Open-Sora 2.0 LoRA - genere {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Note : Open-Sora 2.0 a son trainer natif aussi (recommande pour gros datasets)
# https://github.com/hpcaitech/Open-Sora/tree/main/scripts
output_dir = '{str(output_folder / 'output').replace(chr(92), '/')}'
dataset = '{str(output_folder / 'dataset_opensora.toml').replace(chr(92), '/')}'
epochs = {cfg['default_epochs']}
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4

[model]
type = 'open-sora'
diffusers_path = 'C:/AI/models/Open-Sora-v2.0'
dtype = 'bfloat16'

[adapter]
type = 'lora'
rank = {cfg['network_dim']}

[optimizer]
type = 'adamw_optimi'
lr = {cfg['learning_rate']}

# dataset_opensora.toml :
# [[directory]]
# path = '{str(images_folder).replace(chr(92), '/')}'
# resolutions = [[720, 720], [1280, 720]]
# frame_buckets = [1, 25, 49, 97]
"""


def _launch_instructions(target, cfg, persona_name, output_folder):
    fmt = cfg["config_format"]
    if fmt == "kohya_toml" or fmt == "kohya_toml_flux":
        return (f"1. Lance Kohya SS GUI\n"
                f"2. Onglet LoRA > Tools > Load config > {fmt == 'kohya_toml_flux' and 'kohya_flux_config.toml' or 'kohya_config.toml'}\n"
                f"3. Verifie le chemin du modele de base\n"
                f"4. Train model\n"
                f"5. Sortie : output/{persona_name}_lora.safetensors")
    if fmt == "aitoolkit_yaml":
        return ("1. Installe ai-toolkit (git clone + pip install)\n"
                f"2. Copie ai_toolkit_config.yaml dans ai-toolkit/config/\n"
                f"3. python run.py config/ai_toolkit_config.yaml\n"
                f"4. Sortie : output/{persona_name}_flux_lora/")
    if fmt == "musubi_toml":
        return ("1. Installe musubi-tuner (git clone + pip install)\n"
                "2. Edite launch_musubi.bat : verifie MUSUBI_DIR et --dit path\n"
                "3. Double-clic launch_musubi.bat\n"
                f"4. Sortie : output/{persona_name}_wan_lora.safetensors")
    if fmt == "diffpipe_toml":
        return ("1. Installe diffusion-pipe (git clone)\n"
                "2. Edite diffusion_pipe_config.toml : adapte transformer_path etc.\n"
                "3. deepspeed --num_gpus=1 train.py --config diffusion_pipe_config.toml\n"
                f"4. Sortie : output/")
    if fmt == "onetrainer_hint":
        return ("1. Lance OneTrainer\n"
                "2. Charge un preset SDXL LoRA\n"
                f"3. Concepts > Add > pointe vers images/\n"
                "4. Le trigger word est deja dans les captions\n"
                "5. Train")
    if fmt == "kohya_toml_sd35":
        return ("1. Installe kohya-ss/sd-scripts branche sd3 :\n"
                "   git clone -b sd3 https://github.com/kohya-ss/sd-scripts\n"
                "2. Verifie les chemins clip_l / clip_g / t5xxl dans kohya_sd35_config.toml\n"
                "3. accelerate launch sd3_train_network.py --config kohya_sd35_config.toml\n"
                f"4. Sortie : output/{persona_name}_sd35_lora.safetensors")
    if fmt == "kohya_toml_hunyuan_dit":
        return ("1. Installe le trainer HunyuanDiT (Tencent/HunyuanDiT)\n"
                "2. Adapte les chemins dans kohya_hunyuan_dit_config.toml\n"
                "3. python hydit_train_network.py --config kohya_hunyuan_dit_config.toml\n"
                f"4. Sortie : output/{persona_name}_hunyuan_dit_lora.safetensors")
    if fmt == "aitoolkit_yaml_chroma":
        return ("1. Installe ai-toolkit (git clone + pip install)\n"
                "2. Copie ai_toolkit_chroma_config.yaml dans ai-toolkit/config/\n"
                "3. python run.py config/ai_toolkit_chroma_config.yaml\n"
                f"4. Sortie : output/{persona_name}_chroma_lora/")
    if fmt == "diffpipe_toml_sana":
        return ("1. Installe diffusion-pipe (git clone)\n"
                "2. Telecharge Sana 1600M 1024px sur HuggingFace\n"
                "3. Adapte transformer_path / text_encoder_path / vae_path\n"
                "4. Cree aussi dataset_sana.toml (template dans le .toml principal)\n"
                "5. deepspeed --num_gpus=1 train.py --config diffusion_pipe_sana.toml\n"
                f"6. Sortie : output/")
    if fmt == "musubi_toml_wan21":
        return ("1. Installe musubi-tuner\n"
                "2. Edite launch_musubi_wan21.bat : verifie MUSUBI_DIR et --dit\n"
                "3. Double-clic launch_musubi_wan21.bat\n"
                f"4. Sortie : output/{persona_name}_wan21_lora.safetensors")
    if fmt == "diffpipe_toml_ltx":
        return ("1. Installe diffusion-pipe\n"
                "2. Telecharge LTX-Video 0.9 (Lightricks)\n"
                "3. Adapte diffusers_path dans diffusion_pipe_ltx.toml\n"
                "4. Cree dataset_ltx.toml (template dans le .toml principal)\n"
                "5. deepspeed --num_gpus=1 train.py --config diffusion_pipe_ltx.toml\n"
                "Note : LTX peut entrainer en image-only OR video frames.\n"
                "Pour video, ajoute des clips MP4 dans le dossier images/")
    if fmt == "diffpipe_toml_cogvideox":
        return ("1. Installe cogvideox-factory : git clone https://github.com/a-r-r-o-w/cogvideox-factory\n"
                "2. Edite les variables dans cogvideox_factory_config.yaml\n"
                "3. accelerate launch training/cogvideox_text_to_video_lora.py (voir commentaires)\n"
                "Note : 24 Go VRAM mini recommande pour CogVideoX-5b")
    if fmt == "diffpipe_toml_mochi":
        return ("1. Installe diffusion-pipe\n"
                "2. Telecharge mochi-1-preview (Genmo)\n"
                "3. Adapte diffusers_path + cree dataset_mochi.toml\n"
                "4. deepspeed --num_gpus=1 train.py --config diffusion_pipe_mochi.toml\n"
                "Note : Mochi est tres lourd. Vraie utilite : etyle/concept, persona moins ideal")
    if fmt == "diffpipe_toml_opensora":
        return ("1. Installe diffusion-pipe OU le trainer natif Open-Sora (hpcaitech/Open-Sora)\n"
                "2. Le trainer natif gere mieux les datasets larges\n"
                "3. Cree dataset_opensora.toml (template dans le .toml principal)\n"
                "4. deepspeed --num_gpus=1 train.py --config diffusion_pipe_opensora.toml")
    return ""
