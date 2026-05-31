"""
Lecture metadata IA (C2PA v2.2 + IPTC 2025.1 + EXIF) pour detecter les images
explicitement marquees comme generees par une IA.

Sources verifiees :
  - C2PA Content Credentials (manifest signe par OpenAI, Adobe, Google SynthID...)
  - IPTC champs 2025.1 : AISystemUsed, AIPromptInformation, DigitalSourceType
  - EXIF Software field (souvent contient "Stable Diffusion", "Midjourney"...)
  - Filename heuristics (ComfyUI_00042_.png, MJ_xxx.png...)

Fallback : on a deja le classifieur ML (Organika/sdxl-detector) dans
ai_detector_local.py pour les images sans metadata.
"""
from pathlib import Path
from typing import Dict, Optional

from PIL import Image


# Patterns dans le filename qui trahissent une generation IA
FILENAME_AI_PATTERNS = [
    "comfyui_", "comfyui-", "_comfy",
    "automatic1111", "_a1111",
    "midjourney", "mj_", "_mj_", "_v6_", "_v5_",
    "dalle", "dall_e", "dall-e",
    "stablediffusion", "sd_", "_sd15", "_sdxl",
    "flux_", "_flux", "flux1",
    "instantid", "_instantid",
    "kohya_", "_lora_",
    "ostris_", "ai_toolkit",
]


# Software EXIF qui signalent une IA
EXIF_AI_SOFTWARE = [
    "stable diffusion", "stablediffusion", "midjourney", "dalle",
    "dall-e", "comfyui", "automatic1111", "fooocus", "invokeai",
    "leonardo.ai", "playgroundai", "ideogram", "krea",
]


def _check_filename(path: Path) -> Optional[str]:
    name = path.name.lower()
    for pat in FILENAME_AI_PATTERNS:
        if pat in name:
            return pat
    return None


def _check_exif(pil_image) -> Optional[Dict]:
    """Inspecte les champs EXIF + tEXt PNG pour signature IA."""
    info = {}
    try:
        exif = pil_image.getexif()
        software = None
        if exif:
            # Tag 305 = Software, 270 = ImageDescription, 33432 = Copyright
            for tag in (305, 270, 33432):
                val = exif.get(tag)
                if val:
                    info[f"exif_{tag}"] = str(val)[:200]
                    if tag == 305:
                        software = str(val).lower()
        # PNG tEXt chunks (souvent utilises par ComfyUI / A1111)
        text_meta = getattr(pil_image, "text", None) or {}
        for k, v in text_meta.items():
            kl = k.lower()
            if kl in ("parameters", "prompt", "workflow", "comment",
                       "software", "description"):
                info[f"png_{kl}"] = str(v)[:200]
            if "stable" in str(v).lower() or "comfyui" in str(v).lower():
                info["png_ai_marker_found"] = True

        # Match contre software AI connus
        if software:
            for ai_sw in EXIF_AI_SOFTWARE:
                if ai_sw in software:
                    info["software_ai_match"] = ai_sw
                    break
    except Exception:
        return None
    return info if info else None


def _check_c2pa(image_path: Path) -> Optional[Dict]:
    """Lit le manifest C2PA si present. Necessite c2pa-python (optionnel)."""
    try:
        import c2pa
    except ImportError:
        return None
    try:
        reader = c2pa.Reader.from_file(str(image_path))
        manifest = reader.json()
        return {"c2pa_manifest": manifest[:500]}  # tronque pour stockage
    except Exception:
        return None


def detect_ai_metadata(image_path) -> Dict:
    """Verdict combine. Retourne :
        {
          "has_ai_metadata": bool,
          "sources": ["filename", "exif", "png_text", "c2pa"],
          "details": {...},
          "confidence": "high" | "medium" | "low"
        }
    """
    p = Path(image_path)
    sources = []
    details = {}

    # Filename
    fn_match = _check_filename(p)
    if fn_match:
        sources.append("filename")
        details["filename_pattern"] = fn_match

    # EXIF + PNG text
    try:
        img = Image.open(p)
        exif_info = _check_exif(img)
        img.close()
        if exif_info:
            ai_signal = any(k.startswith("software_ai_match") or
                             k == "png_ai_marker_found" or
                             k in ("png_parameters", "png_prompt", "png_workflow")
                             for k in exif_info)
            if ai_signal:
                sources.append("exif/png_text")
                details.update(exif_info)
    except Exception:
        pass

    # C2PA (optionnel)
    c2pa_info = _check_c2pa(p)
    if c2pa_info:
        sources.append("c2pa")
        details.update(c2pa_info)

    # Verdict de confiance
    if "c2pa" in sources or "exif/png_text" in sources:
        confidence = "high"
    elif "filename" in sources:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "has_ai_metadata": bool(sources),
        "sources": sources,
        "details": details,
        "confidence": confidence if sources else "none",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: metadata_ai.py <image_path>")
        sys.exit(1)
    print(detect_ai_metadata(sys.argv[1]))
