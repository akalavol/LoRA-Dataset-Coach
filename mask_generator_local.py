"""
Generateur de masques sujet (alpha matte) - reutilise BriaRMBG-1.4 (le
modele installe avec ComfyUI-BRIA_AI-RMBG ou telecharge depuis HF).

Pour OneTrainer : creer un fichier <image>-masklabel.png en noir/blanc
(sujet blanc, fond noir) -> OneTrainer focalise la loss sur le sujet.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize


# Emplacements possibles du module briarmbg.py
BRIA_NODE_DIRS = [
    r"C:\AI\ComfyUI-future\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-BRIA_AI-RMBG",
    r"C:\AI\ComfyUI-Zluda\custom_nodes\ComfyUI-BRIA_AI-RMBG",
]


def _ensure_briarmbg_importable():
    """Ajoute le dossier custom_node au sys.path pour importer briarmbg.BriaRMBG."""
    for d in BRIA_NODE_DIRS:
        if Path(d).is_dir() and (Path(d) / "briarmbg.py").is_file():
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
    raise RuntimeError(
        "briarmbg.py introuvable. Installe ComfyUI-BRIA_AI-RMBG dans custom_nodes/"
    )


def _ensure_model_downloaded(node_dir):
    """Cherche RMBG-1.4/model.pth, telecharge depuis HF si absent."""
    target = Path(node_dir) / "RMBG-1.4" / "model.pth"
    if target.is_file() and target.stat().st_size > 100_000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"STEP Telechargement RMBG-1.4 (~176 Mo, une seule fois)...",
          file=sys.stderr, flush=True)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="briaai/RMBG-1.4",
            filename="model.pth",
            local_dir=str(target.parent),
        )
        # Renomme si besoin
        if Path(downloaded) != target:
            Path(downloaded).rename(target)
    except Exception as e:
        raise RuntimeError(f"Echec telechargement RMBG-1.4 : {e}")
    return target


class MaskGenerator:
    """Wrapper BriaRMBG. Genere un masque alpha (PIL.Image L) pour une image."""

    def __init__(self):
        node_dir = _ensure_briarmbg_importable()
        model_path = _ensure_model_downloaded(node_dir)

        from briarmbg import BriaRMBG
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"STEP Chargement RMBG-1.4 ({self.device})...",
              file=sys.stderr, flush=True)
        self.net = BriaRMBG()
        self.net.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def generate_mask(self, pil_image, threshold=None):
        """Retourne un PIL.Image en mode "L" (grayscale) representant le masque.
        Blanc = sujet, noir = fond. Format attendu par OneTrainer.

        Si threshold est fourni (0.0-1.0), binarise le masque (sujet pur blanc
        ou fond pur noir, plus de gradients alpha)."""
        orig = pil_image.convert("RGB")
        w, h = orig.size

        # Resize 1024x1024 pour le modele
        im = orig.resize((1024, 1024), Image.BILINEAR)
        im_np = np.array(im)
        t = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        t = normalize(t, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        if self.device == "cuda":
            t = t.cuda()

        result = self.net(t)
        result = torch.squeeze(
            F.interpolate(result[0][0], size=(h, w), mode="bilinear"), 0
        )
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi + 1e-8)
        arr = (result * 255).cpu().data.numpy().astype(np.uint8)
        # Squeeze pour passer en 2D
        arr2d = np.squeeze(arr)
        mask = Image.fromarray(arr2d, mode="L")

        if threshold is not None:
            arr_bw = (arr2d >= int(threshold * 255)).astype(np.uint8) * 255
            mask = Image.fromarray(arr_bw, mode="L")

        return mask


def generate_masks_for_folder(folder, viable_names=None, threshold=None,
                                 suffix="-masklabel.png", progress_cb=None):
    """Genere des masques OneTrainer pour toutes les images d'un dossier.

    viable_names : liste de noms de fichiers a traiter (None = tous les .png/.jpg)
    threshold : 0.5 pour masque binaire OneTrainer, None pour grayscale alpha
    suffix : convention OneTrainer = "-masklabel.png" a cote de l'image

    Retourne {written, skipped, errors}.
    """
    folder = Path(folder)
    images = sorted([f for f in folder.iterdir()
                      if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")])
    if viable_names is not None:
        images = [f for f in images if f.name in set(viable_names)]

    if not images:
        return {"written": 0, "skipped": 0, "errors": ["Aucune image"]}

    gen = MaskGenerator()
    written, skipped, errors = 0, 0, []
    total = len(images)

    for idx, img_path in enumerate(images):
        if progress_cb:
            try:
                progress_cb(idx + 1, total, img_path.name)
            except Exception:
                pass

        # Skip si masque existe deja
        mask_path = img_path.with_name(img_path.stem + suffix)
        if mask_path.exists() and mask_path.stat().st_size > 100:
            skipped += 1
            continue

        try:
            pil = Image.open(img_path)
            mask = gen.generate_mask(pil, threshold=threshold)
            mask.save(mask_path, "PNG")
            written += 1
        except Exception as e:
            errors.append(f"{img_path.name}: {e}")

    return {"written": written, "skipped": skipped, "errors": errors, "total": total}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mask_generator_local.py <image_or_folder> [--threshold 0.5]")
        sys.exit(1)
    target = Path(sys.argv[1])
    th = None
    if "--threshold" in sys.argv:
        th = float(sys.argv[sys.argv.index("--threshold") + 1])
    if target.is_file():
        gen = MaskGenerator()
        m = gen.generate_mask(Image.open(target), threshold=th)
        out = target.with_name(target.stem + "-masklabel.png")
        m.save(out)
        print(f"Saved {out}")
    else:
        result = generate_masks_for_folder(target, threshold=th)
        print(result)
