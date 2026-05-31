"""
WD14 Tagger standalone - reutilise le modele installe par ComfyUI-WD14-Tagger
(ou le telecharge si absent) pour generer des captions sur un dossier d'images.

Usage en module :
    from wd14_local import WD14Tagger
    tagger = WD14Tagger()  # init unique (charge le modele)
    tags = tagger.tag(pil_image)  # -> "tag1, tag2, tag3"
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Dossiers possibles ou trouver le modele WD14
WD14_MODEL_DIRS = [
    r"C:\AI\ComfyUI-future\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-WD14-Tagger\models",
    r"C:\AI\ComfyUI-Zluda\custom_nodes\ComfyUI-WD14-Tagger\models",
]

DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"
HF_REPO = f"SmilingWolf/{DEFAULT_MODEL}"


def _ensure_model_downloaded(model_name=DEFAULT_MODEL):
    """Cherche les fichiers .onnx + .csv dans les dossiers connus, telecharge si absent."""
    for d in WD14_MODEL_DIRS:
        onnx = Path(d) / f"{model_name}.onnx"
        csv_path = Path(d) / f"{model_name}.csv"
        if onnx.is_file() and csv_path.is_file():
            return str(onnx), str(csv_path)

    # Pas trouve : telechargement dans le 1er dossier
    target_dir = Path(WD14_MODEL_DIRS[0])
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx = target_dir / f"{model_name}.onnx"
    csv_path = target_dir / f"{model_name}.csv"

    print(f"STEP Telechargement WD14 {model_name} (~330 Mo, une seule fois)...",
          file=sys.stderr, flush=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=f"SmilingWolf/{model_name}",
                        filename="model.onnx", local_dir=str(target_dir),
                        local_dir_use_symlinks=False)
        # Renomme model.onnx -> wd-v1-4-moat-tagger-v2.onnx
        src = target_dir / "model.onnx"
        if src.exists() and src != onnx:
            src.rename(onnx)
        hf_hub_download(repo_id=f"SmilingWolf/{model_name}",
                        filename="selected_tags.csv", local_dir=str(target_dir),
                        local_dir_use_symlinks=False)
        src = target_dir / "selected_tags.csv"
        if src.exists() and src != csv_path:
            src.rename(csv_path)
    except Exception as e:
        print(f"STEP Echec telechargement WD14 : {e}", file=sys.stderr, flush=True)
        return None, None

    return str(onnx), str(csv_path)


class WD14Tagger:
    """Tagger WD14 reutilisable (charge le modele une fois)."""

    def __init__(self, threshold=0.35, character_threshold=0.85,
                 exclude_tags=None, replace_underscore=True):
        self.threshold = threshold
        self.character_threshold = character_threshold
        self.exclude_tags = set(t.strip().lower() for t in (exclude_tags or [])) if exclude_tags else set()
        self.replace_underscore = replace_underscore

        onnx_path, csv_path = _ensure_model_downloaded()
        if not onnx_path:
            raise RuntimeError("Modele WD14 introuvable et echec du telechargement")

        # Init session ONNX (CPU - on n'a pas besoin de GPU pour 30 photos)
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.input_size = self.input.shape[1]
        self.output_name = self.session.get_outputs()[0].name

        # Charge les tags
        self.tags = []
        self.general_index = None
        self.character_index = None
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if self.general_index is None and row[2] == "0":
                    self.general_index = i
                elif self.character_index is None and row[2] == "4":
                    self.character_index = i
                tag = row[1].replace("_", " ") if self.replace_underscore else row[1]
                self.tags.append(tag)

    def tag(self, pil_image):
        """Retourne une string `tag1, tag2, tag3` pour une image PIL."""
        # Resize + pad to square (fond blanc)
        img = pil_image.convert("RGB")
        ratio = self.input_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (self.input_size, self.input_size), (255, 255, 255))
        square.paste(img, ((self.input_size - new_size[0]) // 2,
                           (self.input_size - new_size[1]) // 2))

        arr = np.array(square).astype(np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR
        arr = np.expand_dims(arr, 0)

        probs = self.session.run([self.output_name], {self.input.name: arr})[0][0]
        result = list(zip(self.tags, probs))

        general = [(t, p) for t, p in result[self.general_index:self.character_index]
                   if p > self.threshold]
        character = [(t, p) for t, p in result[self.character_index:]
                     if p > self.character_threshold]

        all_tags = character + general
        all_tags = [(t, p) for t, p in all_tags if t.lower() not in self.exclude_tags]

        # Tri par probabilite decroissante
        all_tags.sort(key=lambda x: -x[1])
        return ", ".join(t.replace("(", "\\(").replace(")", "\\)") for t, _ in all_tags)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: wd14_local.py <image_path>")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    tagger = WD14Tagger()
    print(tagger.tag(img))
