"""
Detecteur d'images IA generees - reposes sur Organika/sdxl-detector
(99.6% accuracy sur SDXL, generalise correctement a Flux/Wan/InstantID).

Modele ViT ~350 Mo, telecharge au 1er run via HuggingFace Hub.

Usage :
    from ai_detector_local import AIDetector
    det = AIDetector()
    result = det.detect(pil_image)
    # result = {"ai_score": 0.97, "is_ai": True, "label": "artificial"}
"""
import sys
from pathlib import Path


class AIDetector:
    """Wrapper sdxl-detector pour scorer si une image est IA-generated."""

    DEFAULT_MODEL = "Organika/sdxl-detector"

    def __init__(self, model_name=None, threshold=0.5):
        """threshold : score au-dessus duquel l'image est consideree IA."""
        try:
            import torch
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError(f"transformers/torch manquant : {e}")

        model_name = model_name or self.DEFAULT_MODEL
        device = 0 if torch.cuda.is_available() else -1
        self.device_label = "cuda" if device == 0 else "cpu"

        print(f"STEP Chargement AI detector ({self.device_label}, "
              f"~350 Mo si 1er run)...", file=sys.stderr, flush=True)

        self.pipeline = pipeline(
            "image-classification",
            model=model_name,
            device=device,
        )
        self.threshold = threshold

    def detect(self, pil_image):
        """Retourne {"ai_score", "is_ai", "label"} pour une image PIL."""
        img = pil_image.convert("RGB")
        results = self.pipeline(img, top_k=2)
        # results : [{"label": "artificial", "score": 0.97}, {"label": "human", "score": 0.03}]
        ai_score = 0.0
        for r in results:
            if r["label"].lower() in ("artificial", "ai", "fake", "synthetic", "sdxl", "generated"):
                ai_score = r["score"]
                break
        is_ai = ai_score >= self.threshold
        # Top label pour debug
        top = max(results, key=lambda r: r["score"])
        return {
            "ai_score": round(float(ai_score), 3),
            "is_ai": is_ai,
            "label": top["label"],
        }


if __name__ == "__main__":
    from PIL import Image
    if len(sys.argv) < 2:
        print("Usage: ai_detector_local.py <image_path>")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    det = AIDetector()
    print(det.detect(img))
