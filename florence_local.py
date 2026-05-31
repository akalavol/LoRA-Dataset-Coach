"""
Florence-2 captioner standalone - genere des captions en langage naturel.
Pour les modeles entraines avec T5 (Flux, Wan, Hunyuan) qui beneficient
de descriptions naturelles plutot que de tags booru.

~270M params, ~540 Mo, telecharge depuis HuggingFace au 1er run.
Beaucoup plus leger que BLIP-2 (5 Go) et plus moderne.
"""
import sys
from pathlib import Path


class Florence2Captioner:
    """Wrapper Florence-2 pour caption naturelle d'une image."""

    def __init__(self, model_name="microsoft/Florence-2-base", task="<DETAILED_CAPTION>"):
        """task : <CAPTION> (court), <DETAILED_CAPTION> (moyen), <MORE_DETAILED_CAPTION> (long)."""
        self.task = task
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(f"transformers/torch manquant : {e}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"STEP Chargement Florence-2 ({device}, ~540 Mo si 1er run)...",
              file=sys.stderr, flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.torch = torch

    def caption(self, pil_image):
        """Retourne une caption en langage naturel."""
        img = pil_image.convert("RGB")
        inputs = self.processor(text=self.task, images=img, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype if v.dtype == self.torch.float32 else v.dtype)
                  for k, v in inputs.items()}
        # Les ids doivent rester en int
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()

        with self.torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=self.task, image_size=(img.width, img.height)
        )
        # parsed = {task: "caption naturelle"}
        return parsed.get(self.task, "").strip()


if __name__ == "__main__":
    from PIL import Image
    if len(sys.argv) < 2:
        print("Usage: florence_local.py <image_path>")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    cap = Florence2Captioner()
    print(cap.caption(img))
