"""
JoyCaption Beta One - captioner LLaVA fine-tune.
Standard communautaire 2025-2026 pour datasets LoRA persona (consensus
HuggingFace + CivitAI : meilleur que Florence-2 sur les personnes, qui
hallucine emotions/vetements/contextes).

Modele : fancyfeast/llama-joycaption-beta-one-hf-llava
~8 Go en BF16, ~4 Go en INT4. CUDA recommande.

Modes (CAPTION_TYPE_PROMPT) :
  - "descriptive_caption" : description naturelle moyenne (defaut)
  - "descriptive_caption_short" : courte (1-2 phrases)
  - "descriptive_caption_long" : tres detaillee
  - "tags" : tags style booru via JoyCaption (alternatif a WD14)
  - "stable_diffusion_prompt" : style prompt SD direct
"""
import sys
from pathlib import Path


# Templates de prompts JoyCaption Beta One (extraits du repo officiel)
CAPTION_PROMPTS = {
    "descriptive_caption": (
        "Write a descriptive caption for this image in a formal tone."
    ),
    "descriptive_caption_short": (
        "Write a short descriptive caption for this image in a formal tone."
    ),
    "descriptive_caption_long": (
        "Write a long descriptive caption for this image in a formal tone, "
        "including detailed observations about composition, lighting, and subject."
    ),
    "tags": (
        "Write a list of Booru-like tags for this image."
    ),
    "stable_diffusion_prompt": (
        "Write a stable diffusion prompt for this image."
    ),
}


class JoyCaptioner:
    """Wrapper JoyCaption Beta One pour caption naturelle d'une image."""

    DEFAULT_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"

    def __init__(self, model_name=None, mode="descriptive_caption",
                 max_new_tokens=300, use_int4=False):
        """
        mode : voir CAPTION_PROMPTS
        use_int4 : quantization 4-bit pour fitter en 4 Go VRAM (bitsandbytes requis)
        """
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as e:
            raise RuntimeError(f"transformers/torch manquant : {e}")

        self.mode = mode
        self.prompt = CAPTION_PROMPTS.get(mode, CAPTION_PROMPTS["descriptive_caption"])
        self.max_new_tokens = max_new_tokens

        model_name = model_name or self.DEFAULT_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"STEP Chargement JoyCaption Beta One ({device}, "
              f"{'~4 Go INT4' if use_int4 else '~8 Go BF16'}, 1er run ~plusieurs minutes)...",
              file=sys.stderr, flush=True)

        # Chargement du processeur (image + tokenizer + chat template)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Quantization optionnelle
        load_kwargs = {"torch_dtype": dtype}
        if use_int4 and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                print(f"STEP INT4 indispo ({e}) - fallback BF16",
                      file=sys.stderr, flush=True)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs,
        )
        if device == "cuda" and "quantization_config" not in load_kwargs:
            self.model = self.model.to(device)
        self.model.eval()

        self.device = device
        self.dtype = dtype
        self.torch = torch

    def caption(self, pil_image):
        """Retourne une caption en langage naturel pour une image PIL."""
        img = pil_image.convert("RGB")

        # Format conversation LLaVA (chat template)
        convo = [
            {"role": "system",
             "content": "You are a helpful image captioner."},
            {"role": "user",
             "content": self.prompt},
        ]
        convo_string = self.processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[convo_string], images=[img], return_tensors="pt"
        ).to(self.device)

        # Pixel values doivent etre en bfloat16 sur CUDA
        if "pixel_values" in inputs and self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )
        # Strip le prompt en entree pour ne garder que la generation
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        caption = self.processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return caption.strip()


if __name__ == "__main__":
    from PIL import Image
    if len(sys.argv) < 2:
        print("Usage: joycaption_local.py <image_path> [mode]")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "descriptive_caption"
    cap = JoyCaptioner(mode=mode)
    print(cap.caption(img))
