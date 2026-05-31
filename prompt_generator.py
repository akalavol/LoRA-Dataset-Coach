"""
Targeted prompt generator - turns 'next_to_generate' suggestions
from the analyzer into ready-to-use ComfyUI workflows (InstantID).

For each missing shot type (profil, expressions, plans larges...), produces
a tailored prompt + a workflow JSON file ready to be dragged into ComfyUI.
"""
import json
import re
from datetime import datetime
from pathlib import Path


# ============================================================
# Suggestion -> prompt fragments mapping
# ============================================================

SHOT_PROMPT_MAP = {
    # Angles
    "profil": {
        "key_tags": ["profil", "profile", "yaw 60", "yaw 70", "yaw 80", "side"],
        "prompt_fragment": "side profile view, looking sideways",
        "dp_variation": "{side profile|three-quarter side view|nearly side profile|45-degree side angle}",
    },
    "3/4": {
        "key_tags": ["3/4", "three quarter", "three-quarter", "yaw 20", "yaw 30", "yaw 40"],
        "prompt_fragment": "three-quarter view, slightly turned",
        "dp_variation": "{three-quarter view|slightly turned to the side|partial profile|gentle side angle}",
    },
    # Plans
    "plan moyen": {
        "key_tags": ["plan moyen", "mi-corps", "both", "upper body"],
        "prompt_fragment": "upper body shot, visible from waist up",
        "dp_variation": "{upper body shot|mid shot from waist up|three-quarter length portrait|half-body framing}",
    },
    "plan large": {
        "key_tags": ["plan large", "body_only", "full body", "wide shot"],
        "prompt_fragment": "full body shot, head to toe visible",
        "dp_variation": "{full body shot|wide standing pose|long shot showing entire figure|full-length portrait}",
    },
    # Expressions
    "expression": {
        "key_tags": ["expression", "sourire", "rire", "sérieux", "surpris", "smile"],
        "prompt_fragment": "varied facial expression",
        "dp_variation": "{neutral expression|gentle smile|laughing with teeth|serious focused look|surprised expression|thoughtful gaze}",
    },
    # Generic
    "variete": {
        "key_tags": ["varie", "manque", "ajoute"],
        "prompt_fragment": "diverse pose and expression",
        "dp_variation": "{relaxed pose|natural standing|seated comfortably|hands at waist|hands in pockets}",
    },
}


def _match_suggestion(suggestion_text):
    """Returns the SHOT_PROMPT_MAP entry that best matches the suggestion."""
    text = suggestion_text.lower()
    best = None
    best_score = 0
    for category, data in SHOT_PROMPT_MAP.items():
        score = sum(1 for tag in data["key_tags"] if tag in text)
        if score > best_score:
            best_score = score
            best = (category, data)
    return best


def _extract_count(suggestion_text):
    """Extract a numeric count from 'Genere 5 photos...' or returns 1."""
    m = re.search(r"(\d+)[-\s]?(\d+)?", suggestion_text)
    if m:
        n1 = int(m.group(1))
        n2 = int(m.group(2)) if m.group(2) else n1
        return max(n1, n2)
    return 5


def generate_targeted_prompts(suggestions, base_persona_desc="", reference_image="reference_face_1024.png"):
    """
    Takes a list of `next_to_generate` suggestions from the analyzer summary,
    returns a list of {category, count, full_prompt, dp_prompt, suggestion_origin}
    ready to inject into the InstantID dataset workflow (12).
    """
    results = []
    persona = base_persona_desc.strip() if base_persona_desc else \
              "photo of a young woman, fair smooth skin, natural makeup, "\
              "professional photography, sharp focus, highly detailed skin texture, 50mm lens"

    seen_cats = set()
    for s in suggestions:
        match = _match_suggestion(s)
        if not match:
            continue
        category, data = match
        if category in seen_cats:
            continue
        seen_cats.add(category)
        count = _extract_count(s)

        # Build the full prompt
        fragment = data["prompt_fragment"]
        full_prompt = f"{persona}, {fragment}, soft natural lighting, neutral background"
        # Dynamic prompts variation (for batch generation)
        dp_prompt = (
            f"{persona}, "
            f"{data['dp_variation']}, "
            f"{{soft natural lighting|bright daylight|golden hour glow|soft window light}}, "
            f"{{plain beige background|white studio backdrop|neutral grey background|"
            f"blurred indoor environment}}, "
            f"{{gentle smile|neutral calm expression|soft serious look|warm laughing}}"
        )
        results.append({
            "category": category,
            "count": count,
            "suggestion_origin": s,
            "full_prompt": full_prompt,
            "dp_prompt": dp_prompt,
            "reference_image": reference_image,
        })

    return results


def build_instantid_workflow(prompt_entry, output_prefix="targeted_dataset",
                                checkpoint="RealVisXL_V5.0_fp16.safetensors",
                                batch_count=None):
    """
    Builds a ComfyUI InstantID workflow JSON that can be dropped into ComfyUI.
    Uses DPRandomGenerator for prompt variation + KSampler with random seed.

    Returns the workflow as a Python dict (JSON-serializable).
    """
    count = batch_count or prompt_entry.get("count", 5)
    ref_img = prompt_entry.get("reference_image", "reference_face_1024.png")
    cat = prompt_entry.get("category", "shot")
    dp_prompt = prompt_entry.get("dp_prompt", prompt_entry.get("full_prompt", ""))

    # Workflow basé sur le workflow 12 (InstantID dataset auto-varié)
    workflow = {
        "last_node_id": 13,
        "last_link_id": 17,
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple",
             "pos": [50, 100], "size": [320, 100], "order": 0, "mode": 0,
             "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [1]},
                {"name": "CLIP", "type": "CLIP", "links": [2, 3]},
                {"name": "VAE", "type": "VAE", "links": [4]},
             ],
             "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
             "widgets_values": [checkpoint]},

            {"id": 2, "type": "InstantIDModelLoader",
             "pos": [50, 240], "size": [320, 60], "order": 1, "mode": 0,
             "outputs": [{"name": "INSTANTID", "type": "INSTANTID", "links": [5]}],
             "properties": {"Node name for S&R": "InstantIDModelLoader"},
             "widgets_values": ["ip-adapter.bin"]},

            {"id": 3, "type": "InstantIDFaceAnalysis",
             "pos": [50, 340], "size": [320, 60], "order": 2, "mode": 0,
             "outputs": [{"name": "FACEANALYSIS", "type": "FACEANALYSIS", "links": [6]}],
             "properties": {"Node name for S&R": "InstantIDFaceAnalysis"},
             "widgets_values": ["CUDA"]},

            {"id": 4, "type": "ControlNetLoader",
             "pos": [50, 440], "size": [320, 60], "order": 3, "mode": 0,
             "outputs": [{"name": "CONTROL_NET", "type": "CONTROL_NET", "links": [7]}],
             "properties": {"Node name for S&R": "ControlNetLoader"},
             "widgets_values": ["instantid-controlnet.safetensors"]},

            {"id": 5, "type": "LoadImage",
             "pos": [50, 540], "size": [320, 314], "order": 4, "mode": 0,
             "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [8]}],
             "title": "Reference image",
             "properties": {"Node name for S&R": "LoadImage"},
             "widgets_values": [ref_img, "image"]},

            {"id": 13, "type": "DPRandomGenerator",
             "pos": [420, 100], "size": [400, 220], "order": 5, "mode": 0,
             "outputs": [{"name": "STRING", "type": "STRING", "links": [17]}],
             "title": f"Targeted prompt ({cat}, x{count})",
             "properties": {"Node name for S&R": "DPRandomGenerator"},
             "widgets_values": [dp_prompt, 0, "randomize", "Yes"]},

            {"id": 6, "type": "CLIPTextEncode",
             "pos": [420, 360], "size": [400, 100], "order": 6, "mode": 0,
             "inputs": [
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "text", "type": "STRING", "link": 17, "widget": {"name": "text"}},
             ],
             "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [9]}],
             "title": "Positive (receives dynamic prompt)",
             "properties": {"Node name for S&R": "CLIPTextEncode"},
             "widgets_values": [f"photo of a woman, {cat}"]},

            {"id": 7, "type": "CLIPTextEncode",
             "pos": [420, 500], "size": [400, 120], "order": 7, "mode": 0,
             "inputs": [{"name": "clip", "type": "CLIP", "link": 3}],
             "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [10]}],
             "title": "Negative",
             "properties": {"Node name for S&R": "CLIPTextEncode"},
             "widgets_values": [
                "(worst quality, low quality:1.4), text, watermark, blurry, "
                "deformed, cartoon, 3d render, painting, anime, two people, "
                "multiple faces, extra fingers, bad hands"
             ]},

            {"id": 8, "type": "ApplyInstantID",
             "pos": [870, 200], "size": [320, 280], "order": 8, "mode": 0,
             "inputs": [
                {"name": "instantid", "type": "INSTANTID", "link": 5},
                {"name": "insightface", "type": "FACEANALYSIS", "link": 6},
                {"name": "control_net", "type": "CONTROL_NET", "link": 7},
                {"name": "image", "type": "IMAGE", "link": 8},
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "positive", "type": "CONDITIONING", "link": 9},
                {"name": "negative", "type": "CONDITIONING", "link": 10},
             ],
             "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [11]},
                {"name": "positive", "type": "CONDITIONING", "links": [12]},
                {"name": "negative", "type": "CONDITIONING", "links": [13]},
             ],
             "title": "Apply InstantID (weight 0.8)",
             "properties": {"Node name for S&R": "ApplyInstantID"},
             "widgets_values": [0.8, 0, 1]},

            {"id": 9, "type": "EmptyLatentImage",
             "pos": [870, 510], "size": [320, 110], "order": 9, "mode": 0,
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [14]}],
             "properties": {"Node name for S&R": "EmptyLatentImage"},
             "widgets_values": [832, 1216, count]},   # batch_size = count !

            {"id": 10, "type": "KSampler",
             "pos": [1240, 200], "size": [320, 270], "order": 10, "mode": 0,
             "inputs": [
                {"name": "model", "type": "MODEL", "link": 11},
                {"name": "positive", "type": "CONDITIONING", "link": 12},
                {"name": "negative", "type": "CONDITIONING", "link": 13},
                {"name": "latent_image", "type": "LATENT", "link": 14},
             ],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [15]}],
             "title": "Sampler (seed = randomize)",
             "properties": {"Node name for S&R": "KSampler"},
             "widgets_values": [42, "randomize", 30, 5.0, "dpmpp_2m", "karras", 1.0]},

            {"id": 11, "type": "VAEDecode",
             "pos": [1600, 200], "size": [210, 50], "order": 11, "mode": 0,
             "inputs": [
                {"name": "samples", "type": "LATENT", "link": 15},
                {"name": "vae", "type": "VAE", "link": 4},
             ],
             "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
             "properties": {"Node name for S&R": "VAEDecode"}},

            {"id": 12, "type": "SaveImage",
             "pos": [1600, 300], "size": [400, 450], "order": 12, "mode": 0,
             "inputs": [{"name": "images", "type": "IMAGE", "link": 16}],
             "properties": {},
             "widgets_values": [f"{output_prefix}_{cat.replace(' ', '_')}"]},
        ],
        "links": [
            [1, 1, 0, 8, 4, "MODEL"],
            [2, 1, 1, 6, 0, "CLIP"],
            [3, 1, 1, 7, 0, "CLIP"],
            [4, 1, 2, 11, 1, "VAE"],
            [5, 2, 0, 8, 0, "INSTANTID"],
            [6, 3, 0, 8, 1, "FACEANALYSIS"],
            [7, 4, 0, 8, 2, "CONTROL_NET"],
            [8, 5, 0, 8, 3, "IMAGE"],
            [9, 6, 0, 8, 5, "CONDITIONING"],
            [10, 7, 0, 8, 6, "CONDITIONING"],
            [11, 8, 0, 10, 0, "MODEL"],
            [12, 8, 1, 10, 1, "CONDITIONING"],
            [13, 8, 2, 10, 2, "CONDITIONING"],
            [14, 9, 0, 10, 3, "LATENT"],
            [15, 10, 0, 11, 0, "LATENT"],
            [16, 11, 0, 12, 0, "IMAGE"],
            [17, 13, 0, 6, 1, "STRING"],
        ],
        "groups": [{
            "title": f"Targeted generation : {cat} (x{count})",
            "bounding": [40, 50, 1980, 880],
            "color": "#3a7"
        }],
        "config": {}, "extra": {}, "version": 0.4,
    }
    return workflow


def export_workflows_for_suggestions(suggestions, output_folder,
                                       base_persona_desc="",
                                       reference_image="reference_face_1024.png",
                                       checkpoint="RealVisXL_V5.0_fp16.safetensors"):
    """
    Main entry point. Exports one .json workflow per suggestion category.
    Returns list of {category, count, workflow_path}.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    prompts = generate_targeted_prompts(suggestions, base_persona_desc, reference_image)
    written = []
    for p in prompts:
        wf = build_instantid_workflow(p, checkpoint=checkpoint)
        filename = f"targeted_{p['category'].replace(' ', '_').replace('/', '_')}.json"
        out_path = output_folder / filename
        out_path.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append({
            "category": p["category"], "count": p["count"],
            "workflow_path": str(out_path),
            "origin": p["suggestion_origin"],
        })

    # README
    readme = ["# Targeted generation workflows",
              f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              "",
              "Each .json file is a ready-to-use ComfyUI workflow.",
              "Drag and drop one in ComfyUI to generate the targeted shot type.",
              ""]
    for w in written:
        readme.append(f"## {w['category']} (x{w['count']})")
        readme.append(f"Origin: {w['origin']}")
        readme.append(f"File: `{Path(w['workflow_path']).name}`")
        readme.append("")
    (output_folder / "README.md").write_text("\n".join(readme), encoding="utf-8")

    return written


if __name__ == "__main__":
    import sys
    test_suggestions = [
        "Genere 2-3 photos de profil (yaw 60-80°) pour donner l'angle au LoRA",
        "Genere 4-6 plans moyens (mi-corps, visage visible) pour varier les plans",
        "Expression dominante : sourire — varie : ajoute du serieux, surpris, neutre",
    ]
    out = sys.argv[1] if len(sys.argv) > 1 else "./targeted_workflows"
    result = export_workflows_for_suggestions(test_suggestions, out)
    print(json.dumps(result, indent=2))
