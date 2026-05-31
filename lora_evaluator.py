"""
LoRA Post-Training Evaluator - MirrorMetrics-inspired (2026 standard).

Once you've trained a LoRA, generate ~30 test images with it, then run this
to know if your LoRA actually learned the identity or memorized accessories.

Metrics computed :
  - R-FaceSim         : mean cosine sim of each generated image vs
                         REAL photos of the subject (NOT the anchor used
                         for InstantID, to avoid inflated scores)
  - Copycat Detector  : if generated images are >0.95 similar to dataset
                         photos, the LoRA is copying instead of generalizing
  - Black Hole Ranking: per-test image, identifies which training image
                         it most resembles (collapse signal)
  - Identity score    : mean R-FaceSim, 0-1
  - Generalization    : variance of R-FaceSim (low var = mode collapse)
  - t-SNE visu (opt.) : 2D scatter of embeddings (matplotlib)

CLI usage :
    python lora_evaluator.py <generated_folder> <reference_folder> [output.json]
"""
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


def evaluate_lora(generated_folder, reference_folder,
                   training_folder=None, output_json=None,
                   progress_cb=None):
    """
    generated_folder  : Le dossier contenant N images generees par ton LoRA
                         (lance ton workflow ComfyUI 30 fois avec varying seeds)
    reference_folder  : Le dossier contenant les VRAIES photos du sujet
                         (pas la photo d'ancrage InstantID - des photos
                         independantes de la meme personne, prises avant tout)
    training_folder   : (optionnel) Le dossier d'entrainement, pour detecter
                         le copycat (le LoRA recopie-t-il l'une des photos
                         d'entrainement ?)
    output_json       : Chemin de sortie pour le rapport

    Returns dict with all metrics.
    """
    gen_folder = Path(generated_folder)
    ref_folder = Path(reference_folder)
    if not gen_folder.is_dir() or not ref_folder.is_dir():
        return {"error": f"Folder missing: gen={gen_folder.is_dir()} ref={ref_folder.is_dir()}"}

    exts = (".png", ".jpg", ".jpeg", ".webp")
    gen_files = sorted([f for f in gen_folder.iterdir()
                         if f.is_file() and f.suffix.lower() in exts])
    ref_files = sorted([f for f in ref_folder.iterdir()
                         if f.is_file() and f.suffix.lower() in exts])
    train_files = []
    if training_folder:
        tf = Path(training_folder)
        if tf.is_dir():
            train_files = sorted([f for f in tf.iterdir()
                                   if f.is_file() and f.suffix.lower() in exts])

    if not gen_files:
        return {"error": "No generated images"}
    if len(ref_files) < 3:
        return {"error": "Need at least 3 reference photos for reliable R-FaceSim"}

    # Imports lourds ici
    try:
        import numpy as np
        from PIL import Image
        from insightface.app import FaceAnalysis
    except ImportError as e:
        return {"error": f"Missing module: {e}"}

    # Init insightface
    insight_roots = [
        r"C:\AI\ComfyUI-future\ComfyUI_windows_portable\ComfyUI\models\insightface",
        r"C:\AI\ComfyUI-Zluda\models\insightface",
        os.path.expanduser(r"~\.insightface"),
    ]
    insight_root = next(
        (p for p in insight_roots
         if os.path.isdir(os.path.join(p, "models", "antelopev2"))),
        None,
    )
    if insight_root is None:
        return {"error": "antelopev2 models not found"}

    print("STEP Loading insightface...", file=sys.stderr, flush=True)
    import contextlib
    @contextlib.contextmanager
    def redirect():
        old = sys.stdout
        sys.stdout = sys.stderr
        try: yield
        finally: sys.stdout = old

    with redirect():
        app = FaceAnalysis(name="antelopev2", root=insight_root,
                            providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

    def embed_folder(files, label):
        embs = []
        names = []
        total = len(files)
        for idx, p in enumerate(files):
            if progress_cb:
                try: progress_cb(label, idx + 1, total, p.name)
                except Exception: pass
            print(f"PROGRESS {label} {idx+1}/{total} {p.name}",
                  file=sys.stderr, flush=True)
            try:
                pil = Image.open(p).convert("RGB")
                bgr = np.array(pil)[:, :, ::-1]
                faces = app.get(bgr)
                if not faces:
                    embs.append(None)
                    names.append(p.name)
                    continue
                main = max(faces,
                            key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                embs.append(main.normed_embedding)
                names.append(p.name)
            except Exception as e:
                print(f"STEP Error on {p.name}: {e}", file=sys.stderr, flush=True)
                embs.append(None)
                names.append(p.name)
        return embs, names

    print("STEP Embedding reference photos...", file=sys.stderr, flush=True)
    ref_embs, ref_names = embed_folder(ref_files, "REF")
    print("STEP Embedding generated photos...", file=sys.stderr, flush=True)
    gen_embs, gen_names = embed_folder(gen_files, "GEN")
    train_embs, train_names = [], []
    if train_files:
        print("STEP Embedding training photos (copycat check)...",
              file=sys.stderr, flush=True)
        train_embs, train_names = embed_folder(train_files, "TRAIN")

    # Stats par image generee
    valid_ref_embs = [e for e in ref_embs if e is not None]
    if not valid_ref_embs:
        return {"error": "No face detected in any reference photo"}

    ref_mat = np.array(valid_ref_embs)  # (R, 512)

    per_image = []
    rfacesim_scores = []
    copycat_alerts = []
    blackhole_attribution = {}

    for i, gemb in enumerate(gen_embs):
        gname = gen_names[i]
        record = OrderedDict()
        record["name"] = gname
        record["has_face"] = gemb is not None
        if gemb is None:
            record["r_facesim"] = None
            record["nearest_ref"] = None
            record["nearest_train"] = None
            record["copycat"] = False
            per_image.append(record)
            continue

        # R-FaceSim = mean sim vs reference photos
        sims_ref = ref_mat @ gemb  # (R,)
        r_facesim = float(np.mean(sims_ref))
        nearest_ref_idx = int(np.argmax(sims_ref))
        record["r_facesim"] = round(r_facesim, 3)
        record["r_facesim_max"] = round(float(np.max(sims_ref)), 3)
        record["r_facesim_min"] = round(float(np.min(sims_ref)), 3)
        record["nearest_ref"] = {
            "name": ref_names[nearest_ref_idx] if nearest_ref_idx < len(ref_names) else "?",
            "sim": round(float(sims_ref[nearest_ref_idx]), 3),
        }
        rfacesim_scores.append(r_facesim)

        # Copycat detector : sim to training images
        if train_embs:
            valid_train_embs = [e for e in train_embs if e is not None]
            if valid_train_embs:
                tmat = np.array(valid_train_embs)
                sims_train = tmat @ gemb
                near_train_idx = int(np.argmax(sims_train))
                max_train_sim = float(np.max(sims_train))
                # Map back to train_names index
                valid_train_names = [train_names[k] for k, e in enumerate(train_embs) if e is not None]
                near_name = valid_train_names[near_train_idx] if near_train_idx < len(valid_train_names) else "?"
                record["nearest_train"] = {
                    "name": near_name, "sim": round(max_train_sim, 3),
                }
                if max_train_sim > 0.95:
                    record["copycat"] = True
                    copycat_alerts.append({
                        "generated": gname, "copy_of": near_name,
                        "sim": round(max_train_sim, 3),
                    })
                else:
                    record["copycat"] = False
                # Black Hole : compte combien de generes pointent vers chaque train image
                blackhole_attribution[near_name] = blackhole_attribution.get(near_name, 0) + 1

        per_image.append(record)

    # Stats globales
    if rfacesim_scores:
        import statistics
        rf_mean = round(statistics.mean(rfacesim_scores), 3)
        rf_std = round(statistics.stdev(rfacesim_scores), 3) if len(rfacesim_scores) > 1 else 0
        rf_min = round(min(rfacesim_scores), 3)
        rf_max = round(max(rfacesim_scores), 3)
    else:
        rf_mean = rf_std = rf_min = rf_max = None

    # Black Hole ranking : top-3 des images d'entrainement les plus "copiees"
    black_hole_top = []
    if blackhole_attribution:
        sorted_bh = sorted(blackhole_attribution.items(), key=lambda x: -x[1])
        n_gen_valid = sum(1 for e in gen_embs if e is not None)
        for name, count in sorted_bh[:5]:
            ratio = count / max(n_gen_valid, 1)
            black_hole_top.append({
                "training_image": name, "attributed_count": count,
                "ratio": round(ratio, 2),
            })

    # Verdict
    verdict = _build_verdict(rf_mean, rf_std, len(copycat_alerts),
                              n_gen=len([e for e in gen_embs if e is not None]),
                              n_total_gen=len(gen_embs),
                              black_hole_top=black_hole_top)

    result = {
        "summary": {
            "n_generated": len(gen_files),
            "n_reference": len(ref_files),
            "n_training": len(train_files),
            "n_gen_with_face": sum(1 for e in gen_embs if e is not None),
            "n_ref_with_face": len(valid_ref_embs),
            "r_facesim_mean": rf_mean,
            "r_facesim_std": rf_std,
            "r_facesim_min": rf_min,
            "r_facesim_max": rf_max,
            "copycat_count": len(copycat_alerts),
            "copycat_alerts": copycat_alerts,
            "black_hole_top": black_hole_top,
            "verdict": verdict,
        },
        "per_image": per_image,
    }

    if output_json:
        Path(output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    return result


def _build_verdict(rf_mean, rf_std, copycat_count, n_gen, n_total_gen, black_hole_top):
    """Build the verdict and grade for the trained LoRA."""
    if rf_mean is None:
        return {"grade": "F", "desc": "NO FACES DETECTED",
                 "advice": "Check if generated images actually contain faces.",
                 "score": 0}

    # Penalize copycat
    copycat_penalty = min(30, copycat_count * 5)
    # Penalize face detection failures
    detect_penalty = (1 - n_gen / max(n_total_gen, 1)) * 20
    # Mode collapse : low std = single mode
    collapse_penalty = 0
    if rf_std is not None and rf_std < 0.03 and n_gen >= 10:
        collapse_penalty = 15
    # Black hole : if one training image attracts >40% of generations
    blackhole_penalty = 0
    if black_hole_top and black_hole_top[0].get("ratio", 0) > 0.4:
        blackhole_penalty = 15

    # Score base : R-FaceSim mapped to 0-100
    base_score = max(0, min(100, (rf_mean - 0.3) * 200))  # 0.3 -> 0, 0.8 -> 100
    final_score = max(0, base_score - copycat_penalty - detect_penalty
                          - collapse_penalty - blackhole_penalty)
    final_score = round(final_score, 0)

    if final_score >= 85:
        grade, desc = "A", "EXCELLENT LORA"
        advice = "Identity learned cleanly. Use it confidently."
    elif final_score >= 70:
        grade, desc = "B+", "GOOD LORA"
        advice = "Solid identity. Minor variance loss possible."
    elif final_score >= 55:
        grade, desc = "B", "OK LORA"
        advice = "Identity present but check copycat alerts."
    elif final_score >= 40:
        grade, desc = "C", "WEAK LORA"
        advice = "Identity loose. Retrain with more diverse dataset."
    elif final_score >= 25:
        grade, desc = "D", "BAD LORA"
        advice = ("Identity barely learned OR copycat heavy. "
                   "Check the dataset analyzer first.")
    else:
        grade, desc = "F", "FAILED"
        advice = "Retrain from scratch with better dataset."

    issues = []
    if copycat_count > 0:
        issues.append(f"Copycat: {copycat_count} generated images are >0.95 similar to training photos")
    if rf_std and rf_std < 0.03:
        issues.append(f"Mode collapse: very low variance (std={rf_std})")
    if black_hole_top and black_hole_top[0].get("ratio", 0) > 0.4:
        issues.append(f"Black hole: one training image attracts {black_hole_top[0]['ratio']*100:.0f}% of generations")
    if n_gen < n_total_gen:
        miss = n_total_gen - n_gen
        issues.append(f"{miss} generated image(s) had no detectable face")

    return {
        "grade": grade,
        "desc": desc,
        "score": final_score,
        "advice": advice,
        "issues": issues,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: lora_evaluator.py <generated_folder> <reference_folder> "
              "[training_folder] [output.json]")
        sys.exit(1)
    gen = sys.argv[1]
    ref = sys.argv[2]
    train = sys.argv[3] if len(sys.argv) > 3 else None
    out = sys.argv[4] if len(sys.argv) > 4 else None
    result = evaluate_lora(gen, ref, training_folder=train, output_json=out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
