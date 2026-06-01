"""
Analyseur de coherence pour dataset LoRA visage/corps.
- Detecte les visages avec insightface (antelopev2)
- Calcule similarite cosinus entre embeddings
- Identifie les outliers
- Verifie aussi resolution, ratio, taille

Lance via Python qui a insightface :
  python.exe analyze_dataset.py <folder> <mode>
  mode = 'face' ou 'body' ou 'full'

Output : JSON sur stdout.
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

# Force insightface a NE PAS utiliser onnxruntime-gpu (peut planter sur CPU)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Force UTF-8 sur stdout pour les emojis dans le JSON (sinon cp1252 crash)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


@contextlib.contextmanager
def redirect_stdout_to_stderr():
    """Redirige stdout vers stderr le temps du with - utile pour les libs verbeuses."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def dhash(pil_image, hash_size=8):
    """Difference hash perceptuel (64 bits) - detection de duplicates visuels.
    Hamming < 5 = quasi-identique, < 10 = tres similaire."""
    from PIL import Image
    import numpy as np
    img = pil_image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count("1")


def resolve_device(pref="auto"):
    """Retourne ('cuda'|'cpu', onnx_providers).
    pref = 'auto' | 'cuda' | 'cpu'.
    """
    want_cuda = False
    if pref in ("cuda", "auto"):
        # Dans les deux cas on verifie que torch CUDA est REELLEMENT dispo
        # (sinon clip_model.to('cuda') planterait). 'cuda' force = meme garde-fou.
        try:
            import torch
            want_cuda = torch.cuda.is_available()
        except Exception:
            try:
                import onnxruntime as ort
                want_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            except Exception:
                want_cuda = False
        if pref == "cuda" and not want_cuda:
            print("STEP ⚠ device=cuda demandé mais aucun GPU CUDA détecté — repli sur CPU",
                  file=sys.stderr, flush=True)

    if want_cuda:
        # Verifie que onnxruntime a bien le provider CUDA dispo
        onnx_providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return "cuda", onnx_providers
    return "cpu", ["CPUExecutionProvider"]


# ============================================================
# Scores par target : evalue le dataset pour chaque type de LoRA
# ============================================================

def _grade_from_score(s):
    if s >= 90: return "A+"
    if s >= 80: return "A"
    if s >= 70: return "B+"
    if s >= 60: return "B"
    if s >= 50: return "C"
    if s >= 40: return "D"
    return "F"


def _compute_target_scores(results, viable_imgs, summary, n_viable):
    """Note specifique pour chaque famille de target (SDXL / Flux / Wan / Anime).
    Pas une note par target individuel (19 targets x calculs), mais regroupe
    en 5 familles : SDXL_classic, SDXL_anime, Flux, Wan_video, Hunyuan_video.
    """
    if not viable_imgs:
        return {}

    n_total = len(results)
    # === Stats reutilisables ===
    # Resolution
    res_1024_count = sum(1 for r in viable_imgs
                          if (r.get("width") or 0) >= 1024 and (r.get("height") or 0) >= 1024)
    res_512_count = sum(1 for r in viable_imgs
                         if (r.get("width") or 0) >= 512 and (r.get("height") or 0) >= 512)
    # Captioning
    has_wd14 = sum(1 for r in viable_imgs if r.get("wd14_tags"))
    has_natural = sum(1 for r in viable_imgs
                       if r.get("joycaption") or r.get("natural_caption"))
    has_joy = sum(1 for r in viable_imgs if r.get("joycaption"))
    # AR distribution
    ar_dist = summary.get("aspect_ratio_distribution", {})
    sq = ar_dist.get("square", 0)
    po = ar_dist.get("portrait", 0) + ar_dist.get("tall_portrait", 0)
    la = ar_dist.get("landscape", 0) + ar_dist.get("wide_landscape", 0)
    n_with_ar = sq + po + la
    ar_variety = (1 if sq > 0 else 0) + (1 if po > 0 else 0) + (1 if la > 0 else 0)
    # Artefacts
    art_high = sum(1 for r in viable_imgs if r.get("artifacts_severity") == "high")
    # Diversité
    diversity_score = summary.get("diversity", {}).get("overall_score") or 50

    scores = {}

    # ============ FAMILLE 1 : SDXL classique (photo réelle) ============
    s = 0
    reasons = []
    # Volume (sur 25)
    if n_viable >= 30: s += 25; reasons.append("30+ viables")
    elif n_viable >= 20: s += 22
    elif n_viable >= 15: s += 18
    elif n_viable >= 10: s += 12
    else: s += 5; reasons.append(f"seulement {n_viable} viables")
    # Resolution 1024+ (sur 20)
    if res_1024_count / max(n_viable, 1) >= 0.9: s += 20
    elif res_1024_count / max(n_viable, 1) >= 0.5: s += 15
    elif res_512_count / max(n_viable, 1) >= 0.9: s += 10
    else: s += 5; reasons.append("resolution sub-optimale")
    # Captions WD14 (sur 15)
    if has_wd14 / max(n_viable, 1) >= 0.95: s += 15
    elif has_wd14 / max(n_viable, 1) >= 0.7: s += 10
    else: s += 3; reasons.append("captions WD14 manquantes")
    # Diversite (sur 20)
    s += int(diversity_score * 0.20)
    # AR square dominant (sur 10)
    if n_with_ar > 0 and sq / n_with_ar >= 0.8: s += 10
    elif n_with_ar > 0 and sq / n_with_ar >= 0.5: s += 6
    # Pas d'artefacts severes (sur 10)
    if art_high == 0: s += 10
    else: s += max(0, 10 - 2 * art_high); reasons.append(f"{art_high} artefacts")

    scores["SDXL classique"] = {
        "score": min(100, s), "grade": _grade_from_score(min(100, s)),
        "reason": ", ".join(reasons) if reasons else "tout en ordre",
        "applies_to": ["sdxl_kohya", "onetrainer_sdxl"],
    }

    # ============ FAMILLE 2 : SDXL forks anime (Pony/Illustrious/NoobAI) ============
    s = scores["SDXL classique"]["score"]
    # Plus tolerant aux artefacts (style anime)
    if art_high > 0:
        s += min(8, art_high * 2)  # remet une partie des points
    scores["SDXL anime"] = {
        "score": min(100, s), "grade": _grade_from_score(min(100, s)),
        "reason": "même base que SDXL classique, plus tolérant aux artefacts",
        "applies_to": ["pony_kohya", "illustrious_kohya", "noobai_kohya"],
    }

    # ============ FAMILLE 3 : Flux (T5, ratios variés bénéfiques) ============
    s = 0
    reasons = []
    if n_viable >= 25: s += 25
    elif n_viable >= 15: s += 20
    elif n_viable >= 10: s += 12
    else: s += 5; reasons.append(f"seulement {n_viable} viables")
    # Resolution 1024 (sur 20)
    if res_1024_count / max(n_viable, 1) >= 0.9: s += 20
    elif res_1024_count / max(n_viable, 1) >= 0.5: s += 15
    else: s += 5; reasons.append("res < 1024 sous-optimal pour Flux")
    # Captions naturelles (preferences JoyCaption) (sur 20)
    if has_joy / max(n_viable, 1) >= 0.9: s += 20
    elif has_natural / max(n_viable, 1) >= 0.9: s += 15  # Florence-2 ok mais sous-optimal
    elif has_wd14 / max(n_viable, 1) >= 0.9: s += 7; reasons.append("captions WD14 seulement, JoyCaption recommande")
    else: s += 3; reasons.append("captions manquantes")
    # Diversite AR (Flux benefice du multi-bucket) (sur 15)
    if ar_variety >= 3: s += 15
    elif ar_variety == 2: s += 10
    else: s += 5; reasons.append("ratios peu variés (Flux multi-bucket sous-utilisé)")
    # Diversite globale (sur 15)
    s += int(diversity_score * 0.15)
    # Pas d'artefacts severes (sur 5)
    if art_high == 0: s += 5

    scores["Flux"] = {
        "score": min(100, s), "grade": _grade_from_score(min(100, s)),
        "reason": ", ".join(reasons) if reasons else "tout en ordre",
        "applies_to": ["flux_aitoolkit", "flux_kohya", "chroma_aitoolkit"],
    }

    # ============ FAMILLE 4 : Wan vidéo (ratios vidéo, captions très longues) ============
    s = 0
    reasons = []
    if n_viable >= 30: s += 25
    elif n_viable >= 20: s += 18
    elif n_viable >= 10: s += 10
    else: s += 3; reasons.append(f"vraiment {n_viable} viables ? Wan demande +")
    # Resolution (sur 15) - Wan accepte plus petite
    if res_512_count / max(n_viable, 1) >= 0.9: s += 15
    else: s += 5
    # Captions naturelles longues (sur 20) - Wan T5 adore les phrases longues
    long_caps = sum(1 for r in viable_imgs
                     if (r.get("joycaption") or r.get("natural_caption") or "")
                     and len((r.get("joycaption") or r.get("natural_caption"))) >= 100)
    if long_caps / max(n_viable, 1) >= 0.7: s += 20
    elif has_natural / max(n_viable, 1) >= 0.7: s += 12
    elif has_wd14 / max(n_viable, 1) >= 0.7: s += 4; reasons.append("WD14 trop court pour Wan, JoyCaption recommande")
    # Diversite AR (Wan multi-bucket vidéo) (sur 15)
    if ar_variety >= 2: s += 15
    elif ar_variety == 1: s += 8
    # Portraits + paysages présents (sur 10) - bon pour I2V/T2V
    if po >= 2 and la >= 2: s += 10
    elif po >= 1 or la >= 1: s += 5
    # Diversite globale (sur 15)
    s += int(diversity_score * 0.15)

    scores["Wan vidéo"] = {
        "score": min(100, s), "grade": _grade_from_score(min(100, s)),
        "reason": ", ".join(reasons) if reasons else "tout en ordre",
        "applies_to": ["wan21_musubi", "wan22_musubi"],
    }

    # ============ FAMILLE 5 : Vidéo Hunyuan/Mochi/Open-Sora/LTX/CogVideoX ============
    s = max(0, scores["Wan vidéo"]["score"] - 5)  # similaire mais légèrement plus exigeant
    scores["Vidéo (Hunyuan/Mochi/LTX/CogVideoX)"] = {
        "score": s, "grade": _grade_from_score(s),
        "reason": "même critères que Wan mais ces modèles demandent souvent des clips MP4 en plus",
        "applies_to": ["hunyuan_diffpipe", "ltx_video_diffpipe",
                         "cogvideox_diffpipe", "mochi_diffpipe", "open_sora_diffpipe"],
    }

    return scores


# ============================================================
# Cache d'analyse : evite de re-analyser les photos deja vues
# ============================================================
CACHE_VERSION = 2
CACHE_FILE = ".analyzer_cache.json"


def _file_key(p):
    """Identifie une photo par nom + taille + date de modification.
    Si tu remplaces le fichier (meme nom), le mtime change -> re-analyse."""
    try:
        st = p.stat()
        return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        return p.name


def load_cache(folder):
    """Charge le cache d'analyse. Retourne {} si invalide/absent."""
    cache_path = Path(folder) / CACHE_FILE
    if not cache_path.is_file():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            return {}
        return data
    except Exception:
        return {}


def save_cache(folder, cache_dict):
    """Sauve le cache."""
    cache_path = Path(folder) / CACHE_FILE
    try:
        cache_path.write_text(json.dumps(cache_dict, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    except Exception as e:
        print(f"STEP Echec ecriture cache : {e}", file=sys.stderr, flush=True)


def analyze(folder, mode="full", ref_image=None, captioner_mode="wd14",
             ai_detection=True, device="auto"):
    """
    captioner_mode :
      - "wd14"       : tags booru (SDXL/SD1.5/Kohya) - rapide, ONNX
      - "natural"    : Florence-2 caption naturelle (deprecie en 2026 - hallucine sur personnes)
      - "joycaption" : JoyCaption Beta One - STANDARD 2026 pour LoRA persona
      - "both"       : WD14 + Florence-2 (compat retrocompat)
      - "all"        : WD14 + Florence-2 + JoyCaption (le plus complet)

    ai_detection : si True, ajoute la detection IA + artefacts + metadata
      - sdxl-detector (Organika/sdxl-detector, ViT 350 Mo)
      - WD14 tags -> artefacts anatomiques
      - C2PA/IPTC/EXIF metadata
    """
    # Si l'utilisateur force le CPU, on masque le GPU pour TOUT (torch + onnx +
    # tous les captioners) avant le moindre import lourd.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    folder = Path(folder)
    if not folder.is_dir():
        return {"error": f"Folder not found: {folder}"}

    # Liste les images
    exts = (".png", ".jpg", ".jpeg", ".webp")
    images = sorted([f for f in folder.iterdir()
                     if f.is_file() and f.suffix.lower() in exts])

    if not images:
        return {"error": "Aucune image .png/.jpg/.jpeg/.webp trouvee."}

    print(f"STEP {len(images)} image(s) trouvée(s) — import des librairies (torch/insightface)...",
          file=sys.stderr, flush=True)
    # Import lourd ici (pour pouvoir retourner l'erreur avant si folder vide)
    try:
        import numpy as np
        from PIL import Image
        from insightface.app import FaceAnalysis
    except ImportError as e:
        return {"error": f"Module manquant : {e}"}

    # Resolution du device (GPU si dispo) - partage par insightface + CLIP
    torch_device, onnx_providers = resolve_device(device)
    print(f"STEP Device : {torch_device.upper()} "
          f"(onnx: {', '.join(p.replace('ExecutionProvider','') for p in onnx_providers)})",
          file=sys.stderr, flush=True)

    # Init insightface (utilise antelopev2 deja installe pour InstantID)
    # On cherche le dossier dans plusieurs emplacements possibles
    insight_roots = [
        r"C:\AI\ComfyUI-future\ComfyUI_windows_portable\ComfyUI\models\insightface",
        r"C:\AI\ComfyUI-Zluda\models\insightface",
        os.path.expanduser(r"~\.insightface"),
    ]
    insight_root = None
    for candidate in insight_roots:
        if os.path.isdir(os.path.join(candidate, "models", "antelopev2")):
            insight_root = candidate
            break

    if insight_root is None:
        return {
            "error": "Modeles antelopev2 introuvables. Cherche dans :\n" +
                     "\n".join(f"  - {p}\\models\\antelopev2" for p in insight_roots) +
                     "\nLance d'abord InstantID dans ComfyUI pour les telecharger, "
                     "ou copie les .onnx manuellement."
        }

    print(f"STEP Chargement du modèle de détection de visage (insightface antelopev2, {torch_device.upper()})...",
          file=sys.stderr, flush=True)
    try:
        # Redirige les prints d'insightface vers stderr pour pas polluer le JSON
        with redirect_stdout_to_stderr():
            app = FaceAnalysis(name="antelopev2",
                              root=insight_root,
                              providers=onnx_providers)
            # ctx_id >= 0 = GPU, -1 = CPU pour insightface
            ctx = 0 if torch_device == "cuda" else -1
            app.prepare(ctx_id=ctx, det_size=(640, 640))
        print("STEP Modèle visage prêt.", file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        return {"error": f"Init insightface ({insight_root}) : {e}\n\n{traceback.format_exc()[-500:]}"}

    # ===== Photo de reference (optionnelle) =====
    ref_embedding = None
    ref_info = None
    if ref_image:
        ref_path = Path(ref_image)
        if not ref_path.is_file():
            print(f"STEP Reference introuvable : {ref_path}", file=sys.stderr, flush=True)
        else:
            try:
                print(f"STEP Analyse photo de reference : {ref_path.name}", file=sys.stderr, flush=True)
                ref_pil = Image.open(ref_path).convert("RGB")
                ref_bgr = np.array(ref_pil)[:, :, ::-1]
                ref_faces = app.get(ref_bgr)
                if not ref_faces:
                    ref_info = {"error": f"Aucun visage detecte dans la reference ({ref_path.name})"}
                    print(f"STEP {ref_info['error']}", file=sys.stderr, flush=True)
                else:
                    # Plus gros visage = visage principal
                    ref_main = max(ref_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    ref_embedding = ref_main.normed_embedding
                    ref_info = {
                        "name": ref_path.name,
                        "path": str(ref_path),
                        "face_count_in_ref": len(ref_faces),
                    }
                    if len(ref_faces) > 1:
                        print(f"STEP Reference : {len(ref_faces)} visages detectes, on garde le plus gros", file=sys.stderr, flush=True)
            except Exception as e:
                ref_info = {"error": f"Erreur reference : {e}"}
                print(f"STEP {ref_info['error']}", file=sys.stderr, flush=True)

    # Init CLIP pour analyse corps + expressions (optionnel)
    clip_model = None
    clip_processor = None
    expression_labels = ["neutral expression", "slight smile", "big smile with teeth",
                         "laughing", "serious focused look", "surprised", "sad"]
    expr_text_features = None
    try:
        print(f"STEP Chargement CLIP pour analyse corps + expressions ({torch_device.upper()})...",
              file=sys.stderr, flush=True)
        with redirect_stdout_to_stderr():
            from transformers import CLIPModel, CLIPProcessor
            import torch
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()
            # Place le modele sur le GPU si dispo
            if torch_device == "cuda":
                clip_model = clip_model.to("cuda")
            # Pre-calcule les embeddings texte des expressions (une fois)
            with torch.no_grad():
                txt_inp = clip_processor(text=[f"a photo of a person with {e}" for e in expression_labels],
                                         return_tensors="pt", padding=True)
                if torch_device == "cuda":
                    txt_inp = {k: v.to("cuda") for k, v in txt_inp.items()}
                tf = clip_model.get_text_features(**txt_inp)
                # Selon la version de transformers, get_text_features peut retourner un tensor OU un objet
                if not hasattr(tf, "norm"):
                    if hasattr(tf, "text_embeds") and tf.text_embeds is not None:
                        tf = tf.text_embeds
                    elif hasattr(tf, "pooler_output") and tf.pooler_output is not None:
                        tf = tf.pooler_output
                    else:
                        tf = tf.last_hidden_state.mean(dim=1)
                expr_text_features = tf / tf.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"STEP CLIP indispo ({e}) - analyses corps/expression desactivees", file=sys.stderr, flush=True)
        clip_model = None

    # ===== Init WD14 tagger (pour captions Kohya / SDXL) =====
    sys.path.insert(0, str(Path(__file__).parent))
    wd14 = None
    if captioner_mode in ("wd14", "both"):
        try:
            print("STEP Chargement WD14 tagger (tags booru)...", file=sys.stderr, flush=True)
            from wd14_local import WD14Tagger
            with redirect_stdout_to_stderr():
                wd14 = WD14Tagger(threshold=0.35, character_threshold=0.85,
                                  exclude_tags=["1girl", "1boy", "solo"],
                                  replace_underscore=True)
        except Exception as e:
            print(f"STEP WD14 indispo ({e}) - tags booru desactives",
                  file=sys.stderr, flush=True)
            wd14 = None

    # ===== Init Florence-2 (captions naturelles - fallback) =====
    florence = None
    if captioner_mode in ("natural", "both", "all"):
        try:
            print("STEP Chargement Florence-2 (captions naturelles fallback)...",
                  file=sys.stderr, flush=True)
            from florence_local import Florence2Captioner
            with redirect_stdout_to_stderr():
                florence = Florence2Captioner(task="<DETAILED_CAPTION>")
        except Exception as e:
            print(f"STEP Florence-2 indispo ({e}) - captions Florence desactivees",
                  file=sys.stderr, flush=True)
            florence = None

    # ===== Init AI detector (Organika/sdxl-detector) =====
    ai_detector = None
    if ai_detection:
        try:
            print("STEP Chargement sdxl-detector (detection images IA)...",
                  file=sys.stderr, flush=True)
            from ai_detector_local import AIDetector
            with redirect_stdout_to_stderr():
                ai_detector = AIDetector(threshold=0.5)
        except Exception as e:
            print(f"STEP AI detector indispo ({e}) - detection IA desactivee",
                  file=sys.stderr, flush=True)
            ai_detector = None

    # Import les modules artefact + metadata (legers, pas de modele a charger)
    artifact_mod = None
    metadata_mod = None
    if ai_detection:
        try:
            import artifact_detector_local as artifact_mod
            import metadata_ai as metadata_mod
        except Exception as e:
            print(f"STEP Modules artefact/metadata indispo ({e})",
                  file=sys.stderr, flush=True)

    # ===== Init JoyCaption Beta One (captions naturelles STANDARD 2026) =====
    joycap = None
    if captioner_mode in ("joycaption", "all"):
        try:
            print("STEP Chargement JoyCaption Beta One (captions persona 2026)...",
                  file=sys.stderr, flush=True)
            from joycaption_local import JoyCaptioner
            with redirect_stdout_to_stderr():
                # INT4 si CUDA dispo (rentre en 4 Go VRAM)
                use_int4 = False
                try:
                    import torch
                    use_int4 = torch.cuda.is_available()
                except Exception:
                    pass
                joycap = JoyCaptioner(mode="descriptive_caption", use_int4=use_int4)
        except Exception as e:
            print(f"STEP JoyCaption indispo ({e}) - desactive",
                  file=sys.stderr, flush=True)
            joycap = None

    results = []
    embeddings = []
    body_embeddings = []  # Embeddings CLIP du corps (sous le visage)
    phashes = []  # Perceptual hashes pour duplicates detection
    total = len(images)

    # ===== Charge le cache d'analyse =====
    cache_old = load_cache(folder)
    cache_entries_old = cache_old.get("entries", {}) if cache_old else {}
    cache_new = {"version": CACHE_VERSION, "entries": {}}
    n_from_cache = 0
    n_new = 0
    if cache_entries_old:
        print(f"STEP Cache trouvé : {len(cache_entries_old)} photo(s) déjà analysée(s) "
              f"(les inchangées seront réutilisées)", file=sys.stderr, flush=True)
    else:
        print("STEP Aucun cache (1er scan de ce dossier) — tout sera analysé",
              file=sys.stderr, flush=True)

    # Annonce le total au debut sur stderr (pour la barre de progression GUI)
    print(f"TOTAL {total}", file=sys.stderr, flush=True)

    for idx, img_path in enumerate(images):
        # Progress line sur stderr (parsee par la GUI)
        print(f"PROGRESS {idx+1}/{total} {img_path.name}", file=sys.stderr, flush=True)
        # Path absolu pour preview live dans la GUI
        print(f"PREVIEW {img_path}", file=sys.stderr, flush=True)

        # === Cache hit ? ===
        fkey = _file_key(img_path)
        cached = cache_entries_old.get(fkey)
        if cached and "entry" in cached:
            # Reprend depuis le cache (on saute toute l'analyse)
            entry = OrderedDict(cached["entry"])
            # Path absolu peut avoir change (deplacement de dossier) - on met a jour
            entry["path"] = str(img_path)
            emb = cached.get("face_emb")
            body_emb = cached.get("body_emb")
            phash_v = cached.get("phash")
            embeddings.append(np.array(emb, dtype=np.float32) if emb else None)
            body_embeddings.append(np.array(body_emb, dtype=np.float32) if body_emb else None)
            phashes.append(int(phash_v) if phash_v else None)
            # Garde dans le nouveau cache
            cache_new["entries"][fkey] = cached
            results.append(entry)
            n_from_cache += 1
            # Mini-verdict pour preview live
            mini = {
                "name": entry.get("name"),
                "face_count": entry.get("face_count"),
                "face_proportion": entry.get("face_proportion"),
                "face_yaw": entry.get("face_yaw"),
                "sharpness": entry.get("sharpness"),
                "expression": entry.get("expression"),
                "ref_match": entry.get("ref_match"),
                "face_similarity_to_ref": entry.get("face_similarity_to_ref"),
                "quality_verdict": entry.get("quality_verdict"),
                "wd14_tags": (entry.get("wd14_tags") or "")[:200],
                "cached": True,
            }
            try:
                print(f"IMGINFO {json.dumps(mini, ensure_ascii=False)}", file=sys.stderr, flush=True)
            except Exception:
                pass
            continue

        n_new += 1
        entry = OrderedDict()
        entry["name"] = img_path.name
        entry["path"] = str(img_path)

        try:
            pil_img = Image.open(img_path).convert("RGB")
            entry["width"] = pil_img.width
            entry["height"] = pil_img.height

            # ===== Perceptual hash (pour detection duplicates) =====
            try:
                ph = dhash(pil_img)
                entry["phash"] = str(ph)
                phashes.append(ph)
            except Exception:
                phashes.append(None)
            entry["ratio"] = f"{pil_img.width}:{pil_img.height}"
            entry["megapixels"] = round((pil_img.width * pil_img.height) / 1_000_000, 2)
            entry["filesize_mb"] = round(img_path.stat().st_size / 1024 / 1024, 2)

            # ===== QUALITE PHOTO =====
            try:
                import cv2
                arr = np.array(pil_img)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                # Nettete : variance du Laplacien (plus c'est haut, plus c'est net)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                entry["sharpness"] = round(sharpness, 1)
                # Luminosite moyenne (0-255)
                brightness = float(gray.mean())
                entry["brightness"] = round(brightness, 1)
                # Contraste = ecart-type des niveaux de gris
                contrast = float(gray.std())
                entry["contrast"] = round(contrast, 1)
                # Verdict qualite
                quality_issues = []
                if sharpness < 100:
                    quality_issues.append("flou")
                if brightness < 40:
                    quality_issues.append("trop sombre")
                elif brightness > 220:
                    quality_issues.append("trop clair")
                if contrast < 25:
                    quality_issues.append("peu contraste")
                if entry["megapixels"] < 0.5:
                    quality_issues.append("basse res")
                if quality_issues:
                    entry["quality_verdict"] = " + ".join(quality_issues)
                elif sharpness > 500:
                    entry["quality_verdict"] = "Tres nette"
                else:
                    entry["quality_verdict"] = "OK"
            except Exception as e:
                entry["quality_verdict"] = f"err: {str(e)[:40]}"

            # Conversion en BGR pour insightface
            img_bgr = np.array(pil_img)[:, :, ::-1]
            faces = app.get(img_bgr)

            face_bbox = None
            face_pose = None  # yaw en degres (rotation gauche/droite)
            if not faces:
                entry["face_count"] = 0
                entry["face_status"] = "Aucun visage (plan large/dos ?)"
                entry["view_type"] = "body_only"  # pas de visage = vraisemblablement plan corps
                embeddings.append(None)
            else:
                entry["face_count"] = len(faces)
                # On garde la face avec le plus gros bbox (visage principal)
                main = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                face_bbox = main.bbox
                bbox_w = main.bbox[2] - main.bbox[0]
                bbox_h = main.bbox[3] - main.bbox[1]
                entry["face_size_px"] = int(bbox_w * bbox_h)
                proportion = (bbox_w * bbox_h) / (pil_img.width * pil_img.height) * 100
                entry["face_proportion"] = round(proportion, 1)
                # Classification du type de plan
                if proportion >= 15:
                    entry["view_type"] = "face_only"  # gros plan / headshot
                elif proportion >= 2:
                    entry["view_type"] = "both"  # plan moyen, visage + corps visibles
                else:
                    entry["view_type"] = "body_only"  # visage minuscule, c'est un plan corps
                entry["face_status"] = "OK" if entry["face_count"] == 1 else f"{entry['face_count']} visages"

                # === Detection orientation (yaw) via les 5 keypoints ===
                # kps : oeil_g, oeil_d, nez, coin_bouche_g, coin_bouche_d
                try:
                    if hasattr(main, "pose") and main.pose is not None:
                        # face.pose = [pitch, yaw, roll] en degres dans insightface 0.7+
                        face_pose = float(main.pose[1])
                    elif hasattr(main, "kps"):
                        kps = main.kps
                        # Yaw approx : decalage horizontal du nez par rapport au milieu des yeux
                        eye_mid_x = (kps[0][0] + kps[1][0]) / 2
                        nose_x = kps[2][0]
                        eye_dist = abs(kps[1][0] - kps[0][0])
                        if eye_dist > 0:
                            # ratio 0 = front, ratio +/- 0.5 = profil
                            ratio = (nose_x - eye_mid_x) / eye_dist
                            face_pose = float(ratio * 90)  # approx degres
                    entry["face_yaw"] = round(face_pose, 1) if face_pose is not None else None
                except Exception:
                    entry["face_yaw"] = None

                # Normalisation embedding pour cosine sim
                emb = main.normed_embedding
                embeddings.append(emb)

                # Stocke la bbox pour les crops futurs (Kohya)
                entry["_face_bbox"] = [float(face_bbox[0]), float(face_bbox[1]),
                                       float(face_bbox[2]), float(face_bbox[3])]

                # === Similarite avec la photo de reference ===
                if ref_embedding is not None:
                    try:
                        sim_ref = float(np.dot(emb, ref_embedding))
                        entry["face_similarity_to_ref"] = round(sim_ref, 3)
                        if sim_ref >= 0.5:
                            entry["ref_match"] = "OK"
                        elif sim_ref >= 0.35:
                            entry["ref_match"] = "douteux"
                        else:
                            entry["ref_match"] = "mauvaise personne"
                    except Exception:
                        entry["face_similarity_to_ref"] = None
                        entry["ref_match"] = None

            # ===== Detection EXPRESSION via CLIP (sur le crop visage) =====
            if clip_model is not None and face_bbox is not None:
                try:
                    # Crop autour du visage (avec un peu de marge)
                    margin = 0.15
                    fw = face_bbox[2] - face_bbox[0]
                    fh = face_bbox[3] - face_bbox[1]
                    x1 = max(0, int(face_bbox[0] - fw * margin))
                    y1 = max(0, int(face_bbox[1] - fh * margin))
                    x2 = min(pil_img.width, int(face_bbox[2] + fw * margin))
                    y2 = min(pil_img.height, int(face_bbox[3] + fh * margin))
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    with torch.no_grad():
                        inputs = clip_processor(images=face_crop, return_tensors="pt")
                        if torch_device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        face_feats = clip_model.get_image_features(**inputs)
                        if not hasattr(face_feats, "norm"):
                            if hasattr(face_feats, "image_embeds") and face_feats.image_embeds is not None:
                                face_feats = face_feats.image_embeds
                            elif hasattr(face_feats, "pooler_output") and face_feats.pooler_output is not None:
                                face_feats = face_feats.pooler_output
                            else:
                                face_feats = face_feats.last_hidden_state.mean(dim=1)
                        face_feats = face_feats / face_feats.norm(dim=-1, keepdim=True)
                        sims = (face_feats @ expr_text_features.T)[0]
                        top_idx = int(sims.argmax())
                        # Labels simplifies pour l'affichage
                        short_labels = ["neutre", "sourire leger", "grand sourire",
                                        "rire", "serieux", "surpris", "triste"]
                        entry["expression"] = short_labels[top_idx]
                        entry["expression_confidence"] = round(float(sims[top_idx]), 3)
                except Exception as e:
                    entry["expression"] = None

            # ===== Analyse CORPS via CLIP (zone sous le visage) =====
            if clip_model is not None:
                try:
                    # Crop zone corps : sous le visage si detecte, sinon image complete
                    if face_bbox is not None:
                        # Zone corps = de juste sous le menton jusqu'en bas, full largeur
                        y_top = int(face_bbox[3])  # bas du visage
                        if y_top < pil_img.height - 50:  # au moins 50 px disponibles
                            body_crop = pil_img.crop((0, y_top, pil_img.width, pil_img.height))
                        else:
                            body_crop = pil_img  # visage occupe presque tout, on prend l'image entiere
                    else:
                        body_crop = pil_img

                    # Embedding CLIP
                    with torch.no_grad():
                        inputs = clip_processor(images=body_crop, return_tensors="pt")
                        if torch_device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        feats = clip_model.get_image_features(**inputs)
                        if not hasattr(feats, "norm"):
                            if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                                feats = feats.image_embeds
                            elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                                feats = feats.pooler_output
                            else:
                                feats = feats.last_hidden_state.mean(dim=1)
                        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalise
                        body_embeddings.append(feats[0].cpu().numpy())
                except Exception as e:
                    entry["body_error"] = str(e)[:80]
                    body_embeddings.append(None)
            else:
                body_embeddings.append(None)

            # ===== WD14 captioning (tags booru) =====
            if wd14 is not None:
                try:
                    with redirect_stdout_to_stderr():
                        tags = wd14.tag(pil_img)
                    entry["wd14_tags"] = tags
                    # Ecrit le .txt a cote (format Kohya)
                    caption_path = img_path.with_suffix(".txt")
                    caption_path.write_text(tags, encoding="utf-8")
                except Exception as e:
                    entry["wd14_error"] = str(e)[:80]

            # NB : Florence-2 / JoyCaption (captions naturelles, LENTES) ne sont
            # PAS lancés ici. Ils tournent en PHASE 2, uniquement sur les photos
            # jugées viables (gros gain : on ne caption pas les ratés).

            # ===== Detection IA (sdxl-detector) =====
            if ai_detector is not None:
                try:
                    with redirect_stdout_to_stderr():
                        ai_result = ai_detector.detect(pil_img)
                    entry["ai_score"] = ai_result["ai_score"]
                    entry["ai_label"] = ai_result["label"]
                    entry["is_ai_classifier"] = ai_result["is_ai"]
                except Exception as e:
                    entry["ai_detector_error"] = str(e)[:80]

            # ===== Detection artefacts anatomiques (depuis WD14 + caption) =====
            if artifact_mod is not None:
                try:
                    art_result = artifact_mod.combined_artifact_detection(entry)
                    if art_result["has_artifacts"]:
                        entry["artifacts_categories"] = art_result["artifacts_categories"]
                        entry["artifacts_severity"] = art_result["artifacts_severity"]
                        # Detail dans un sous-champ pour debug
                        entry["_artifacts_detail"] = {
                            "wd14": art_result["artifacts_from_wd14"],
                            "caption": art_result["artifacts_from_caption"],
                        }
                except Exception as e:
                    entry["artifact_detector_error"] = str(e)[:80]

            # ===== Lecture metadata IA (C2PA/IPTC/EXIF/filename) =====
            if metadata_mod is not None:
                try:
                    md = metadata_mod.detect_ai_metadata(img_path)
                    if md["has_ai_metadata"]:
                        entry["ai_metadata_sources"] = md["sources"]
                        entry["ai_metadata_confidence"] = md["confidence"]
                except Exception as e:
                    entry["metadata_error"] = str(e)[:80]

            # ===== Emission mini-verdict pour le panneau live de la GUI =====
            mini = {
                "name": entry.get("name"),
                "face_count": entry.get("face_count"),
                "face_proportion": entry.get("face_proportion"),
                "face_yaw": entry.get("face_yaw"),
                "sharpness": entry.get("sharpness"),
                "expression": entry.get("expression"),
                "ref_match": entry.get("ref_match"),
                "face_similarity_to_ref": entry.get("face_similarity_to_ref"),
                "quality_verdict": entry.get("quality_verdict"),
                "wd14_tags": (entry.get("wd14_tags") or "")[:200],
                "joycaption": (entry.get("joycaption") or "")[:200],
                "ai_score": entry.get("ai_score"),
                "artifacts_severity": entry.get("artifacts_severity"),
                "artifacts_categories": entry.get("artifacts_categories"),
                "ai_metadata_sources": entry.get("ai_metadata_sources"),
            }
            try:
                print(f"IMGINFO {json.dumps(mini, ensure_ascii=False)}",
                      file=sys.stderr, flush=True)
            except Exception:
                pass

            # ===== Ecriture cache pour cette image =====
            face_emb_list = embeddings[-1].tolist() if embeddings[-1] is not None else None
            body_emb_list = body_embeddings[-1].tolist() if body_embeddings[-1] is not None else None
            cache_new["entries"][fkey] = {
                "entry": dict(entry),
                "face_emb": face_emb_list,
                "body_emb": body_emb_list,
                "phash": str(phashes[-1]) if phashes[-1] is not None else None,
            }

            results.append(entry)

        except Exception as e:
            entry["error"] = str(e)
            embeddings.append(None)
            # phashes deja appende avant les except potentiel -- mais si Image.open foire,
            # il n'a pas ete ajoute : assure la coherence
            if len(phashes) <= idx:
                phashes.append(None)
            results.append(entry)

        # ===== Sauvegarde INCREMENTALE du cache (tous les 10 images) =====
        # Ainsi, meme si l'analyse est interrompue (fermeture, crash, timeout),
        # les photos deja analysees sont memorisees et ne seront pas refaites.
        if (idx + 1) % 10 == 0:
            save_cache(folder, cache_new)

    # Flush du cache apres la boucle (capture les dernieres images < palier de 10)
    save_cache(folder, cache_new)

    print("PROGRESS_DONE", file=sys.stderr, flush=True)
    if n_from_cache > 0:
        print(f"STEP Cache : {n_from_cache} reutilisees, {n_new} nouvellement analysees",
              file=sys.stderr, flush=True)

    # ===== Recalcul ref_match pour TOUTES les entries (cas changement de ref) =====
    if ref_embedding is not None:
        for i, emb in enumerate(embeddings):
            if emb is None:
                results[i].pop("face_similarity_to_ref", None)
                results[i].pop("ref_match", None)
                continue
            sim_ref = float(np.dot(emb, ref_embedding))
            results[i]["face_similarity_to_ref"] = round(sim_ref, 3)
            if sim_ref >= 0.5:
                results[i]["ref_match"] = "OK"
            elif sim_ref >= 0.35:
                results[i]["ref_match"] = "douteux"
            else:
                results[i]["ref_match"] = "mauvaise personne"
    else:
        # Pas de ref : nettoie les anciennes valeurs cachees
        for r in results:
            r.pop("face_similarity_to_ref", None)
            r.pop("ref_match", None)

    print("STEP Calcul matrice de similarite...", file=sys.stderr, flush=True)

    # Matrice de similarite (uniquement images avec face)
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
    similarities_per_image = {}
    overall_coherence = None

    if len(valid_idx) >= 2:
        emb_mat = np.array([embeddings[i] for i in valid_idx])
        sim_matrix = emb_mat @ emb_mat.T  # cosine sim car deja normalise

        # Pour chaque image, similarite moyenne avec les autres
        for pos, i in enumerate(valid_idx):
            others = np.delete(sim_matrix[pos], pos)
            avg = float(np.mean(others))
            min_sim = float(np.min(others))
            results[i]["face_similarity_avg"] = round(avg, 3)
            results[i]["face_similarity_min"] = round(min_sim, 3)
            # Verdict adapte selon le type de plan
            view = results[i].get("view_type", "both")
            if view == "body_only":
                # Visage tout petit, similarite peu fiable
                results[i]["verdict"] = "Visage tres petit"
            elif avg < 0.3:
                results[i]["verdict"] = "OUTLIER (visage different)"
            elif avg < 0.5:
                results[i]["verdict"] = "Faible coherence"
            else:
                results[i]["verdict"] = "OK"

        overall_coherence = round(float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])), 3)

    # Verdict pour les images sans visage detecte
    for r in results:
        if "verdict" not in r:
            if r.get("view_type") == "body_only":
                r["verdict"] = "Plan corps / dos"
            else:
                r["verdict"] = "Aucun visage"

    # ===== Detection DUPLICATES (pHash + face sim) =====
    print("STEP Detection des duplicates...", file=sys.stderr, flush=True)
    duplicates_groups = []  # Liste de listes d'index
    if len(results) >= 2:
        assigned = [False] * len(results)
        for i in range(len(results)):
            if assigned[i]:
                continue
            group = [i]
            for j in range(i + 1, len(results)):
                if assigned[j]:
                    continue
                # 1) Quasi-identiques visuels (pHash)
                ph_i = phashes[i] if i < len(phashes) else None
                ph_j = phashes[j] if j < len(phashes) else None
                is_dup = False
                if ph_i is not None and ph_j is not None:
                    hd = hamming_distance(ph_i, ph_j)
                    if hd < 5:
                        is_dup = True
                        results[j]["phash_distance_to_dup"] = hd
                # 2) OU visage quasi-identique (rare mais possible)
                if not is_dup and embeddings[i] is not None and embeddings[j] is not None:
                    face_sim = float(np.dot(embeddings[i], embeddings[j]))
                    if face_sim > 0.96:
                        is_dup = True
                        results[j]["face_sim_to_dup"] = round(face_sim, 3)
                if is_dup:
                    group.append(j)
                    assigned[j] = True
            if len(group) >= 2:
                # Marque tous sauf le premier comme duplicate
                keeper = results[group[0]]["name"]
                duplicates_groups.append([results[k]["name"] for k in group])
                for k in group[1:]:
                    results[k]["duplicate_of"] = keeper
                assigned[group[0]] = True

    # ===== Verdict VIABILITE LoRA (visage) =====
    for r in results:
        reasons = []
        viable = "yes"  # par defaut

        # Duplicate = exclu d'office
        if r.get("duplicate_of"):
            viable = "no"
            reasons.append(f"duplicate de {r['duplicate_of']}")

        # Artefacts anatomiques severes = exclu (mains/yeux pourris polluent le LoRA)
        art_sev = r.get("artifacts_severity")
        art_cats = r.get("artifacts_categories", [])
        if art_sev == "high":
            viable = "no"
            reasons.append(f"artefacts IA : {', '.join(art_cats)}")
        elif art_sev == "medium":
            if viable == "yes":
                viable = "borderline"
            reasons.append(f"artefacts : {', '.join(art_cats)}")

        # Qualite : flou et basse res = killer
        sharpness = r.get("sharpness")
        if sharpness is not None and sharpness < 100:
            viable = "no"
            reasons.append("floue")
        if r.get("megapixels", 1) < 0.4:
            viable = "no"
            reasons.append("basse res")

        # Pas de visage detecte = inutile pour LoRA visage
        if r.get("face_count", 0) == 0:
            viable = "no"
            reasons.append("visage absent (dos ?)")
        elif r.get("face_count", 0) > 1:
            viable = "no"
            reasons.append("plusieurs visages")

        # Si on a une photo de reference, on prend ca comme verite (beaucoup plus fiable)
        sim_ref = r.get("face_similarity_to_ref")
        if sim_ref is not None:
            if sim_ref < 0.35:
                viable = "no"
                reasons.append(f"mauvaise personne (ref {sim_ref:.2f})")
            elif sim_ref < 0.5:
                if viable == "yes":
                    viable = "borderline"
                reasons.append(f"douteux vs ref ({sim_ref:.2f})")
        else:
            # Pas de ref : fallback sur la coherence interne du dataset
            sim = r.get("face_similarity_avg")
            if sim is not None and sim < 0.3:
                viable = "no"
                reasons.append("mauvaise personne")

        # Visage minuscule (< 1%) sur plan large = peu utile en face LoRA
        prop = r.get("face_proportion", 0)
        if prop > 0 and prop < 1:
            if viable == "yes":
                viable = "borderline"
            reasons.append("visage minuscule")

        # Profil/dos via yaw
        yaw = r.get("face_yaw")
        if yaw is not None:
            abs_yaw = abs(yaw)
            if abs_yaw > 75:
                viable = "no"
                reasons.append(f"presque de dos ({yaw:.0f} deg)")
            elif abs_yaw > 50:
                if viable == "yes":
                    viable = "borderline"
                reasons.append(f"profil ({yaw:.0f} deg)")

        # Sombre / sur-expose
        bright = r.get("brightness")
        if bright is not None:
            if bright < 40:
                viable = "no"
                reasons.append("trop sombre")
            elif bright > 220 and viable == "yes":
                viable = "borderline"
                reasons.append("sur-expose")

        # Faible coherence visage (mais pas outlier total) - skip si on a une ref
        if sim_ref is None:
            sim = r.get("face_similarity_avg")
            if sim is not None and 0.3 <= sim < 0.5 and viable == "yes":
                viable = "borderline"
                reasons.append(f"coherence faible (sim {sim:.2f})")

        r["lora_viable"] = viable
        r["lora_reason"] = ", ".join(reasons) if reasons else "OK pour LoRA"

    # ===== PHASE 2 : captions naturelles (LENTES) sur les VIABLES uniquement =====
    # On ne lance Florence/JoyCaption que sur les photos jugées viables/borderline,
    # ce qui évite de gaspiller des minutes sur des photos qui seront jetées.
    if florence is not None or joycap is not None:
        from PIL import Image as _ImgP2
        pending = []
        for r in results:
            if r.get("error"):
                continue
            if r.get("lora_viable") not in ("yes", "borderline"):
                continue
            need = False
            if joycap is not None and not r.get("joycaption"):
                need = True
            if florence is not None and not r.get("natural_caption") and joycap is None:
                need = True
            if need:
                pending.append(r)

        if pending:
            n_skipped = sum(1 for r in results
                            if r.get("lora_viable") == "no" and not r.get("error"))
            print(f"STEP Phase 2 : captions naturelles sur {len(pending)} viable(s) "
                  f"({n_skipped} raté(s) ignoré(s) = temps économisé)",
                  file=sys.stderr, flush=True)
            # Reset de la barre de progression pour la phase 2
            print(f"TOTAL {len(pending)}", file=sys.stderr, flush=True)
            print("STEP 🖼 Génération des captions naturelles (phase 2, viables seulement)…",
                  file=sys.stderr, flush=True)

            for j, r in enumerate(pending):
                print(f"PROGRESS {j+1}/{len(pending)} {r.get('name')}",
                      file=sys.stderr, flush=True)
                print(f"PREVIEW {r.get('path')}", file=sys.stderr, flush=True)
                try:
                    pil_p2 = _ImgP2.open(r["path"]).convert("RGB")
                except Exception as e:
                    r["caption_error"] = f"open: {str(e)[:60]}"
                    continue
                p = Path(r["path"])
                # Florence-2 (si actif et pas deja fait)
                if florence is not None and not r.get("natural_caption"):
                    try:
                        with redirect_stdout_to_stderr():
                            cap = florence.caption(pil_p2)
                        r["natural_caption"] = cap
                        p.with_suffix(".nat.txt").write_text(cap, encoding="utf-8")
                    except Exception as e:
                        r["florence_error"] = str(e)[:80]
                # JoyCaption (si actif et pas deja fait)
                if joycap is not None and not r.get("joycaption"):
                    try:
                        with redirect_stdout_to_stderr():
                            cap = joycap.caption(pil_p2)
                        r["joycaption"] = cap
                        p.with_suffix(".joy.txt").write_text(cap, encoding="utf-8")
                        if captioner_mode == "joycaption":
                            r["natural_caption"] = cap
                    except Exception as e:
                        r["joycaption_error"] = str(e)[:80]

                # Mini-verdict live (montre la caption qui vient d'etre generee)
                mini2 = {
                    "name": r.get("name"),
                    "face_count": r.get("face_count"),
                    "face_proportion": r.get("face_proportion"),
                    "face_yaw": r.get("face_yaw"),
                    "sharpness": r.get("sharpness"),
                    "expression": r.get("expression"),
                    "ref_match": r.get("ref_match"),
                    "face_similarity_to_ref": r.get("face_similarity_to_ref"),
                    "quality_verdict": r.get("quality_verdict"),
                    "wd14_tags": (r.get("wd14_tags") or "")[:200],
                    "joycaption": (r.get("joycaption") or r.get("natural_caption") or "")[:200],
                    "ai_score": r.get("ai_score"),
                    "artifacts_severity": r.get("artifacts_severity"),
                    "artifacts_categories": r.get("artifacts_categories"),
                }
                try:
                    print(f"IMGINFO {json.dumps(mini2, ensure_ascii=False)}",
                          file=sys.stderr, flush=True)
                except Exception:
                    pass

                # Sauvegarde incrementale : memorise la caption dans le cache
                # pour ne pas la regenerer si l'analyse est interrompue ensuite.
                try:
                    p_img = Path(r["path"])
                    fk2 = _file_key(p_img)
                    if fk2 in cache_new["entries"]:
                        cache_new["entries"][fk2]["entry"] = dict(r)
                    if (j + 1) % 10 == 0:
                        save_cache(folder, cache_new)
                except Exception:
                    pass

            save_cache(folder, cache_new)
            print("PROGRESS_DONE", file=sys.stderr, flush=True)
            print("STEP Captions naturelles terminées.", file=sys.stderr, flush=True)

    # ===== Matrice de similarite CORPS (CLIP) =====
    overall_body_coherence = None
    valid_body_idx = [i for i, e in enumerate(body_embeddings) if e is not None]
    if len(valid_body_idx) >= 2:
        body_mat = np.array([body_embeddings[i] for i in valid_body_idx])
        body_sim = body_mat @ body_mat.T
        for pos, i in enumerate(valid_body_idx):
            others = np.delete(body_sim[pos], pos)
            avg_b = float(np.mean(others))
            # N'enregistre la sim corps que si le corps est REELLEMENT visible
            view = results[i].get("view_type", "both")
            if view == "face_only":
                # Close-up : corps non visible, on note "NA"
                results[i]["body_similarity_avg"] = None
                results[i]["body_verdict"] = "NA (close-up)"
            else:
                results[i]["body_similarity_avg"] = round(avg_b, 3)
                if avg_b < 0.7:
                    results[i]["body_verdict"] = "Corps tres different"
                elif avg_b < 0.82:
                    results[i]["body_verdict"] = "Corps variable"
                else:
                    results[i]["body_verdict"] = "Corps coherent"
        # Coherence globale corps : uniquement sur les images ou le corps est visible
        body_visible_idx = [j for j, i in enumerate(valid_body_idx)
                           if results[i].get("view_type") != "face_only"]
        if len(body_visible_idx) >= 2:
            sub = body_sim[np.ix_(body_visible_idx, body_visible_idx)]
            overall_body_coherence = round(float(np.mean(sub[np.triu_indices_from(sub, k=1)])), 3)

    # Stats globales
    widths = [r.get("width") for r in results if r.get("width")]
    heights = [r.get("height") for r in results if r.get("height")]

    n_viable = sum(1 for r in results if r.get("lora_viable") == "yes")
    n_border = sum(1 for r in results if r.get("lora_viable") == "borderline")
    n_unusable = sum(1 for r in results if r.get("lora_viable") == "no")
    n_duplicates = sum(1 for r in results if r.get("duplicate_of"))
    n_captioned = sum(1 for r in results if r.get("wd14_tags"))
    # === Stats IA / artefacts ===
    n_ai_classifier = sum(1 for r in results if r.get("is_ai_classifier"))
    n_ai_metadata = sum(1 for r in results if r.get("ai_metadata_sources"))
    n_artifacts_high = sum(1 for r in results if r.get("artifacts_severity") == "high")
    n_artifacts_medium = sum(1 for r in results if r.get("artifacts_severity") == "medium")
    ai_scores = [r.get("ai_score") for r in results if r.get("ai_score") is not None]
    ai_score_avg = round(sum(ai_scores) / len(ai_scores), 3) if ai_scores else None

    # Distribution des expressions (sur images viables ou borderline seulement)
    expr_counts = {}
    for r in results:
        if r.get("lora_viable") in ("yes", "borderline") and r.get("expression"):
            expr_counts[r["expression"]] = expr_counts.get(r["expression"], 0) + 1

    # Stats reference
    ref_match_stats = None
    if ref_embedding is not None:
        sims_to_ref = [r.get("face_similarity_to_ref") for r in results
                       if r.get("face_similarity_to_ref") is not None]
        if sims_to_ref:
            n_wrong = sum(1 for s in sims_to_ref if s < 0.35)
            n_doubt = sum(1 for s in sims_to_ref if 0.35 <= s < 0.5)
            n_ok = sum(1 for s in sims_to_ref if s >= 0.5)
            ref_match_stats = {
                "avg": round(sum(sims_to_ref) / len(sims_to_ref), 3),
                "min": round(min(sims_to_ref), 3),
                "max": round(max(sims_to_ref), 3),
                "ok": n_ok,
                "doubt": n_doubt,
                "wrong": n_wrong,
            }

    summary = {
        "total_images": len(images),
        "with_face": sum(1 for r in results if r.get("face_count", 0) > 0),
        "no_face": sum(1 for r in results if r.get("face_count", 0) == 0),
        "multiple_faces": sum(1 for r in results if r.get("face_count", 0) > 1),
        "overall_face_coherence": overall_coherence,  # 0 (different) -> 1 (identique)
        "overall_body_coherence": overall_body_coherence,  # 0.5 -> 1.0 typiquement (CLIP)
        "resolution_min": f"{min(widths)}x{min(heights)}" if widths else None,
        "resolution_max": f"{max(widths)}x{max(heights)}" if widths else None,
        "lora_viable": n_viable,
        "lora_borderline": n_border,
        "lora_unusable": n_unusable,
        "duplicates_count": n_duplicates,
        "duplicates_groups": duplicates_groups,
        "captions_written": n_captioned,
        "ai_classifier_count": n_ai_classifier,
        "ai_metadata_count": n_ai_metadata,
        "ai_score_avg": ai_score_avg,
        "artifacts_high_count": n_artifacts_high,
        "artifacts_medium_count": n_artifacts_medium,
        "expressions": expr_counts,
        "verdict": None,  # rempli plus bas (apres calcul du grade)
        "reference": ref_info,
        "reference_match": ref_match_stats,
    }

    # Recommandations
    recommendations = []
    # === Artefacts IA (CRITIQUE - polluent le LoRA) ===
    if n_artifacts_high > 0:
        recommendations.append(
            f"❌ {n_artifacts_high} photo(s) avec artefacts IA severes (mains/yeux pourris) — "
            f"a virer absolument, le LoRA va apprendre les defauts"
        )
    if n_artifacts_medium > 0:
        recommendations.append(
            f"⚠️ {n_artifacts_medium} photo(s) avec artefacts moyens (anatomie, membres) — "
            f"a verifier individuellement"
        )
    # === Images IA non taguees (pollution si dataset suppose etre photo reelle) ===
    if n_ai_classifier > 0:
        pct = round(n_ai_classifier * 100 / len(results), 0) if results else 0
        if pct >= 50:
            recommendations.append(
                f"💡 {n_ai_classifier}/{len(results)} ({pct:.0f}%) photos detectees IA-generated — "
                f"normal si dataset InstantID/Flux. Sinon attention au mix."
            )
        else:
            recommendations.append(
                f"⚠️ {n_ai_classifier} photo(s) detectee(s) IA-generated dans un dataset suppose reel — "
                f"verifie via double-clic, pollution possible"
            )
    if n_ai_metadata > 0:
        recommendations.append(
            f"📋 {n_ai_metadata} photo(s) avec metadata IA explicite (C2PA/EXIF/filename)"
        )
    # === Duplicates ===
    if n_duplicates > 0:
        recommendations.append(f"❌ {n_duplicates} duplicate(s) detecte(s) - overfit garanti, a virer absolument")
    # === Captions WD14 ===
    if n_captioned > 0:
        recommendations.append(f"💡 {n_captioned} caption(s) Kohya generee(s) (.txt a cote des images)")
    # === Reference (si presente) ===
    if ref_info and "error" in ref_info:
        recommendations.append(f"⚠️ Reference inutilisable : {ref_info['error']}")
    elif ref_match_stats:
        ok, doubt, wrong = ref_match_stats["ok"], ref_match_stats["doubt"], ref_match_stats["wrong"]
        avg = ref_match_stats["avg"]
        if wrong == 0 and doubt == 0:
            recommendations.append(f"✅ Identite verifiee (vs ref) : {ok} photos correspondent (avg {avg})")
        elif wrong == 0:
            recommendations.append(f"💡 Identite OK avec {doubt} douteuses (vs ref, avg {avg})")
        else:
            recommendations.append(f"❌ {wrong} photo(s) ne correspondent PAS a la reference - a virer absolument")
            if doubt > 0:
                recommendations.append(f"⚠️ {doubt} photo(s) douteuse(s) vs reference - a verifier manuellement")
    if summary["no_face"] > 0:
        recommendations.append(f"⚠️ {summary['no_face']} image(s) sans visage detecte - a retirer pour LoRA visage")
    if summary["multiple_faces"] > 0:
        recommendations.append(f"⚠️ {summary['multiple_faces']} image(s) avec plusieurs visages - risque de confusion")
    if overall_coherence is not None:
        if overall_coherence < 0.4:
            recommendations.append(f"❌ Coherence visage tres faible ({overall_coherence}) - le dataset melange plusieurs personnes")
        elif overall_coherence < 0.6:
            recommendations.append(f"⚠️ Coherence visage moyenne ({overall_coherence}) - certaines images sont differentes")
        else:
            recommendations.append(f"✅ Coherence visage OK ({overall_coherence})")
    if overall_body_coherence is not None:
        if overall_body_coherence < 0.65:
            recommendations.append(f"⚠️ Coherence corps faible ({overall_body_coherence}) - silhouettes/proportions tres differentes")
        elif overall_body_coherence < 0.78:
            recommendations.append(f"💡 Coherence corps moyenne ({overall_body_coherence}) - normal si tenues/poses varient beaucoup")
        else:
            recommendations.append(f"✅ Coherence corps OK ({overall_body_coherence})")
    if widths and (max(widths) - min(widths) > 200 or max(heights) - min(heights) > 200):
        recommendations.append("⚠️ Resolutions tres variees - prefere uniformiser (1024x1024 typique)")

    # Stats qualite
    blurry_count = sum(1 for r in results if r.get("sharpness", 999) < 100)
    dark_count = sum(1 for r in results if r.get("brightness", 128) < 40)
    bright_count = sum(1 for r in results if r.get("brightness", 128) > 220)
    if blurry_count > 0:
        recommendations.append(f"❌ {blurry_count} image(s) floue(s) (nettete < 100) - a virer")
    if dark_count > 0:
        recommendations.append(f"⚠️ {dark_count} image(s) trop sombre(s) - eclairage insuffisant")
    if bright_count > 0:
        recommendations.append(f"⚠️ {bright_count} image(s) sur-exposee(s) - blanc cassant")
    sharp_imgs = [r.get("sharpness", 0) for r in results if r.get("sharpness")]
    if sharp_imgs:
        avg_sharp = sum(sharp_imgs) / len(sharp_imgs)
        if avg_sharp > 300:
            recommendations.append(f"✅ Qualite moyenne du dataset : nette (avg {avg_sharp:.0f})")
        elif avg_sharp > 150:
            recommendations.append(f"💡 Qualite moyenne : acceptable (avg {avg_sharp:.0f})")
        else:
            recommendations.append(f"⚠️ Qualite moyenne : floue (avg {avg_sharp:.0f}) - dataset trop mou")

    # Stats viabilite LoRA = la conclusion finale
    total = len(images)
    if total > 0:
        recommendations.append(f"🧬 Pour LoRA visage : {n_viable} viable, {n_border} borderline, {n_unusable} a virer")
        if n_viable < 15:
            recommendations.append(f"⚠️ Seulement {n_viable} images vraiment viables - vise 20+ pour un LoRA stable")
        elif n_viable >= 20:
            recommendations.append(f"✅ Dataset suffisant pour entrainer ({n_viable} images viables)")

    # ===== VERDICT GLOBAL : note + plan d'action =====
    target_min = 20  # minimum viable pour un LoRA stable
    target_ideal = 30  # ideal
    n_kept_after_cleanup = n_viable + n_border  # ce qu'on garde si on vire que les ❌

    # Note de qualite globale
    if n_viable == 0:
        grade = "F"
        grade_desc = "INUTILISABLE"
    elif n_viable < 10:
        grade = "D"
        grade_desc = "TROP MAIGRE"
    elif n_viable < 15:
        grade = "C"
        grade_desc = "INSUFFISANT"
    elif n_viable < target_min:
        grade = "B-"
        grade_desc = "LIMITE"
    elif n_viable < target_ideal:
        grade = "B+"
        grade_desc = "BON"
    elif n_viable >= target_ideal and (overall_coherence or 0) > 0.6:
        grade = "A"
        grade_desc = "EXCELLENT"
    else:
        grade = "B"
        grade_desc = "OK"

    # Plan d'action concret
    action_lines = []
    if n_unusable > 0:
        action_lines.append(f"Vire les {n_unusable} photo(s) marquees a virer → il restera {n_kept_after_cleanup} photos exploitables")
    if n_viable < target_min:
        manque = target_min - n_viable
        action_lines.append(f"Genere {manque} photo(s) supplementaire(s) pour atteindre le minimum de {target_min}")
    elif n_viable < target_ideal:
        manque = target_ideal - n_viable
        action_lines.append(f"Optionnel : ajoute {manque} photo(s) pour viser l'ideal de {target_ideal}")
    else:
        action_lines.append(f"Dataset suffisant : tu peux lancer le training Kohya SS")

    summary_verdict = {
        "grade": grade,
        "grade_desc": grade_desc,
        "actions": action_lines,
        "after_cleanup": n_kept_after_cleanup,
        "viable_now": n_viable,
        "target_min": target_min,
        "target_ideal": target_ideal,
    }
    summary["verdict"] = summary_verdict  # injecte le verdict dans le summary

    # Analyse de la diversite d'expression
    if expr_counts:
        total_with_expr = sum(expr_counts.values())
        expr_summary = ", ".join(f"{k} ({v})" for k, v in sorted(expr_counts.items(), key=lambda x: -x[1]))
        recommendations.append(f"😊 Expressions detectees : {expr_summary}")
        # Verifie le dominance
        top_expr, top_count = max(expr_counts.items(), key=lambda x: x[1])
        if top_count / total_with_expr > 0.7:
            recommendations.append(f"⚠️ Trop d'une seule expression ({top_expr} = {top_count/total_with_expr:.0%}) - le LoRA risque d'etre biaise")
        elif len(expr_counts) < 2:
            recommendations.append("⚠️ Une seule expression detectee - ajoute des sourires/serieux pour varier")
        elif len(expr_counts) >= 3 and top_count / total_with_expr < 0.6:
            recommendations.append("✅ Bonne diversite d'expressions (le LoRA generalisera bien)")
    if len(images) < 15:
        recommendations.append(f"⚠️ Seulement {len(images)} images - vise 20-30 minimum pour un LoRA stable")
    elif len(images) > 50:
        recommendations.append(f"💡 {len(images)} images - peut-etre trop, 25-40 suffit souvent")

    # ===== Analyse FREQUENCE TAGS (alerte overfit attributs accessoires) =====
    # Si un tag accessoire (vetement, fond, lumiere) apparait dans >70% des
    # photos viables, le LoRA va le memoriser comme "fait partie de la persona".
    tag_freq = {}
    tag_overfit_alerts = []
    overfit_viable = [r for r in results if r.get("lora_viable") in ("yes", "borderline")]
    n_with_tags = sum(1 for r in overfit_viable if r.get("wd14_tags"))
    if n_with_tags >= 5:
        # Comptage
        for r in overfit_viable:
            tags_str = r.get("wd14_tags") or ""
            seen = set()
            for raw in tags_str.split(","):
                t = raw.strip().lower()
                if not t or t in seen:
                    continue
                seen.add(t)
                tag_freq[t] = tag_freq.get(t, 0) + 1

        # Categories d'attributs "risque overfit" (apparence accessoire, pas identite)
        ACCESSORY_PATTERNS = {
            "vetement": ["shirt", "dress", "jacket", "coat", "blouse", "sweater",
                          "hoodie", "tshirt", "t-shirt", "top", "skirt", "pants",
                          "jeans", "shorts", "bra", "bikini", "uniform"],
            "fond": ["indoors", "outdoors", "background", "wall", "room", "kitchen",
                      "bedroom", "bathroom", "studio", "street", "park", "beach",
                      "forest", "sky", "garden", "office", "cafe", "interior"],
            "lumiere": ["lighting", "sunlight", "backlight", "shadow", "dim",
                         "bright", "dark", "neon", "golden hour", "natural light"],
            "accessoire": ["earrings", "necklace", "glasses", "sunglasses", "hat",
                            "scarf", "watch", "ring", "bracelet", "tie", "bag",
                            "headphones", "mask"],
            "couleur_fond": ["white background", "black background", "grey background",
                              "blue background", "plain background", "simple background"],
        }

        # Sort par frequence decroissante
        sorted_tags = sorted(tag_freq.items(), key=lambda x: -x[1])

        # Top tags les plus frequents (info)
        top_tags = [(t, c, round(c / n_with_tags, 2)) for t, c in sorted_tags[:20]]

        # Alertes overfit
        for tag, count in sorted_tags:
            ratio = count / n_with_tags
            if ratio < 0.5:
                break  # tries decroissant, plus rien d'utile en dessous

            # Quelle categorie ?
            category = None
            for cat, patterns in ACCESSORY_PATTERNS.items():
                if any(p in tag for p in patterns):
                    category = cat
                    break
            if category is None:
                continue  # tag identite ("woman", "long hair") - on ignore

            severity = "ALERTE" if ratio >= 0.75 else "ATTENTION"
            tag_overfit_alerts.append({
                "tag": tag, "category": category,
                "count": count, "ratio": round(ratio, 2),
                "severity": severity,
            })

    # ===== Distribution & suggestions de generation ciblees =====
    viable_imgs = [r for r in results if r.get("lora_viable") in ("yes", "borderline")]
    distribution = {}
    next_to_generate = []
    if viable_imgs:
        # --- Angles ---
        front = sum(1 for r in viable_imgs if r.get("face_yaw") is not None and abs(r["face_yaw"]) < 15)
        three_q = sum(1 for r in viable_imgs if r.get("face_yaw") is not None and 15 <= abs(r["face_yaw"]) < 50)
        profil = sum(1 for r in viable_imgs if r.get("face_yaw") is not None and 50 <= abs(r["face_yaw"]))
        n_with_yaw = front + three_q + profil
        # --- Plans (face_only / both / body_only) ---
        face_only = sum(1 for r in viable_imgs if r.get("view_type") == "face_only")
        both_v = sum(1 for r in viable_imgs if r.get("view_type") == "both")
        body_only = sum(1 for r in viable_imgs if r.get("view_type") == "body_only")
        # --- Expressions ---
        expr_dist = {}
        for r in viable_imgs:
            e = r.get("expression")
            if e:
                expr_dist[e] = expr_dist.get(e, 0) + 1

        distribution = {
            "angles": {"front": front, "three_quarter": three_q, "profil": profil},
            "plans": {"face_only": face_only, "both": both_v, "body_only": body_only},
            "expressions": expr_dist,
        }

        # === Suggestions concretes ===
        n_viable_total = len(viable_imgs)
        # Angles
        if n_with_yaw >= 5:
            front_pct = front / n_with_yaw
            tq_pct = three_q / n_with_yaw
            profil_pct = profil / n_with_yaw
            # Cible : ~55% front, ~30% 3/4, ~10% profil
            if front_pct > 0.75:
                manque = max(3, int(n_viable_total * 0.3) - three_q)
                next_to_generate.append(f"Genere {manque} photos de 3/4 (yaw 20-45°) — trop de face actuellement ({front_pct*100:.0f}%)")
            if tq_pct < 0.15 and three_q < 4:
                next_to_generate.append(f"Genere 4-6 photos de 3/4 (yaw 20-45°) — il en manque, {three_q} actuellement")
            if profil < 2 and n_viable_total >= 15:
                next_to_generate.append("Genere 2-3 photos de profil (yaw 60-80°) pour donner l'angle au LoRA")

        # Plans
        if face_only > 0 and both_v == 0 and body_only == 0 and n_viable_total >= 10:
            next_to_generate.append(f"Tout est en close-up : genere 5-8 plans moyens (visage + corps visibles)")
        if both_v == 0 and n_viable_total >= 15:
            next_to_generate.append("Genere 4-6 plans moyens (mi-corps, visage visible) pour varier les plans")
        if body_only == 0 and n_viable_total >= 20:
            next_to_generate.append("Optionnel : 2-3 plans larges pour que le LoRA apprenne aussi la silhouette")

        # Expressions
        if expr_dist:
            top_expr = max(expr_dist.values())
            n_with_expr = sum(expr_dist.values())
            if top_expr / max(n_with_expr, 1) > 0.6 and len(expr_dist) < 3:
                dominant = [e for e, c in expr_dist.items() if c == top_expr][0]
                next_to_generate.append(
                    f"Expression dominante : « {dominant} » ({top_expr}/{n_with_expr}) — varie : "
                    f"ajoute du sourire dent, du sérieux, du surpris…")
            elif len(expr_dist) < 3 and n_viable_total >= 15:
                missing = ", ".join(e for e in ["sourire", "rire", "sérieux", "surpris"] if e not in str(expr_dist))[:60]
                next_to_generate.append(f"Peu de variete d'expressions ({len(expr_dist)} types) — vise au moins 4 expressions differentes")

    # ===== SCORES PAR TARGET (suggestion automatique du meilleur target) =====
    target_scores = _compute_target_scores(results, viable_imgs, summary, n_viable)
    summary["target_scores"] = target_scores
    # Tri par score decroissant pour suggerer le meilleur
    if target_scores:
        best_target = max(target_scores.items(), key=lambda x: x[1]["score"])
        recommendations.append(
            f"🎯 Meilleur target pour ce dataset : « {best_target[0]} » "
            f"({best_target[1]['grade']}, {best_target[1]['score']}/100) — "
            f"{best_target[1]['reason']}"
        )

    # ===== DIVERSITÉ DU DATASET (clustering CLIP + tags) =====
    # Mesure : est-ce que le LoRA va apprendre une vraie persona ou juste
    # 30 variations de la même photo ?
    diversity = {
        "clip_clusters": 0,
        "clip_score": None,           # 0 (homogene) -> 100 (varie)
        "tag_distinct_top30": 0,
        "dominant_attributes": [],    # categories tag_overfit_alerts
        "overall_score": None,        # 0-100
        "verdict": "?",
    }

    # 1) Diversite via clustering des body_embeddings CLIP (qu'on a deja)
    valid_body = [(i, e) for i, e in enumerate(body_embeddings) if e is not None]
    if len(valid_body) >= 5:
        idxs = [i for i, _ in valid_body]
        body_mat = np.array([e for _, e in valid_body])
        sim_b = body_mat @ body_mat.T
        # Pour chaque image, count combien sont quasi-identiques (sim > 0.92)
        # -> cluster = un groupe d'images visuellement très similaires
        # Union-find simple
        N = len(valid_body)
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i in range(N):
            for j in range(i + 1, N):
                if sim_b[i, j] > 0.92:
                    union(i, j)
        clusters = {}
        for k in range(N):
            r = find(k)
            clusters.setdefault(r, []).append(idxs[k])
        n_clusters = len(clusters)
        diversity["clip_clusters"] = n_clusters
        # Ratio de clusters par rapport au total = score de diversite visuelle
        clip_score = min(100, int(n_clusters / N * 100))
        diversity["clip_score"] = clip_score
        # Detail des gros clusters (>=3 photos similaires = potentiellement
        # des photos qui se ressemblent trop)
        big_clusters = [
            {"size": len(v), "names": [results[i].get("name") for i in v[:5]]}
            for v in clusters.values() if len(v) >= 3
        ]
        diversity["big_clusters"] = big_clusters

    # 2) Diversite via tags WD14 (distinct dans top 30)
    if summary.get("tag_frequency_top"):
        diversity["tag_distinct_top30"] = len(summary["tag_frequency_top"])

    # 3) Attributs dominants (overfit detected)
    if summary.get("tag_overfit_alerts"):
        cats = list({a["category"] for a in summary["tag_overfit_alerts"]})
        diversity["dominant_attributes"] = cats

    # 4) Score global
    score_parts = []
    if diversity["clip_score"] is not None:
        score_parts.append(diversity["clip_score"])
    if summary.get("tag_overfit_alerts"):
        # Penalite : -15 par categorie d'overfit
        pen = max(0, 100 - 15 * len(set(a["category"] for a in summary["tag_overfit_alerts"])))
        score_parts.append(pen)
    if score_parts:
        overall = sum(score_parts) / len(score_parts)
        diversity["overall_score"] = round(overall, 0)
        if overall >= 75:
            diversity["verdict"] = "Excellente diversité"
        elif overall >= 55:
            diversity["verdict"] = "Diversité correcte"
        elif overall >= 35:
            diversity["verdict"] = "Diversité limitée"
        else:
            diversity["verdict"] = "Dataset trop homogène"

    summary["diversity"] = diversity

    # Reco diversite
    if diversity.get("overall_score") is not None:
        sc = diversity["overall_score"]
        if sc < 40:
            recommendations.append(
                f"❌ Diversité du dataset : {sc}/100 — {diversity['verdict']}. "
                f"Le LoRA risque de mémoriser plus que la persona "
                f"(attributs accessoires : {', '.join(diversity['dominant_attributes']) or '(voir tags)'}"
            )
        elif sc < 60:
            recommendations.append(
                f"⚠️ Diversité : {sc}/100 — {diversity['verdict']}. "
                f"Varie {', '.join(diversity['dominant_attributes']) or 'plus les angles/tenues/fonds'} pour un LoRA plus generaliste"
            )
        else:
            recommendations.append(f"✅ Diversité : {sc}/100 — {diversity['verdict']}")
    if diversity.get("big_clusters"):
        for c in diversity["big_clusters"][:3]:
            recommendations.append(
                f"🔁 Cluster de {c['size']} photos quasi-similaires : "
                f"{', '.join(c['names'][:3])}{'…' if len(c['names']) > 3 else ''}"
            )

    # ===== Distribution ASPECT-RATIO (alerte buckets cassés) =====
    # Important pour Flux multi-bucket et Wan vidéo : un dataset 39 tall / 1 wide
    # va casser le bucketing du trainer.
    ar_distribution = {"square": 0, "portrait": 0, "landscape": 0,
                        "tall_portrait": 0, "wide_landscape": 0}
    ar_alerts = []
    ar_target_recos = []
    ar_buckets = []
    for r in viable_imgs:
        w, h = r.get("width"), r.get("height")
        if not w or not h:
            continue
        ar = w / h
        ar_buckets.append((r.get("name"), round(ar, 2), w, h))
        if 0.9 <= ar <= 1.1:
            ar_distribution["square"] += 1
        elif 0.5 <= ar < 0.9:
            ar_distribution["portrait"] += 1
        elif ar < 0.5:
            ar_distribution["tall_portrait"] += 1   # ex 9:21 - probleme bucket
        elif 1.1 < ar <= 2.0:
            ar_distribution["landscape"] += 1
        else:
            ar_distribution["wide_landscape"] += 1  # ex 21:9 - probleme bucket

    n_with_ar = sum(ar_distribution.values())
    if n_with_ar >= 5:
        sq = ar_distribution["square"]
        po = ar_distribution["portrait"] + ar_distribution["tall_portrait"]
        la = ar_distribution["landscape"] + ar_distribution["wide_landscape"]
        sq_pct = sq / n_with_ar
        po_pct = po / n_with_ar
        la_pct = la / n_with_ar

        # Alertes deséquilibre
        if ar_distribution["tall_portrait"] >= 3:
            ar_alerts.append(
                f"⚠️ {ar_distribution['tall_portrait']} photo(s) très portrait (AR < 0.5) — "
                f"risque de cassure du bucketing Flux/Wan"
            )
        if ar_distribution["wide_landscape"] >= 3:
            ar_alerts.append(
                f"⚠️ {ar_distribution['wide_landscape']} photo(s) très paysage (AR > 2.0) — "
                f"risque de cassure du bucketing"
            )
        if sq_pct > 0.95:
            ar_target_recos.append("✅ Quasi-100% carré → idéal SDXL/SD1.5/Kohya (crop centre)")
            ar_target_recos.append("⚠️ Pour Flux/Wan multi-bucket : pense à ajouter des ratios variés")
        elif sq_pct > 0.7 and la_pct < 0.1 and po_pct < 0.2:
            ar_target_recos.append("✅ Dominante carré → parfait SDXL, OK pour Flux (mais multi-bucket sous-utilisé)")
        elif po_pct > 0.5:
            ar_target_recos.append("📱 Dominante portrait → Wan I2V/T2V portrait, Flux portrait OK")
        elif la_pct > 0.5:
            ar_target_recos.append("🖼 Dominante paysage → Wan I2V/T2V paysage, Flux paysage OK")
        elif sq_pct > 0.3 and po_pct > 0.2 and la_pct > 0.1:
            ar_target_recos.append("✅ Mix sain de ratios → idéal Flux/Wan multi-bucket")

    summary["aspect_ratio_distribution"] = ar_distribution
    summary["aspect_ratio_alerts"] = ar_alerts
    summary["aspect_ratio_target_recos"] = ar_target_recos

    # Reco panneau
    for a in ar_alerts:
        recommendations.append(a)
    for r in ar_target_recos:
        recommendations.append(f"📐 AR : {r}")

    # ===== Detection photos floues RECUPERABLES (upscale possible) =====
    # Sharpness entre 50 et 100 = floue mais sauvable avec SUPIR/UltraSharp
    blurry_recoverable = []
    for r in results:
        sh = r.get("sharpness")
        if sh is not None and 50 <= sh < 100 and r.get("face_count", 0) > 0:
            # Visage detecte = ca peut valoir le coup de tenter l'upscale
            r["upscale_candidate"] = True
            blurry_recoverable.append(r.get("name"))

    # Mise a jour summary
    summary["distribution"] = distribution
    summary["next_to_generate"] = next_to_generate
    summary["blurry_recoverable"] = blurry_recoverable
    summary["tag_frequency_top"] = top_tags if n_with_tags >= 5 else []
    summary["tag_overfit_alerts"] = tag_overfit_alerts

    # Reco overfit (tres visible dans le panneau)
    if tag_overfit_alerts:
        # Groupe par categorie pour message clair
        by_cat = {}
        for a in tag_overfit_alerts:
            by_cat.setdefault(a["category"], []).append(a)
        for cat, alerts in by_cat.items():
            # Garde les 3 plus frequents par categorie
            alerts.sort(key=lambda x: -x["count"])
            top = alerts[:3]
            tags_str = ", ".join(f"« {a['tag']} » ({int(a['ratio']*100)}%)" for a in top)
            icon = "❌" if any(a["severity"] == "ALERTE" for a in top) else "⚠️"
            recommendations.append(
                f"{icon} Overfit {cat} possible : {tags_str} — varie pour eviter "
                f"que le LoRA memorise ces traits comme partie de la persona"
            )

    # Reco upscale (visible dans le panneau)
    if blurry_recoverable:
        recommendations.append(
            f"🔧 {len(blurry_recoverable)} photo(s) floue(s) mais récupérable(s) (nettete 50-100) — "
            f"upscale possible avec SUPIR ou UltraSharp dans ComfyUI"
        )
    if next_to_generate:
        recommendations.append("🎯 À générer ensuite pour compléter le dataset :")
        for s in next_to_generate:
            recommendations.append(f"   → {s}")

    # ===== Mise a jour cache : on stocke l'entry FINAL (avec verdict, lora_viable, etc.) =====
    # Pour chaque image, retrouve le fkey correspondant et synchronise l'entry
    files_by_name = {p.name: p for p in images}
    for r in results:
        nm = r.get("name")
        p = files_by_name.get(nm)
        if not p:
            continue
        fk = _file_key(p)
        cached_entry = cache_new["entries"].get(fk)
        if cached_entry is None:
            continue
        cached_entry["entry"] = dict(r)

    save_cache(folder, cache_new)
    if n_from_cache > 0:
        print(f"STEP Cache mis a jour ({len(cache_new['entries'])} entrees)", file=sys.stderr, flush=True)

    return {
        "summary": summary,
        "recommendations": recommendations,
        "images": results,
        "cache_stats": {"reused": n_from_cache, "new": n_new},
    }


class NpEncoder(json.JSONEncoder):
    """Convertit les types numpy en types Python pour JSON."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: analyze_dataset.py <folder> [mode]"}))
        sys.exit(1)
    folder = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    cap = sys.argv[4] if len(sys.argv) > 4 else "wd14"  # wd14 / natural / both / joycaption / all
    dev = sys.argv[5] if len(sys.argv) > 5 else "auto"  # auto / cuda / cpu
    result = analyze(folder, mode, ref_image=ref, captioner_mode=cap, device=dev)
    print(json.dumps(result, ensure_ascii=False, indent=2, cls=NpEncoder))
