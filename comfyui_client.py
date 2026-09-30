"""
Pilotage d'un ComfyUI LOCAL via son API HTTP pour generer le jeu de test
de l'evaluateur LoRA (bibliotheque standard uniquement, pas de dependance).

Principe :
  1. L'utilisateur exporte SON workflow depuis ComfyUI en "Save (API Format)".
     On ne fournit pas de graphe par modele : FLUX.2, Qwen-Image, Wan 2.2...
     ont chacun leur graphe, celui de l'utilisateur est le seul qui marche
     sur sa machine.
  2. On patche ce workflow pour chaque image : prompt, seed, LoRA + force,
     prefixe de sortie.
  3. On genere N images AVEC le LoRA et (option) les memes seeds SANS le LoRA
     (force 0) -> baseline A/B : le gain d'identite du LoRA devient mesurable.

Routes utilisees (ComfyUI server.py) :
  GET  /system_stats        ping
  GET  /models/loras        liste des LoRA installes
  POST /prompt              met un workflow en file -> prompt_id
  GET  /history/{id}        etat + fichiers produits
  GET  /view                telecharge une image produite
"""
import copy
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8188"

# Prompts de test : cadrages / lumieres / expressions / contextes varies.
# Un LoRA qui generalise garde l'identite sur TOUS ; un LoRA qui a
# memorise ne reussit que ceux proches du dataset.
DEFAULT_TEST_PROMPTS = [
    "close-up portrait photo of {trigger}, neutral expression, soft window light",
    "photo of {trigger} laughing, outdoors in a park, golden hour",
    "full body photo of {trigger} walking in a city street at night, neon lights",
    "profile view of {trigger}, looking to the side, studio lighting, grey background",
    "photo of {trigger} sitting in a cafe, reading a book, candid shot",
    "{trigger} wearing a winter coat and scarf, snowy mountain background",
    "three-quarter view portrait of {trigger}, serious expression, dramatic rim light",
    "photo of {trigger} at the beach, wind in the hair, overcast sky",
    "{trigger} in a business suit, office background, smiling, corporate headshot",
    "low angle photo of {trigger} on a rooftop at sunset, wide shot",
]

SAVE_NODE_HINTS = ("SaveImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM", "SaveVideo")
SEED_KEYS = ("seed", "noise_seed")


class ComfyUIError(RuntimeError):
    pass


# ============================================================
# HTTP
# ============================================================

def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _get_json(url, timeout=10):
    return json.loads(_get(url, timeout).decode("utf-8"))


def ping(base_url=DEFAULT_URL):
    """True si ComfyUI repond."""
    try:
        _get_json(f"{base_url}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def list_loras(base_url=DEFAULT_URL):
    """Noms des LoRA tels que ComfyUI les attend dans lora_name."""
    return _get_json(f"{base_url}/models/loras")


def queue_prompt(base_url, workflow, client_id):
    body = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # ComfyUI renvoie 400 + node_errors si le graphe est invalide
        detail = e.read().decode("utf-8", "replace")[:800]
        raise ComfyUIError(f"ComfyUI a refuse le workflow (HTTP {e.code}) : {detail}")
    if data.get("node_errors"):
        raise ComfyUIError(f"Erreurs de nodes : {json.dumps(data['node_errors'])[:800]}")
    return data["prompt_id"]


def wait_for(base_url, prompt_id, timeout=1800, poll=1.0):
    """Attend la fin d'un prompt et renvoie son entree d'historique."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = _get_json(f"{base_url}/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise ComfyUIError(f"Execution ComfyUI en erreur : {json.dumps(msgs)[:800]}")
            if status.get("completed", True):
                return entry
        time.sleep(poll)
    raise ComfyUIError(f"Timeout : prompt {prompt_id} non termine apres {timeout}s")


def download_outputs(base_url, entry, dest_folder, stem):
    """Telecharge les images produites. Renvoie la liste des chemins ecrits."""
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    written = []
    for node_out in entry.get("outputs", {}).values():
        for i, img in enumerate(node_out.get("images", [])):
            if img.get("type") == "temp":      # previews : pas des sorties finales
                continue
            q = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            ext = Path(img["filename"]).suffix or ".png"
            out = dest_folder / f"{stem}{'' if i == 0 else f'_{i}'}{ext}"
            out.write_bytes(_get(f"{base_url}/view?{q}", timeout=60))
            written.append(out)
    return written


# ============================================================
# PATCH DU WORKFLOW (format API)
# ============================================================

def load_api_workflow(path):
    wf = json.loads(Path(path).read_text(encoding="utf-8"))
    if "nodes" in wf and "links" in wf:
        raise ComfyUIError(
            "Ce fichier est un workflow format UI. Dans ComfyUI : menu Workflow > "
            "Export (API) / 'Save (API Format)', puis donne ce fichier-la.")
    if not isinstance(wf, dict) or not all(isinstance(v, dict) and "class_type" in v
                                           for v in wf.values()):
        raise ComfyUIError("JSON non reconnu comme workflow ComfyUI format API.")
    return wf


def _lora_nodes(wf):
    return [nid for nid, n in wf.items()
            if "lora" in n["class_type"].lower() and "lora_name" in n.get("inputs", {})]


def _find_positive_text_node(wf):
    """Node texte positif : placeholder {prompt} en priorite, sinon on remonte
    l'entree 'positive' d'un sampler (via FluxGuidance & co) jusqu'a un node 'text'."""
    for nid, n in wf.items():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and "{prompt}" in v:
                return nid, k
    for n in wf.values():
        link = n.get("inputs", {}).get("positive")
        depth = 0
        while isinstance(link, list) and depth < 6:
            node = wf.get(str(link[0]))
            if node is None:
                break
            inputs = node.get("inputs", {})
            for key in ("text", "prompt", "text_g"):
                if isinstance(inputs.get(key), str):
                    return str(link[0]), key
            link = inputs.get("conditioning") or inputs.get("positive")
            depth += 1
    return None, None


def inspect_workflow(wf):
    """Resume ce qu'on sait patcher - affiche a l'utilisateur AVANT de lancer."""
    text_node, text_key = _find_positive_text_node(wf)
    # Seed "en dur" (int) seulement : une seed reliee a un autre node (liste) n'est pas touchee
    seeds = [nid for nid, n in wf.items()
             if any(isinstance(n.get("inputs", {}).get(k), int) for k in SEED_KEYS)]
    saves = [nid for nid, n in wf.items() if n["class_type"] in SAVE_NODE_HINTS]
    problems = []
    if not _lora_nodes(wf):
        problems.append("aucun node LoRA (LoraLoader / LoraLoaderModelOnly) dans le workflow")
    if not text_node:
        problems.append("prompt positif introuvable : mets {prompt} dans ton texte positif")
    if not saves:
        problems.append("aucun node SaveImage : les images ne seraient pas recuperables")
    if not seeds:
        problems.append("aucune seed modifiable : toutes les images seraient identiques")
    return {"lora_nodes": _lora_nodes(wf), "text_node": text_node, "text_key": text_key,
            "seed_nodes": seeds, "save_nodes": saves, "problems": problems}


def patch_workflow(wf, prompt, seed, lora_name, strength, prefix):
    wf = copy.deepcopy(wf)
    info = inspect_workflow(wf)
    if info["problems"]:
        raise ComfyUIError("Workflow inutilisable : " + " ; ".join(info["problems"]))
    node = wf[info["text_node"]]["inputs"]
    cur = node[info["text_key"]]
    node[info["text_key"]] = cur.replace("{prompt}", prompt) if "{prompt}" in cur else prompt
    for nid in info["seed_nodes"]:
        for k in SEED_KEYS:
            if isinstance(wf[nid]["inputs"].get(k), int):
                wf[nid]["inputs"][k] = seed
    # Le PREMIER node LoRA recoit le LoRA teste ; les autres (LoRA de style,
    # lightning...) restent tels quels.
    first = wf[info["lora_nodes"][0]]["inputs"]
    first["lora_name"] = lora_name
    for k in ("strength_model", "strength", "lora_strength"):
        if k in first:
            first[k] = strength
    if "strength_clip" in first:
        first["strength_clip"] = strength
    for nid in info["save_nodes"]:
        wf[nid]["inputs"]["filename_prefix"] = prefix
    return wf


# ============================================================
# GENERATION DU JEU DE TEST
# ============================================================

def natural_key(name):
    """Tri 'humain' : lin_2 avant lin_10 (les checkpoints d'epoch se trient bien)."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def filter_loras(loras, pattern):
    """LoRA dont le nom contient pattern (insensible a la casse), tri naturel."""
    # Separateurs normalises : ComfyUI sous Windows renvoie "lora_eval\\run\\x.safetensors"
    pat = (pattern or "").strip().lower().replace("\\", "/")
    return sorted((l for l in loras if pat and pat in l.lower().replace("\\", "/")),
                  key=natural_key)


def safe_dir_name(lora_name):
    return Path(lora_name.replace("\\", "/")).stem.replace(" ", "_")


def _run_jobs(wf, jobs, base_url, progress_cb, stop_flag):
    """jobs : (kind, key, i, prompt, seed, lora_name, strength, dest)."""
    client_id = uuid.uuid4().hex
    res = {"files": {}, "errors": [], "prompts": {}}
    fails_in_a_row = 0
    for k, (kind, key, i, p, seed, lora, s, dest) in enumerate(jobs, 1):
        if stop_flag and stop_flag():
            res["errors"].append("arrete par l'utilisateur")
            break
        stem = f"{kind}_{i:03d}_s{seed}"
        if progress_cb:
            progress_cb(k, len(jobs), f"{key} · {stem}")
        try:
            pid = queue_prompt(base_url, patch_workflow(wf, p, seed, lora, s,
                                                        f"lora_eval/{safe_dir_name(key)}_{stem}"),
                               client_id)
            files = download_outputs(base_url, wait_for(base_url, pid), dest, stem)
            if not files:
                res["errors"].append(f"{key}/{stem}: aucune image produite")
            res["files"].setdefault(key, []).extend(str(f) for f in files)
            res["prompts"][stem] = p
            fails_in_a_row = 0
        except ComfyUIError as e:
            res["errors"].append(f"{key}/{stem}: {e}")
            fails_in_a_row += 1
            # Workflow casse = inutile de relancer 200 fois la meme erreur
            if fails_in_a_row >= 3:
                res["errors"].append("3 echecs d'affilee : arret")
                break
    return res


def _prepare(workflow_path, lora_names, base_url):
    if not ping(base_url):
        raise ComfyUIError(f"ComfyUI ne repond pas sur {base_url}. Lance-le d'abord.")
    wf = load_api_workflow(workflow_path)
    installed = list_loras(base_url)
    missing = [l for l in lora_names if l not in installed]
    if missing:
        raise ComfyUIError(f"LoRA absent(s) de ComfyUI/models/loras : {', '.join(missing)} "
                           f"({len(installed)} LoRA installes).")
    return wf


def generate_batch(workflow_path, lora_names, trigger, out_folder,
                   n_images=20, strength=1.0, with_baseline=True,
                   prompts=None, base_url=DEFAULT_URL, seed0=1000,
                   progress_cb=None, stop_flag=None):
    """
    Teste un LOT de LoRA (ex: les checkpoints lin_000001..lin_000004) dans les
    memes conditions : memes prompts, memes seeds. Structure produite :
        out_folder/<lora>/lora_000_s1000.png ...
        out_folder/_baseline/baseline_000_s1000.png ...   (une seule fois, partagee)
    Renvoie {"folders": {lora_name: dossier}, "files": {...}, "errors": [...]}.
    """
    lora_names = list(lora_names)
    if not lora_names:
        raise ComfyUIError("Aucun LoRA a tester.")
    wf = _prepare(workflow_path, lora_names, base_url)
    prompts = prompts or DEFAULT_TEST_PROMPTS
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    folders, jobs = {}, []
    for i in range(n_images):
        p = prompts[i % len(prompts)].replace("{trigger}", trigger)
        if with_baseline:
            # Force 0 = modele de base seul ; le LoRA charge importe peu
            jobs.append(("baseline", "_baseline", i, p, seed0 + i, lora_names[0], 0.0,
                         out_folder / "_baseline"))
    for lora in lora_names:
        dest = out_folder if len(lora_names) == 1 else out_folder / safe_dir_name(lora)
        folders[lora] = str(dest)
        for i in range(n_images):
            p = prompts[i % len(prompts)].replace("{trigger}", trigger)
            jobs.append(("lora", lora, i, p, seed0 + i, lora, strength, dest))

    res = _run_jobs(wf, jobs, base_url, progress_cb, stop_flag)
    res["folders"] = folders
    res["baseline_folder"] = str(out_folder / "_baseline") if with_baseline else None
    (out_folder / "eval_prompts.json").write_text(
        json.dumps({"loras": lora_names, "strength": strength, "trigger": trigger,
                    "n_images": n_images, "seed0": seed0, "prompts": res["prompts"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def generate_eval_set(workflow_path, lora_name, trigger, out_folder,
                      n_images=20, strength=1.0, with_baseline=True,
                      prompts=None, base_url=DEFAULT_URL, seed0=1000,
                      progress_cb=None, stop_flag=None):
    """Un seul LoRA : images dans out_folder/, baseline dans out_folder/_baseline/."""
    r = generate_batch(workflow_path, [lora_name], trigger, out_folder, n_images, strength,
                       with_baseline, prompts, base_url, seed0, progress_cb, stop_flag)
    return {"lora": r["files"].get(lora_name, []),
            "baseline": r["files"].get("_baseline", []),
            "errors": r["errors"], "prompts": r["prompts"]}


# ============================================================
# CLASSEMENT D'UN LOT
# ============================================================

def rank_batch(results):
    """
    results : {lora_name: resultat lora_evaluator (dict)} + cle optionnelle
    "_baseline". Renvoie une liste triee du meilleur au pire.
    Critere : score du verdict (inclut deja copycat / mode collapse), puis gain
    d'identite vs baseline. Un LoRA qui copie le dataset ne peut pas gagner.
    """
    base = results.get("_baseline", {}).get("summary", {}).get("r_facesim_mean")
    rows = []
    for name, r in results.items():
        if name == "_baseline":
            continue
        if "error" in r:
            rows.append({"lora": name, "error": r["error"], "score": -1})
            continue
        s = r.get("summary", {})
        v = s.get("verdict", {})
        m = s.get("r_facesim_mean")
        rows.append({
            "lora": name,
            "grade": v.get("grade", "?"),
            "score": v.get("score", 0),
            "r_facesim_mean": m,
            "r_facesim_std": s.get("r_facesim_std"),
            "copycat": s.get("copycat_count", 0),
            "gain": round(m - base, 4) if (m is not None and base is not None) else None,
        })
    rows.sort(key=lambda r: (r["score"], r.get("gain") or -9), reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows
