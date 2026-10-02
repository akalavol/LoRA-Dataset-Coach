"""
Createur de LoRA : lance l'entrainement d'un dossier prepare par lora_prep.py,
suit la progression, liste les checkpoints et les envoie a ComfyUI pour le
test en lot (comfyui_client.generate_batch).

Aucune dependance Tk ici : la GUI (manager.py) ne fait qu'afficher.

Runners reconnus dans un dossier prepare :
  - launch_*.bat          -> musubi-tuner (targets musubi_generic + legacy)
                             les lignes "set CLE=valeur" sont editables depuis la GUI
  - ai_toolkit*.yaml      -> ai-toolkit : <python ai-toolkit> run.py <yaml>
Les autres formats (Kohya GUI, diffusion-pipe, OneTrainer) restent manuels :
voir le README.txt du dossier.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

SET_RE = re.compile(r"^set\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.IGNORECASE)
# tqdm : " 12%|███       | 300/2500 [..]"  (musubi, ai-toolkit)
TQDM_RE = re.compile(r"(\d{1,3})%\|[^|]*\|\s*(\d+)/(\d+)")
EPOCH_RE = re.compile(r"epoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(r"(?:avr_loss|loss)[=:\s]+([0-9]*\.[0-9]+(?:e-?\d+)?)", re.IGNORECASE)

CHECKPOINT_EXTS = (".safetensors",)


# ============================================================
# DETECTION
# ============================================================

def detect_runner(folder):
    """Renvoie {"kind": "bat"|"aitoolkit"|None, "file": Path|None, "readme": str|None}."""
    folder = Path(folder)
    if not folder.is_dir():
        return {"kind": None, "file": None, "error": f"Dossier introuvable : {folder}"}
    bats = sorted(folder.glob("launch_*.bat"))
    yamls = sorted(folder.glob("ai_toolkit*.yaml"))
    readme = folder / "README.txt"
    info = {"readme": readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else None}
    if bats:
        return {"kind": "bat", "file": bats[0], **info}
    if yamls:
        return {"kind": "aitoolkit", "file": yamls[0], **info}
    return {"kind": None, "file": None,
            "error": "Pas de launch_*.bat ni de ai_toolkit*.yaml : ce target se lance "
                     "depuis son propre trainer (voir README.txt).", **info}


def read_bat_vars(bat_path):
    """Lignes 'set CLE=valeur' du .bat -> dict ordonne (chemins modeles, MUSUBI_DIR...)."""
    out = {}
    for line in Path(bat_path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = SET_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def write_bat_vars(bat_path, new_vars):
    """Reecrit uniquement les lignes 'set CLE=...' modifiees (le reste est intact)."""
    p = Path(bat_path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = SET_RE.match(line.strip())
        if m and m.group(1) in new_vars:
            lines[i] = f"set {m.group(1)}={new_vars[m.group(1)]}"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_paths(bat_vars):
    """Chemins declares dans le .bat qui n'existent pas (avant de lancer 3 h de cache)."""
    missing = []
    for k, v in bat_vars.items():
        if k == "MODEL_VERSION":
            continue
        if ("\\" in v or "/" in v) and not Path(v).exists():
            missing.append(f"{k} = {v}")
    return missing


def output_dir(folder):
    return Path(folder) / "output"


# ============================================================
# LANCEMENT
# ============================================================

def build_command(runner, aitoolkit_dir=None, aitoolkit_python=None):
    """Commande + cwd pour subprocess."""
    f = Path(runner["file"])
    if runner["kind"] == "bat":
        if os.name != "nt":
            raise RuntimeError("Les .bat musubi ne se lancent que sous Windows.")
        return ["cmd", "/c", str(f)], str(f.parent)
    if runner["kind"] == "aitoolkit":
        d = Path(aitoolkit_dir or "")
        if not (d / "run.py").is_file():
            raise RuntimeError(f"ai-toolkit introuvable : {d / 'run.py'} (onglet Config).")
        py = aitoolkit_python or sys.executable
        return [py, "run.py", str(f)], str(d)
    raise RuntimeError(runner.get("error") or "Runner inconnu.")


def start(cmd, cwd):
    """Lance le process : stdout+stderr fusionnes, stdin ferme (les 'pause' du .bat
    rendent la main), groupe de process pour pouvoir tout arreter."""
    kw = dict(cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
              stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
              bufsize=1)
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    return subprocess.Popen(cmd, env=env, **kw)


def stop(proc):
    """Arrete l'arbre entier (accelerate lance des sous-process python)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.kill()


def parse_progress(line):
    """Extrait {pct, step, total, epoch, epochs, loss} d'une ligne de log (ou {})."""
    out = {}
    m = TQDM_RE.search(line)
    if m:
        out.update(pct=int(m.group(1)), step=int(m.group(2)), total=int(m.group(3)))
    m = EPOCH_RE.search(line)
    if m:
        out.update(epoch=int(m.group(1)), epochs=int(m.group(2)))
    m = LOSS_RE.search(line)
    if m:
        try:
            out["loss"] = float(m.group(1))
        except ValueError:
            pass
    return out


# ============================================================
# CHECKPOINTS -> COMFYUI
# ============================================================

def list_checkpoints(folder):
    """Checkpoints produits (output/**.safetensors), tri naturel par nom."""
    from comfyui_client import natural_key
    out = output_dir(folder)
    if not out.is_dir():
        return []
    files = [p for p in out.rglob("*") if p.suffix.lower() in CHECKPOINT_EXTS
             and "optimizer" not in p.name.lower()]
    return sorted(files, key=lambda p: natural_key(p.name))


def publish_to_comfyui(checkpoints, comfy_loras_dir, run_name):
    """Copie les checkpoints dans ComfyUI/models/loras/lora_eval/<run_name>/.
    Renvoie le filtre a utiliser dans le test en lot."""
    dest = Path(comfy_loras_dir) / "lora_eval" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for c in checkpoints:
        target = dest / Path(c).name
        if not target.exists() or target.stat().st_mtime < Path(c).stat().st_mtime:
            shutil.copy2(c, target)
        copied.append(target)
    return {"dest": str(dest), "copied": [str(p) for p in copied],
            "filter": f"lora_eval/{run_name}/"}
