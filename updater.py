"""
Auto-updater for LoRA-Dataset-Coach via GitHub releases.

Checks the latest release on github.com/akalavol/LoRA-Dataset-Coach,
compares with the local VERSION file or git tag, proposes update.

Two modes:
  1. Git mode: if the install is a git clone, runs `git pull`
  2. Zip mode: download the source tarball from GitHub and overwrite

Usage:
    from updater import check_for_updates, apply_update
    info = check_for_updates()
    # info = {"current": "v1.0.0", "latest": "v1.1.0", "update_available": True,
    #          "url": "...", "notes": "..."}
"""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

GITHUB_REPO = "akalavol/LoRA-Dataset-Coach"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}"
VERSION_FILE = Path(__file__).parent / "VERSION"


def get_current_version() -> str:
    """Reads VERSION file. Returns 'v0.0.0' if missing."""
    if VERSION_FILE.is_file():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    # Fallback : try git
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "v0.0.0"


def _normalize_version(v: str) -> tuple:
    """Parses 'v1.2.3' or '1.2.3' into a comparable tuple."""
    v = v.lstrip("v").strip()
    parts = re.findall(r"\d+", v)
    return tuple(int(p) for p in parts) if parts else (0,)


def _fetch_latest_release() -> Optional[dict]:
    """Fetches the latest release from GitHub API."""
    url = f"{GITHUB_API_BASE}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LoRA-Dataset-Coach-Updater",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Update check failed: {e}", file=sys.stderr)
        return None


def check_for_updates() -> dict:
    """
    Returns:
        {
          "current": "v1.0.0",
          "latest": "v1.1.0" or None,
          "update_available": bool,
          "url": str,
          "notes": str (release body),
          "published_at": str,
          "is_git_install": bool,
          "error": str (if any)
        }
    """
    current = get_current_version()
    is_git = (Path(__file__).parent / ".git").is_dir()

    rel = _fetch_latest_release()
    if rel is None:
        return {
            "current": current, "latest": None,
            "update_available": False,
            "error": "Cannot reach GitHub (no internet or rate limited)",
            "is_git_install": is_git,
        }

    latest_tag = rel.get("tag_name") or "v0.0.0"
    update_available = _normalize_version(latest_tag) > _normalize_version(current)

    return {
        "current": current,
        "latest": latest_tag,
        "update_available": update_available,
        "url": rel.get("html_url", ""),
        "notes": (rel.get("body") or "")[:2000],
        "published_at": rel.get("published_at", ""),
        "is_git_install": is_git,
        "zipball_url": rel.get("zipball_url", ""),
    }


def apply_update_git() -> dict:
    """git pull on the install. Returns {success, message}."""
    root = Path(__file__).parent
    if not (root / ".git").is_dir():
        return {"success": False, "message": "Not a git install"}
    try:
        out = subprocess.check_output(
            ["git", "pull", "--ff-only"],
            cwd=str(root), stderr=subprocess.STDOUT, text=True,
        )
        return {"success": True, "message": out.strip()}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": e.output}


def apply_update_zip(zipball_url: str, backup: bool = True) -> dict:
    """Download the release zipball, replace files in the install. Returns {success, message}."""
    root = Path(__file__).parent
    tmp_dir = root / "_update_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    # Extraction dans un sous-dossier dédié, et le zip téléchargé À PART,
    # pour ne pas confondre les deux.
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / "release.zip"

    try:
        req = urllib.request.Request(
            zipball_url, headers={"User-Agent": "LoRA-Dataset-Coach-Updater"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            zip_path.write_bytes(resp.read())
    except Exception as e:
        return {"success": False, "message": f"Téléchargement échoué : {e}"}

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        return {"success": False, "message": f"Décompression échouée : {e}"}

    # GitHub enveloppe tout dans un dossier akalavol-LoRA-Dataset-Coach-<sha>/
    # On prend le SEUL sous-dossier de extract_dir (en ignorant tout fichier).
    sub_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    extracted_root = sub_dirs[0] if sub_dirs else extract_dir

    # Fichiers qu'on NE doit jamais écraser (config locale de l'utilisateur)
    PROTECTED = {"config.json", "config.local.json"}

    # Backup des .py actuels
    backup_dir = root / "_backup_before_update"
    if backup:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        backup_dir.mkdir()
        for py in root.glob("*.py"):
            try:
                shutil.copy2(py, backup_dir / py.name)
            except Exception:
                pass

    # Copie les nouveaux fichiers
    replaced = []
    for src in extracted_root.rglob("*"):
        if not src.is_file():
            continue
        rel_path = src.relative_to(extracted_root)
        # L'exclusion porte sur le chemin RELATIF (pas sur src qui contient _update_tmp !)
        if any(part in (".git", "__pycache__") for part in rel_path.parts):
            continue
        if rel_path.name in PROTECTED:
            continue
        dst = root / rel_path
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            replaced.append(str(rel_path))
        except Exception as e:
            # Un .py verrouillé/illisible ne doit pas tout casser
            print(f"Skip {rel_path}: {e}", file=sys.stderr)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not replaced:
        return {"success": False,
                "message": "Aucun fichier copié (archive vide ou structure inattendue)."}

    new_ver = get_current_version()  # relit le VERSION fraîchement écrit
    return {
        "success": True,
        "message": f"{len(replaced)} fichier(s) mis à jour → version {new_ver}.\n"
                   f"Backup dans {backup_dir.name}/",
        "replaced": replaced[:30],
    }


if __name__ == "__main__":
    info = check_for_updates()
    print(json.dumps(info, indent=2))
    if info.get("update_available"):
        print("\nRun apply_update_git() or apply_update_zip(info['zipball_url']) to update.")
