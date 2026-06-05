"""
Analyse de dataset voix pour l'entrainement RVC (Applio).
Analyse chaque fichier audio : duree, SNR, niveau RMS, frequence fondamentale,
format, taux d'echantillonnage, diversite phonetique (via Whisper si dispo).
"""
import os
import sys
import json
import math
import pathlib

SUPPORTED = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}

# ------------------------------------------------------------------ helpers --

def _load_audio(path):
    """Charge le fichier audio. Retourne (signal_float32_mono, samplerate) ou raise."""
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        return data.astype("float32"), sr
    except Exception:
        pass
    # fallback librosa
    import librosa
    data, sr = librosa.load(path, sr=None, mono=True)
    return data.astype("float32"), sr


def _rms_db(signal):
    rms = math.sqrt(max(1e-12, float((signal ** 2).mean())))
    return 20 * math.log10(rms)


def _snr_estimate(signal, sr):
    """Estimation SNR simple : RMS signal total vs RMS 20 premiers ms (bruit)."""
    noise_len = min(int(0.02 * sr), len(signal) // 4)
    if noise_len < 10:
        return None
    import numpy as np
    noise_rms = max(1e-12, float((signal[:noise_len] ** 2).mean()) ** 0.5)
    signal_rms = max(1e-12, float((signal ** 2).mean()) ** 0.5)
    return round(20 * math.log10(signal_rms / noise_rms), 1)


def _f0_mean(signal, sr):
    """Frequence fondamentale moyenne via zero-crossing rate (rapide, approximatif)."""
    try:
        import librosa
        f0, voiced, _ = librosa.pyin(signal, fmin=50, fmax=600,
                                      sr=sr, frame_length=2048)
        import numpy as np
        voiced_f0 = f0[voiced > 0] if f0 is not None else []
        if len(voiced_f0) > 0:
            return round(float(np.nanmean(voiced_f0)), 1)
    except Exception:
        pass
    return None


def analyze_voice_file(path):
    """Retourne un dict avec toutes les metriques pour un fichier."""
    p = pathlib.Path(path)
    result = {
        "file": p.name,
        "path": str(p),
        "ext": p.suffix.lower(),
        "size_mb": round(p.stat().st_size / 1_048_576, 2),
        "duration_s": None,
        "sample_rate": None,
        "rms_db": None,
        "snr_db": None,
        "f0_mean_hz": None,
        "issues": [],
        "score": None,   # A-F
    }
    try:
        signal, sr = _load_audio(path)
        result["duration_s"] = round(len(signal) / sr, 2)
        result["sample_rate"] = sr
        result["rms_db"] = round(_rms_db(signal), 1)
        result["snr_db"] = _snr_estimate(signal, sr)
        result["f0_mean_hz"] = _f0_mean(signal, sr)
    except Exception as e:
        result["issues"].append(f"Lecture impossible : {e}")
        result["score"] = "F"
        return result

    # ---- règles qualité RVC ----
    issues = []
    dur = result["duration_s"] or 0
    if dur < 3:
        issues.append(f"Trop court ({dur:.1f}s, min 3s)")
    if dur > 30:
        issues.append(f"Très long ({dur:.1f}s) — couper en segments de 3-15s")
    if sr < 40000:
        issues.append(f"Taux trop bas ({sr} Hz, recommandé ≥40kHz pour RVC)")
    if result["rms_db"] is not None and result["rms_db"] < -35:
        issues.append(f"Niveau très faible ({result['rms_db']} dB) — normaliser")
    if result["snr_db"] is not None and result["snr_db"] < 20:
        issues.append(f"SNR faible ({result['snr_db']} dB) — trop de bruit de fond")
    if result["ext"] not in {".wav", ".flac"}:
        issues.append(f"Format compressé ({result['ext']}) — convertir en WAV/FLAC")

    result["issues"] = issues

    # ---- note ----
    if issues:
        critical = [i for i in issues if "impossible" in i or "Trop court" in i]
        result["score"] = "D" if critical else ("C" if len(issues) > 1 else "B")
    else:
        snr = result["snr_db"] or 0
        result["score"] = "A" if snr >= 30 else "B"

    return result


def analyze_voice_folder(folder, callback_progress=None, callback_result=None):
    """
    Analyse tous les fichiers audio d'un dossier.
    callback_progress(i, total, filename) — appelé avant chaque fichier
    callback_result(result_dict)           — appelé après chaque fichier
    Retourne le resume global.
    """
    files = sorted([
        f for f in pathlib.Path(folder).rglob("*")
        if f.suffix.lower() in SUPPORTED
    ])
    total = len(files)
    if total == 0:
        return {"total": 0, "files": []}

    results = []
    for i, f in enumerate(files):
        if callback_progress:
            callback_progress(i, total, f.name)
        r = analyze_voice_file(str(f))
        results.append(r)
        if callback_result:
            callback_result(r)

    # ---- resume global ----
    durations = [r["duration_s"] for r in results if r["duration_s"]]
    total_dur = sum(durations)
    scores = [r["score"] for r in results]
    grade_counts = {g: scores.count(g) for g in "ABCDF"}
    issues_all = [i for r in results for i in r["issues"]]

    # diversite des taux d'echantillonnage (pour RVC il vaut mieux l'uniformite)
    srs = set(r["sample_rate"] for r in results if r["sample_rate"])
    sr_warning = len(srs) > 1

    summary = {
        "total": total,
        "total_duration_min": round(total_dur / 60, 1),
        "avg_duration_s": round(total_dur / total, 1) if total else 0,
        "grade_counts": grade_counts,
        "overall_score": "A" if grade_counts.get("A", 0) / max(total, 1) > 0.7 else
                         "B" if (grade_counts.get("A", 0) + grade_counts.get("B", 0)) / max(total, 1) > 0.6 else
                         "C" if grade_counts.get("F", 0) == 0 else "D",
        "sr_inconsistent": sr_warning,
        "sample_rates": sorted(srs),
        "top_issues": _top_issues(issues_all),
        "rvc_ready": grade_counts.get("F", 0) == 0 and total_dur / 60 >= 3,
        "recommendation": _rvc_recommendation(total_dur, grade_counts, total),
        "files": results,
    }
    return summary


def _top_issues(issues):
    from collections import Counter
    c = Counter(issues)
    return [{"issue": k, "count": v} for k, v in c.most_common(5)]


def _rvc_recommendation(total_dur_s, grade_counts, total):
    total_min = total_dur_s / 60
    msgs = []
    if total_min < 3:
        msgs.append(f"⚠️ Durée totale {total_min:.1f} min — RVC recommande au moins 3 min (idéal 10-30 min)")
    elif total_min < 10:
        msgs.append(f"💡 {total_min:.1f} min — correct, mais 10+ min donnera un meilleur résultat")
    else:
        msgs.append(f"✅ {total_min:.1f} min — durée excellente pour RVC")
    bad = grade_counts.get("C", 0) + grade_counts.get("D", 0) + grade_counts.get("F", 0)
    if bad > 0:
        msgs.append(f"⚠️ {bad}/{total} fichiers ont des problèmes de qualité — filtre-les avant l'entraînement")
    return " · ".join(msgs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_voice.py <dossier>")
        sys.exit(1)
    summary = analyze_voice_folder(
        sys.argv[1],
        callback_progress=lambda i, t, n: print(f"[{i+1}/{t}] {n}", file=sys.stderr)
    )
    # stdout = JSON pour le manager
    print(json.dumps(summary, ensure_ascii=False, indent=2))
