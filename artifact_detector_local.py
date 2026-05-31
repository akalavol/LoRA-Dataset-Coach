"""
Detecteur d'artefacts anatomiques IA (mains pourries, yeux asymetriques,
membres dupliques, anatomie deformee) - approche multi-source.

Strategie 2026 :
  1. Detection via WD14 tags (gratuit, on les a deja)
     -> tags : bad_hands, extra_fingers, bad_anatomy, mutated_hands,
        disfigured, asymmetric_eyes, bad_proportions, deformed
  2. Optionnel : HADM (Human Artifact Detection Models) - paper arXiv 2411.13842
     -> non installe par defaut (Detectron2 lourd), instructions dans README

Output par image : entry["artifacts"] = list de problemes detectes
"""
import re
from typing import Dict, List, Optional


# ============================================================
# Mapping des tags WD14 vers categories d'artefacts IA
# Seuils de confiance specifiques (WD14 retourne des probas)
# ============================================================

ARTIFACT_TAGS = {
    "mains": [
        "bad hands", "bad_hands", "extra fingers", "extra_fingers",
        "missing fingers", "missing_fingers", "mutated hands", "mutated_hands",
        "fused fingers", "deformed hands", "deformed_hands",
        "six fingers", "extra digit", "extra_digit",
    ],
    "yeux": [
        "bad eyes", "bad_eyes", "asymmetric eyes", "asymmetric_eyes",
        "lazy eye", "deformed eyes", "crossed eyes",
    ],
    "membres": [
        "extra limbs", "extra_limbs", "extra arms", "extra_arms",
        "extra legs", "missing limb", "missing_limb",
        "mutated limbs", "mutated_limbs", "fused limbs",
    ],
    "anatomie": [
        "bad anatomy", "bad_anatomy", "bad proportions", "bad_proportions",
        "deformed", "disfigured", "mutation", "mutated",
        "malformed", "weird body",
    ],
    "qualite": [
        "lowres", "low quality", "low_quality", "worst quality", "worst_quality",
        "jpeg artifacts", "blurry face",
    ],
}


def detect_artifacts_from_wd14(wd14_tags_string: Optional[str]) -> Dict:
    """Parse une string WD14 et retourne les artefacts trouves.

    wd14_tags est juste une chaine "tag1, tag2, tag3" sans scores
    -> on detecte la presence d'un mot-cle, sans seuil de confiance.
    Pour des seuils precis, il faudrait re-tagger l'image avec acces aux probas.
    """
    if not wd14_tags_string:
        return {"artifacts": [], "categories": [], "severity": "none"}

    tags_lower = wd14_tags_string.lower()
    found = []
    found_categories = set()

    for cat, patterns in ARTIFACT_TAGS.items():
        for pattern in patterns:
            if pattern in tags_lower:
                found.append({"category": cat, "tag": pattern})
                found_categories.add(cat)

    severity = "none"
    if "mains" in found_categories or "yeux" in found_categories:
        severity = "high"   # bug visuel direct sur le visage/mains
    elif "membres" in found_categories or "anatomie" in found_categories:
        severity = "medium"
    elif "qualite" in found_categories:
        severity = "low"

    return {
        "artifacts": found,
        "categories": sorted(found_categories),
        "severity": severity,
    }


def detect_artifacts_from_natural_caption(caption: Optional[str]) -> Dict:
    """Florence-2/JoyCaption mentionnent parfois directement les defauts.
    Cherche des expressions du genre "deformed hand", "extra finger",
    "asymmetric eyes" dans la caption naturelle.
    """
    if not caption:
        return {"artifacts": [], "categories": [], "severity": "none"}

    caption_lower = caption.lower()
    found = []
    found_categories = set()

    patterns_natural = {
        "mains": [
            r"deformed hand", r"extra finger", r"missing finger",
            r"six finger", r"fused finger", r"malformed hand",
        ],
        "yeux": [
            r"asymmetric eye", r"misaligned eye", r"deformed eye",
            r"strange eye", r"lazy eye",
        ],
        "membres": [
            r"extra arm", r"extra leg", r"missing arm", r"missing leg",
            r"three arms", r"three legs",
        ],
        "anatomie": [
            r"deformed body", r"disfigured", r"unnatural pose",
            r"distorted body", r"twisted limb",
        ],
    }

    for cat, regexes in patterns_natural.items():
        for rx in regexes:
            if re.search(rx, caption_lower):
                found.append({"category": cat, "pattern": rx})
                found_categories.add(cat)

    severity = "none"
    if "mains" in found_categories or "yeux" in found_categories:
        severity = "high"
    elif "membres" in found_categories or "anatomie" in found_categories:
        severity = "medium"

    return {
        "artifacts": found,
        "categories": sorted(found_categories),
        "severity": severity,
    }


def combined_artifact_detection(entry: Dict) -> Dict:
    """Combine les sources WD14 + caption naturelle pour un verdict final."""
    wd14_result = detect_artifacts_from_wd14(entry.get("wd14_tags"))
    natural_result = detect_artifacts_from_natural_caption(
        entry.get("joycaption") or entry.get("natural_caption")
    )

    # Severite max
    severities = ["none", "low", "medium", "high"]
    sev = max(
        severities.index(wd14_result["severity"]),
        severities.index(natural_result["severity"]),
    )
    final_severity = severities[sev]

    all_categories = sorted(set(wd14_result["categories"]) | set(natural_result["categories"]))

    return {
        "artifacts_from_wd14": wd14_result["artifacts"],
        "artifacts_from_caption": natural_result["artifacts"],
        "artifacts_categories": all_categories,
        "artifacts_severity": final_severity,
        "has_artifacts": bool(all_categories),
    }


if __name__ == "__main__":
    # Test rapide
    sample_entry = {
        "wd14_tags": "1girl, brown hair, smile, indoors, bad hands, extra fingers",
        "joycaption": "A woman with deformed hands and asymmetric eyes, sitting in a chair."
    }
    print(combined_artifact_detection(sample_entry))
