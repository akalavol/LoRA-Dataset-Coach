"""
Export du compte rendu d'analyse dataset LoRA en PDF paysage.
Utilise fpdf2 (leger, pur Python).
"""
import datetime
from pathlib import Path

from fpdf import FPDF


# Palette Catppuccin Mocha (couleurs hex en tuples RGB)
COLORS = {
    "bg":      (30, 30, 46),
    "card":    (49, 50, 68),
    "text":    (205, 214, 244),
    "dim":     (166, 173, 200),
    "accent":  (137, 180, 250),
    "accent2": (245, 194, 231),
    "green":   (166, 227, 161),
    "yellow":  (249, 226, 175),
    "red":     (243, 139, 168),
    "orange":  (250, 179, 135),
}


class AnalysisPDF(FPDF):
    def header(self):
        # Bandeau haut
        self.set_fill_color(*COLORS["card"])
        self.rect(0, 0, self.w, 18, "F")
        self.set_text_color(*COLORS["accent2"])
        self.set_font("Helvetica", "B", 14)
        self.set_y(5)
        self.set_x(10)
        self.cell(0, 8, "Rapport d'analyse - Dataset LoRA", new_x="LMARGIN", new_y="NEXT")
        # Reset
        self.set_text_color(*COLORS["text"])

    def footer(self):
        self.set_y(-12)
        self.set_text_color(*COLORS["dim"])
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Page {self.page_no()} / {{nb}}  -  Genere le {datetime.datetime.now():%Y-%m-%d %H:%M}",
                 align="C")


def _color_for_tag(tag):
    return {"ok": COLORS["green"], "warn": COLORS["yellow"], "err": COLORS["red"]}.get(tag, COLORS["text"])


def _classify(img):
    """Tag couleur base sur la viabilite LoRA (priorite max)."""
    viable = img.get("lora_viable", "yes")
    if viable == "no":
        return "err"
    if viable == "borderline":
        return "warn"
    return "ok"


def export(data, folder_path, out_path):
    """
    data : dict retourne par analyze_dataset.py
    folder_path : chemin du dataset analyse
    out_path : chemin du PDF a generer
    """
    pdf = AnalysisPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    pdf.set_fill_color(*COLORS["bg"])
    pdf.set_text_color(*COLORS["text"])
    pdf.add_page()
    # Fond noir sur toute la page (Catppuccin)
    pdf.rect(0, 18, pdf.w, pdf.h - 18, "F")

    # ===== 1. Dataset info =====
    pdf.ln(4)
    pdf.set_text_color(*COLORS["dim"])
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(10)
    pdf.cell(0, 5, f"Dossier analyse : {folder_path}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(10)
    pdf.cell(0, 5, f"Date : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===== 1.5 VERDICT GLOBAL (encadre en haut) =====
    summary = data.get("summary", {})
    verdict = summary.get("verdict", {})
    if verdict:
        grade = verdict.get("grade", "?")
        desc = verdict.get("grade_desc", "")
        viable_now = verdict.get("viable_now", 0)
        after = verdict.get("after_cleanup", 0)
        target_min = verdict.get("target_min", 20)
        target_ideal = verdict.get("target_ideal", 30)
        actions = verdict.get("actions", [])

        # Couleur selon grade
        gc = COLORS["green"] if grade in ("A", "B", "B+") else (
             COLORS["yellow"] if grade in ("B-", "C") else COLORS["red"])

        # Cartouche verdict
        pdf.set_fill_color(*COLORS["card"])
        x_start = 10
        cart_y = pdf.get_y()
        pdf.rect(x_start, cart_y, pdf.w - 20, 26, "F")

        # Grade enorme a gauche
        pdf.set_text_color(*gc)
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_xy(x_start + 4, cart_y + 2)
        pdf.cell(28, 22, grade, align="C")

        # Texte a droite
        pdf.set_xy(x_start + 36, cart_y + 3)
        pdf.set_text_color(*gc)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 7, f"VERDICT DATASET : {desc}")

        pdf.set_xy(x_start + 36, cart_y + 11)
        pdf.set_text_color(*COLORS["text"])
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, f"{viable_now} viables maintenant -> {after} apres cleanup  (cible {target_min}-{target_ideal})")

        # Bloc reference (si utilisee)
        pdf.set_y(cart_y + 28)
        ref_info = summary.get("reference") or {}
        ref_match = summary.get("reference_match") or {}
        if ref_info and "error" not in ref_info and ref_match:
            ok = ref_match.get("ok", 0)
            doubt = ref_match.get("doubt", 0)
            wrong = ref_match.get("wrong", 0)
            avg = ref_match.get("avg", 0)
            if wrong == 0 and doubt == 0:
                rc = COLORS["green"]; line = f"OK - {ok} photos correspondent (avg {avg})"
            elif wrong == 0:
                rc = COLORS["yellow"]; line = f"{ok} OK, {doubt} douteuses (avg {avg})"
            else:
                rc = COLORS["red"]; line = f"{wrong} mauvaise(s) personne(s), {doubt} douteuse(s), {ok} OK (avg {avg})"
            pdf.set_text_color(*rc)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(10)
            pdf.cell(0, 6, f"Photo de reference : {ref_info.get('name','?')} -> {line}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif ref_info and "error" in ref_info:
            pdf.set_text_color(*COLORS["yellow"])
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(10)
            pdf.cell(0, 6, f"Reference : {ref_info['error']}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        if actions:
            pdf.set_text_color(*COLORS["accent"])
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(10)
            pdf.cell(0, 6, "Plan d'action :", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*COLORS["text"])
            pdf.set_font("Helvetica", "", 9.5)
            for a in actions:
                pdf.set_x(14)
                pdf.cell(0, 5, f"- {a}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # ===== 2. Summary =====
    pdf.set_text_color(*COLORS["accent"])
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(10)
    pdf.cell(0, 7, "Statistiques globales", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*COLORS["text"])

    stats = [
        ("Total images", summary.get("total_images", 0)),
        ("LoRA viables", summary.get("lora_viable", 0)),
        ("Borderline", summary.get("lora_borderline", 0)),
        ("A virer", summary.get("lora_unusable", 0)),
        ("Coherence visage", summary.get("overall_face_coherence", "N/A")),
        ("Coherence corps", summary.get("overall_body_coherence", "N/A")),
        ("Resolution min", summary.get("resolution_min", "N/A")),
        ("Resolution max", summary.get("resolution_max", "N/A")),
    ]
    # Tableau 4 colonnes
    col_w = (pdf.w - 20) / 4
    for i, (label, val) in enumerate(stats):
        if i % 4 == 0 and i > 0:
            pdf.ln()
        pdf.set_x(10 + (i % 4) * col_w)
        pdf.set_text_color(*COLORS["dim"])
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_w, 4, label, new_x="RIGHT", new_y="TOP")
        pdf.set_x(10 + (i % 4) * col_w)
        pdf.set_y(pdf.get_y() + 4)
        pdf.set_text_color(*COLORS["accent2"])
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(col_w, 5, str(val), new_x="RIGHT", new_y="TOP")
        pdf.set_y(pdf.get_y() - 4)
    pdf.ln(8)

    # ===== 3. Recommandations =====
    recos = data.get("recommendations", [])
    if recos:
        pdf.ln(3)
        pdf.set_text_color(*COLORS["accent"])
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(10)
        pdf.cell(0, 7, "Recommandations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9.5)
        import re
        for r in recos:
            # Filtre emojis non supportes par fonte par defaut, garde texte
            r_clean = (r.replace("✅", "[OK]").replace("⚠️", "[!]").replace("⚠", "[!]")
                        .replace("❌", "[X]").replace("💡", "[i]").replace("📊", "[#]")
                        .replace("🧬", "[LoRA]").replace("🎯", "[*]").replace("📷", "[img]"))
            # Supprime tout autre emoji / caractere non-supporte par helvetica (>U+FFFF)
            r_clean = re.sub(r"[\U00010000-\U0010ffff]", "", r_clean)
            # Couleur selon prefixe
            if "[X]" in r_clean:
                pdf.set_text_color(*COLORS["red"])
            elif "[!]" in r_clean:
                pdf.set_text_color(*COLORS["yellow"])
            elif "[OK]" in r_clean:
                pdf.set_text_color(*COLORS["green"])
            else:
                pdf.set_text_color(*COLORS["text"])
            pdf.set_x(12)
            pdf.multi_cell(pdf.w - 24, 5, r_clean)

    # ===== 4. Tableau images =====
    pdf.ln(4)
    pdf.set_text_color(*COLORS["accent"])
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(10)
    pdf.cell(0, 7, "Detail par image", new_x="LMARGIN", new_y="NEXT")

    headers = ["Image", "Resol.", "Net.", "Vis.", "%vis", "Pose", "Expression", "S.vis", "S.cor", "Qualite", "LoRA - raison"]
    col_widths = [50, 20, 13, 9, 12, 12, 25, 12, 12, 30, 82]  # ~277mm

    # Header
    pdf.set_fill_color(*COLORS["card"])
    pdf.set_text_color(*COLORS["accent"])
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(10)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 7, h, border=0, fill=True)
    pdf.ln()

    # Lignes
    pdf.set_font("Helvetica", "", 8.5)
    for img in data.get("images", []):
        tag = _classify(img)
        row_color = _color_for_tag(tag)

        # Si la page est presque pleine, saute
        if pdf.get_y() > pdf.h - 25:
            pdf.add_page()
            pdf.rect(0, 18, pdf.w, pdf.h - 18, "F")
            pdf.set_fill_color(*COLORS["card"])
            pdf.set_text_color(*COLORS["accent"])
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(10)
            for h, w in zip(headers, col_widths):
                pdf.cell(w, 7, h, border=0, fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 8.5)

        name = img.get("name", "?")[:42]
        res = f"{img.get('width', '?')}x{img.get('height', '?')}"
        sharp = f"{img.get('sharpness', 0):.0f}" if img.get("sharpness") is not None else "-"
        faces = str(img.get("face_count", "?"))
        prop = f"{img.get('face_proportion', 0):.1f}%" if img.get("face_proportion") else "-"
        yaw_val = img.get("face_yaw")
        yaw = f"{yaw_val:+.0f}" if yaw_val is not None else "-"
        expr = (img.get("expression") or "-")[:14]
        sim_avg = img.get("face_similarity_avg")
        sim = f"{sim_avg:.2f}" if sim_avg is not None else "-"
        body_avg = img.get("body_similarity_avg")
        body = f"{body_avg:.2f}" if body_avg is not None else "NA"
        qual = (img.get("quality_verdict", "-"))[:22]
        viable = img.get("lora_viable", "yes")
        prefix = {"yes": "[OK] ", "borderline": "[!] ", "no": "[X] "}.get(viable, "")
        lora = (prefix + img.get("lora_reason", "-"))[:55]

        # Caractere ASCII pour visibilite : remplace les emojis non supportes
        qual = qual.replace("✅", "OK").replace("⚠️", "!")

        values = [name, res, sharp, faces, prop, yaw, expr, sim, body, qual, lora]
        pdf.set_text_color(*row_color)
        pdf.set_x(10)
        for v, w in zip(values, col_widths):
            pdf.cell(w, 5.5, str(v), border=0)
        pdf.ln()

    # Output
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return str(out_path)


if __name__ == "__main__":
    # Test : prend un JSON depuis stdin et genere un PDF
    import json
    import sys
    if len(sys.argv) < 3:
        print("Usage : export_pdf.py <json_file> <output_pdf>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    out = export(data, "TEST", sys.argv[2])
    print(out)
