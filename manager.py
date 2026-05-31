"""
AI File Manager - Gestion visuelle des LoRAs et fichiers I/O ComfyUI.
Theme dark Catppuccin, sans dependances.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import Tk, Frame, Label, Button, StringVar, Text, filedialog, messagebox, ttk

# PIL pour le preview live des photos analysees (optionnel)
try:
    from PIL import Image as PILImage, ImageTk as PILImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# Path vers le python embeded de ComfyUI-future (qui a insightface)
COMFYUI_FUTURE_PY = r"C:\AI\ComfyUI-future\ComfyUI_windows_portable\python_embeded\python.exe"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Modeles proposes dans le dropdown (du plus rapide au plus precis)
PROMPT_MODELS = [
    ("gemma3:4b",     "Rapide (~3s) - Qualite correcte"),
    ("gemma4:latest", "Lent (~30s premier appel, ~10s ensuite) - Meilleure qualite"),
]

# System prompts par type de generation

# ============================================================
# Theme
# ============================================================
BG       = "#1e1e2e"   # base
BG2      = "#181825"   # mantle (dark)
CARD     = "#313244"   # surface0
CARD_HI  = "#45475a"   # surface1
TEXT     = "#cdd6f4"   # text
TEXT_DIM = "#a6adc8"   # subtext0
ACCENT   = "#89b4fa"   # blue
ACCENT2  = "#f5c2e7"   # pink
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_H1     = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_EMOJI  = ("Segoe UI Emoji", 18)

# ============================================================
# Config
# ============================================================
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    # Path to the ComfyUI python_embeded.exe that has insightface + torch CUDA
    "comfyui_python": COMFYUI_FUTURE_PY,
    # Default folder pre-filled in the analyzer
    "datasets_dir": "C:\\AI\\datasets",
}



def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)




def open_in_explorer(path):
    p = Path(path)
    if not p.exists():
        messagebox.showerror("Erreur", f"Dossier introuvable :\n{path}")
        return
    if sys.platform == "win32":
        os.startfile(str(p))


# ============================================================
# Widgets stylises
# ============================================================
def make_button(parent, text, command, primary=False, danger=False, width=None):
    if primary:
        bg, fg, ahover = ACCENT, BG, "#74c7ec"
    elif danger:
        bg, fg, ahover = RED, BG, "#eba0ac"
    else:
        bg, fg, ahover = CARD_HI, TEXT, "#585b70"
    b = Button(parent, text=text, command=command, font=FONT_BODY,
               bg=bg, fg=fg, activebackground=ahover, activeforeground=fg,
               relief="flat", padx=14, pady=7, cursor="hand2",
               borderwidth=0)
    if width:
        b.configure(width=width)
    return b


# ============================================================
# App
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()

        root.title("LoRA Dataset Coach")
        root.geometry("1180x780")
        root.configure(bg=BG)
        root.minsize(900, 600)

        # Chemin du Python ComfyUI (configurable, sinon valeur par defaut)
        self.comfyui_py = self.cfg.get("comfyui_python") or self.comfyui_py

        # Style ttk pour treeview dark
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.Treeview",
                        background=CARD, foreground=TEXT, fieldbackground=CARD,
                        rowheight=28, borderwidth=0, font=FONT_BODY)
        style.configure("Dark.Treeview.Heading",
                        background=BG2, foreground=ACCENT, font=FONT_H1,
                        relief="flat", borderwidth=0)
        style.map("Dark.Treeview", background=[("selected", ACCENT)],
                 foreground=[("selected", BG)])
        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=BG2, foreground=TEXT_DIM,
                        padding=[20, 8], font=FONT_BODY, borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                 background=[("selected", CARD)],
                 foreground=[("selected", TEXT)])

        # Header
        header = Frame(root, bg=BG, pady=14, padx=20)
        header.pack(fill="x")
        Label(header, text="🧬  LoRA Dataset Coach",
              font=FONT_TITLE, fg=TEXT, bg=BG).pack(side="left")
        Label(header, text="Analyse · Prépare · Évalue tes datasets LoRA",
              font=FONT_BODY, fg=TEXT_DIM, bg=BG).pack(side="left", padx=15, pady=8)

        # Status bar (cree AVANT les tabs car ils l'utilisent pendant init)
        self.status_var = StringVar(value="Pret")
        status_bar = Frame(root, bg=BG2, padx=15, pady=6)
        status_bar.pack(fill="x", side="bottom")
        Label(status_bar, textvariable=self.status_var,
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG2, anchor="w").pack(fill="x")

        # Notebook
        self.notebook = ttk.Notebook(root, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)
        notebook = self.notebook

        self._build_analyzer_tab(notebook)
        self._build_evaluator_tab(notebook)
        self._build_config_tab(notebook)

    def select_tab(self, name):
        """Selectionne un tab par nom court."""
        mapping = {"analyzer": 0, "evaluator": 1, "config": 2}
        if name in mapping:
            self.notebook.select(mapping[name])

    # ------------------ (removed: LoRAs / Outputs / Inputs / Prompt tabs) ------------------
    # ============================================================
    # ÉVALUATEUR LoRA POST-TRAIN (Lot E partie 1)
    # MirrorMetrics-inspired : R-FaceSim, Copycat, Black Hole
    # ============================================================
    def _build_evaluator_tab(self, parent):
        frame = Frame(parent, bg=BG, padx=15, pady=12)
        parent.add(frame, text="  📊 Évaluer LoRA  ")

        Label(frame, text="Évaluateur LoRA post-entraînement",
              font=FONT_H1, fg=ACCENT2, bg=BG).pack(anchor="w")
        Label(frame,
              text="Une fois ton LoRA entraîné, génère ~30 images de test, "
                   "puis fournis-les ici avec des photos réelles du sujet "
                   "(différentes de celles d'entraînement). On calcule R-FaceSim, "
                   "Copycat, Black Hole — le standard 2026 (MirrorMetrics).",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG, wraplength=900,
              justify="left").pack(anchor="w", pady=(0, 10))

        # === Champs ===
        def pick_folder_for(var, msg=None):
            d = filedialog.askdirectory(initialdir=var.get())
            if d:
                var.set(d.replace("/", "\\"))

        # Generated folder
        gen_frame = Frame(frame, bg=CARD, padx=10, pady=8)
        gen_frame.pack(fill="x", pady=4)
        Label(gen_frame, text="🎲 Images générées par ton LoRA :",
              font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        self.eval_gen_path = StringVar(value=r"C:\AI\datasets\lora_test_output")
        tk.Entry(gen_frame, textvariable=self.eval_gen_path, font=FONT_MONO,
                  bg=BG2, fg=TEXT, insertbackground=TEXT, relief="flat"
                  ).pack(side="left", fill="x", expand=True, ipady=5, padx=8)
        make_button(gen_frame, "📂",
                     lambda: pick_folder_for(self.eval_gen_path)).pack(side="left")

        # Reference folder (REAL photos of subject)
        ref_frame = Frame(frame, bg=CARD, padx=10, pady=8)
        ref_frame.pack(fill="x", pady=4)
        Label(ref_frame, text="📸 Photos RÉELLES du sujet :",
              font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        self.eval_ref_path = StringVar(value="")
        tk.Entry(ref_frame, textvariable=self.eval_ref_path, font=FONT_MONO,
                  bg=BG2, fg=TEXT, insertbackground=TEXT, relief="flat"
                  ).pack(side="left", fill="x", expand=True, ipady=5, padx=8)
        make_button(ref_frame, "📂",
                     lambda: pick_folder_for(self.eval_ref_path)).pack(side="left")
        Label(frame,
              text="   ⚠️ Doivent être différentes des photos d'entraînement (sinon score gonflé)",
              font=FONT_SMALL, fg=YELLOW, bg=BG).pack(anchor="w", padx=10, pady=(0, 4))

        # Training folder (optional)
        train_frame = Frame(frame, bg=CARD, padx=10, pady=8)
        train_frame.pack(fill="x", pady=4)
        Label(train_frame, text="🧬 Dataset d'entraînement (optionnel, copycat check) :",
              font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        self.eval_train_path = StringVar(value="")
        tk.Entry(train_frame, textvariable=self.eval_train_path, font=FONT_MONO,
                  bg=BG2, fg=TEXT, insertbackground=TEXT, relief="flat"
                  ).pack(side="left", fill="x", expand=True, ipady=5, padx=8)
        make_button(train_frame, "📂",
                     lambda: pick_folder_for(self.eval_train_path)).pack(side="left")

        # Bouton lancer
        Button(frame, text="🚀 Évaluer le LoRA", font=FONT_H1, bg=ACCENT2, fg=BG,
               relief="flat", padx=20, pady=10, cursor="hand2",
               command=self._run_evaluator).pack(pady=12)

        # Progress
        self.eval_phase = Label(frame, text="⏸ En attente",
                                 font=FONT_BODY, fg=TEXT_DIM, bg=BG)
        self.eval_phase.pack(anchor="w", pady=4)
        self.eval_progress = ttk.Progressbar(frame, mode="determinate",
                                              length=600, maximum=100)
        self.eval_progress.pack(fill="x", pady=2)

        # Verdict block
        self.eval_verdict_frame = Frame(frame, bg=CARD, padx=15, pady=10)
        # not packed yet

        # Détail tableau
        eval_tree_frame = Frame(frame, bg=BG)
        eval_tree_frame.pack(fill="both", expand=True, pady=8)
        cols = ("face", "rsim", "rsim_max", "near_ref", "near_train", "copycat")
        self.eval_tree = ttk.Treeview(eval_tree_frame, columns=cols, show="tree headings",
                                       height=14)
        self.eval_tree.heading("#0", text="Image")
        self.eval_tree.heading("face", text="Visage")
        self.eval_tree.heading("rsim", text="R-FaceSim")
        self.eval_tree.heading("rsim_max", text="Max")
        self.eval_tree.heading("near_ref", text="Réf proche")
        self.eval_tree.heading("near_train", text="Train proche")
        self.eval_tree.heading("copycat", text="Copycat ?")
        self.eval_tree.column("#0", width=180)
        self.eval_tree.column("face", width=70, anchor="center")
        self.eval_tree.column("rsim", width=80, anchor="center")
        self.eval_tree.column("rsim_max", width=70, anchor="center")
        self.eval_tree.column("near_ref", width=200, anchor="w")
        self.eval_tree.column("near_train", width=200, anchor="w")
        self.eval_tree.column("copycat", width=90, anchor="center")
        sb = ttk.Scrollbar(eval_tree_frame, orient="vertical",
                            command=self.eval_tree.yview)
        self.eval_tree.configure(yscrollcommand=sb.set)
        self.eval_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.eval_tree.tag_configure("ok", foreground=GREEN)
        self.eval_tree.tag_configure("warn", foreground=YELLOW)
        self.eval_tree.tag_configure("err", foreground=RED)

    def _run_evaluator(self):
        gen = self.eval_gen_path.get()
        ref = self.eval_ref_path.get()
        train = self.eval_train_path.get().strip()

        if not Path(gen).is_dir():
            messagebox.showerror("Erreur", f"Dossier images générées introuvable :\n{gen}")
            return
        if not Path(ref).is_dir():
            messagebox.showerror("Erreur",
                f"Dossier photos référence introuvable :\n{ref}\n\n"
                "Tu dois fournir des photos RÉELLES du sujet (≠ entraînement)")
            return

        self.eval_phase.config(text="⚙ Chargement insightface...", fg=ACCENT)
        self.eval_progress.config(value=0, mode="indeterminate")
        self.eval_progress.start(15)
        self.eval_tree.delete(*self.eval_tree.get_children())
        for w in self.eval_verdict_frame.winfo_children():
            w.destroy()
        self.eval_verdict_frame.pack_forget()

        threading.Thread(target=self._eval_subprocess,
                          args=(gen, ref, train), daemon=True).start()

    def _eval_subprocess(self, gen, ref, train):
        import time as _time
        script = str(Path(__file__).parent / "lora_evaluator.py")
        try:
            cmd = [self.comfyui_py, script, gen, ref]
            if train:
                cmd.append(train)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1,
            )

            def read_progress():
                for raw in iter(proc.stderr.readline, ""):
                    line = raw.strip()
                    if line.startswith("STEP "):
                        msg = line[5:]
                        self.root.after(0, lambda m=msg:
                            self.eval_phase.config(text=f"⚙ {m}", fg=ACCENT))
                    elif line.startswith("PROGRESS "):
                        # PROGRESS REF 5/20 imagename
                        try:
                            parts = line.split(" ", 3)
                            kind = parts[1]
                            cur, tot = parts[2].split("/")
                            n = parts[3] if len(parts) > 3 else ""
                            self.root.after(0, lambda k=kind, c=cur, t=tot, n=n:
                                self.eval_phase.config(
                                    text=f"⚙ Embedding {k} {c}/{t} — {n[:30]}", fg=ACCENT))
                        except Exception:
                            pass

            t = threading.Thread(target=read_progress, daemon=True)
            t.start()
            stdout, _ = proc.communicate(timeout=600)
            result = json.loads(stdout.strip()) if stdout.strip() else {"error": "no output"}
            self.root.after(0, lambda r=result: self._show_evaluator_result(r))
        except Exception as e:
            import traceback
            err = f"{e}\n\n{traceback.format_exc()[-500:]}"
            self.root.after(0, lambda e=err: messagebox.showerror("Erreur évaluation", e))

    def _show_evaluator_result(self, data):
        self.eval_progress.stop()
        self.eval_progress.config(mode="determinate", value=100)
        if "error" in data:
            self.eval_phase.config(text=f"❌ {data['error']}", fg=RED)
            return

        s = data.get("summary", {})
        v = s.get("verdict", {})
        self.eval_phase.config(text=f"✅ Évaluation terminée — {s.get('n_generated', 0)} générées vs {s.get('n_reference', 0)} réfs",
                                 fg=GREEN)

        # Bloc verdict
        grade = v.get("grade", "?")
        score = v.get("score", 0)
        desc = v.get("desc", "")
        advice = v.get("advice", "")
        issues = v.get("issues", [])
        color = {"A": GREEN, "B+": GREEN, "B": GREEN, "C": YELLOW,
                 "D": RED, "F": RED}.get(grade, TEXT)

        top = Frame(self.eval_verdict_frame, bg=CARD)
        top.pack(fill="x")
        Label(top, text=grade, font=("Segoe UI", 32, "bold"),
              fg=color, bg=CARD).pack(side="left", padx=(0, 15))
        right = Frame(top, bg=CARD)
        right.pack(side="left", fill="x", expand=True)
        Label(right, text=f"VERDICT LoRA : {desc}  ({score}/100)",
              font=FONT_H1, fg=color, bg=CARD, anchor="w").pack(fill="x")
        Label(right, text=f"R-FaceSim : moyenne {s.get('r_facesim_mean')} (std {s.get('r_facesim_std')}, min {s.get('r_facesim_min')}, max {s.get('r_facesim_max')})",
              font=FONT_BODY, fg=TEXT_DIM, bg=CARD, anchor="w").pack(fill="x")
        if s.get("copycat_count", 0) > 0:
            Label(right, text=f"❌ {s['copycat_count']} copycat detecté(s) (LoRA recopie au lieu de généraliser)",
                  font=FONT_BODY, fg=RED, bg=CARD, anchor="w").pack(fill="x")

        Label(self.eval_verdict_frame, text=f"💡 {advice}",
              font=FONT_BODY, fg=ACCENT, bg=CARD,
              anchor="w", wraplength=900, justify="left").pack(fill="x", pady=(6, 2))
        for issue in issues:
            Label(self.eval_verdict_frame, text=f"  ⚠️ {issue}",
                  font=FONT_SMALL, fg=YELLOW, bg=CARD,
                  anchor="w", wraplength=900, justify="left").pack(fill="x")

        # Black hole ranking
        bh = s.get("black_hole_top", [])
        if bh:
            Label(self.eval_verdict_frame, text="🕳 Black Hole Ranking (training images les plus copiées) :",
                  font=FONT_H1, fg=ACCENT, bg=CARD, anchor="w").pack(fill="x", pady=(10, 4))
            for entry in bh[:5]:
                Label(self.eval_verdict_frame,
                      text=f"  • {entry['training_image']} attire {entry['attributed_count']} gens "
                           f"({entry['ratio']*100:.0f}%)",
                      font=FONT_SMALL, fg=TEXT_DIM if entry['ratio'] < 0.3 else YELLOW,
                      bg=CARD, anchor="w").pack(fill="x")

        self.eval_verdict_frame.pack(fill="x", padx=10, pady=(0, 6),
                                       before=self.eval_progress.master if hasattr(self.eval_progress, 'master') else None)
        # Si le pack avant ne marche pas, pack normalement
        if not self.eval_verdict_frame.winfo_ismapped():
            self.eval_verdict_frame.pack(fill="x", padx=10, pady=(0, 6))

        # Tableau detail
        for entry in data.get("per_image", []):
            name = entry.get("name", "?")
            face = "✅" if entry.get("has_face") else "❌"
            rsim = entry.get("r_facesim")
            rsim_str = f"{rsim:.3f}" if rsim is not None else "-"
            rsim_max = entry.get("r_facesim_max")
            rsim_max_str = f"{rsim_max:.3f}" if rsim_max is not None else "-"
            nref = entry.get("nearest_ref") or {}
            ntrain = entry.get("nearest_train") or {}
            near_ref_str = f"{nref.get('name', '-')} ({nref.get('sim', '-')})" if nref else "-"
            near_tr_str = f"{ntrain.get('name', '-')} ({ntrain.get('sim', '-')})" if ntrain else "-"
            copycat = "❌ OUI" if entry.get("copycat") else "✓"

            if not entry.get("has_face"):
                tag = "err"
            elif entry.get("copycat"):
                tag = "err"
            elif rsim is not None and rsim < 0.4:
                tag = "warn"
            else:
                tag = "ok"
            self.eval_tree.insert("", "end", text=f"  {name}",
                                    values=(face, rsim_str, rsim_max_str,
                                              near_ref_str, near_tr_str, copycat),
                                    tags=(tag,))

    def _build_config_tab(self, parent):
        frame = Frame(parent, bg=BG, padx=20, pady=20)
        parent.add(frame, text="  ⚙ Config  ")

        Label(frame, text="Configuration",
              font=FONT_H1, fg=TEXT, bg=BG).pack(anchor="w", pady=(0, 5))
        Label(frame, text="Adapte selon ton install. Sauvegardé dans config.json.",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(anchor="w", pady=(0, 20))

        self.entries = {}
        # key : (icon, label, desc, picker_type)  picker_type = "file" | "dir"
        labels = {
            "comfyui_python": ("🐍", "Python ComfyUI",
                                "python.exe de ComfyUI (a insightface + torch CUDA)", "file"),
            "datasets_dir":   ("📂", "Dossier datasets par défaut",
                                "Pré-rempli dans l'analyseur", "dir"),
        }
        for key, (icon, label, desc, picker) in labels.items():
            card = Frame(frame, bg=CARD, padx=14, pady=12)
            card.pack(fill="x", pady=4)

            top_row = Frame(card, bg=CARD)
            top_row.pack(fill="x")
            Label(top_row, text=icon, font=FONT_EMOJI, fg=ACCENT2, bg=CARD).pack(side="left", padx=(0, 10))
            Label(top_row, text=label, font=FONT_H1, fg=TEXT, bg=CARD).pack(side="left")
            Label(top_row, text=desc, font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(side="left", padx=10)

            entry_row = Frame(card, bg=CARD)
            entry_row.pack(fill="x", pady=(8, 0))
            var = StringVar(value=self.cfg.get(key, ""))
            entry = tk.Entry(entry_row, textvariable=var, font=FONT_MONO,
                             bg=BG2, fg=TEXT, insertbackground=TEXT,
                             relief="flat", borderwidth=0)
            entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
            if picker == "file":
                make_button(entry_row, "📂", lambda v=var: self._pick_file(v)).pack(side="left")
            else:
                make_button(entry_row, "📂", lambda v=var: self._pick_folder(v)).pack(side="left")
            self.entries[key] = var

        btn_row = Frame(frame, bg=BG)
        btn_row.pack(pady=25)
        make_button(btn_row, "💾 Enregistrer", self._save_cfg, primary=True).pack(side="left", padx=5)
        make_button(btn_row, "↺ Reset defauts", self._reset_cfg).pack(side="left", padx=5)

        # ===== Section MAJ logiciel =====
        Label(frame, text="Mise à jour du logiciel",
              font=FONT_H1, fg=TEXT, bg=BG).pack(anchor="w", pady=(30, 5))
        upd_card = Frame(frame, bg=CARD, padx=14, pady=12)
        upd_card.pack(fill="x", pady=4)

        # Lecture version locale
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from updater import get_current_version
            current_v = get_current_version()
        except Exception:
            current_v = "v0.0.0"

        ver_row = Frame(upd_card, bg=CARD)
        ver_row.pack(fill="x")
        Label(ver_row, text=f"📦 Version locale : {current_v}",
              font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        Label(ver_row, text="  ·  github.com/akalavol/LoRA-Dataset-Coach",
              font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(side="left")

        self.upd_status = Label(upd_card, text="", font=FONT_SMALL,
                                 fg=TEXT_DIM, bg=CARD, anchor="w",
                                 wraplength=900, justify="left")
        self.upd_status.pack(fill="x", pady=(8, 0))

        upd_btn_row = Frame(upd_card, bg=CARD)
        upd_btn_row.pack(fill="x", pady=(8, 0))
        make_button(upd_btn_row, "🔄 Vérifier maintenant",
                     self._check_for_updates, primary=True).pack(side="left", padx=5)
        self.upd_apply_btn = Button(upd_btn_row, text="⬇ Installer la mise à jour",
                                      font=FONT_BODY, bg=CARD_HI, fg=TEXT_DIM,
                                      relief="flat", padx=12, pady=6,
                                      state="disabled", cursor="hand2",
                                      command=self._apply_update)
        self.upd_apply_btn.pack(side="left", padx=5)

        # ===== Section CHANGELOG (Journal des nouveautés) =====
        Label(frame, text="Journal des nouveautés",
              font=FONT_H1, fg=TEXT, bg=BG).pack(anchor="w", pady=(25, 5))
        Label(frame, text="Toutes les modifications, ajouts et corrections de chaque version.",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(anchor="w", pady=(0, 8))

        chg_card = Frame(frame, bg=CARD, padx=14, pady=12)
        chg_card.pack(fill="both", expand=True, pady=4)

        chg_header = Frame(chg_card, bg=CARD)
        chg_header.pack(fill="x", pady=(0, 6))
        Label(chg_header, text="📋 CHANGELOG.md", font=FONT_BODY,
              fg=ACCENT2, bg=CARD).pack(side="left")
        make_button(chg_header, "🔄 Recharger",
                     self._load_changelog).pack(side="right", padx=4)
        make_button(chg_header, "🌐 Voir sur GitHub",
                     lambda: self._open_url(
                         "https://github.com/akalavol/LoRA-Dataset-Coach/blob/main/CHANGELOG.md"
                     )).pack(side="right", padx=4)

        chg_scroll_frame = Frame(chg_card, bg=CARD)
        chg_scroll_frame.pack(fill="both", expand=True)
        self.changelog_text = Text(
            chg_scroll_frame, bg=BG2, fg=TEXT, font=FONT_MONO,
            relief="flat", padx=8, pady=8, wrap="word", height=18,
            insertbackground=TEXT,
        )
        chg_sb = ttk.Scrollbar(chg_scroll_frame, orient="vertical",
                                 command=self.changelog_text.yview)
        self.changelog_text.configure(yscrollcommand=chg_sb.set)
        self.changelog_text.pack(side="left", fill="both", expand=True)
        chg_sb.pack(side="right", fill="y")

        # Tags coloration markdown basique
        self.changelog_text.tag_configure("h1", foreground=ACCENT,
                                            font=("Segoe UI", 14, "bold"),
                                            spacing1=12, spacing3=4)
        self.changelog_text.tag_configure("h2", foreground=ACCENT2,
                                            font=("Segoe UI", 12, "bold"),
                                            spacing1=10, spacing3=4)
        self.changelog_text.tag_configure("h3", foreground=GREEN,
                                            font=("Segoe UI", 11, "bold"),
                                            spacing1=6, spacing3=2)
        self.changelog_text.tag_configure("bullet", foreground=TEXT_DIM)

        # Charge le contenu initial
        self._load_changelog()

    def _check_for_updates(self):
        self.upd_status.config(text="🔍 Vérification en cours…", fg=TEXT_DIM)
        self.upd_apply_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM)
        def worker():
            try:
                from updater import check_for_updates
                info = check_for_updates()
                self.root.after(0, lambda: self._on_update_check_done(info))
            except Exception as e:
                self.root.after(0, lambda e=e: self.upd_status.config(
                    text=f"❌ Erreur : {e}", fg=RED))
        threading.Thread(target=worker, daemon=True).start()

    def _on_update_check_done(self, info):
        if info.get("error"):
            self.upd_status.config(text=f"⚠️ {info['error']}", fg=YELLOW)
            return
        current = info.get("current", "?")
        latest = info.get("latest", "?")
        if info.get("update_available"):
            notes = (info.get("notes") or "")[:300]
            self.upd_status.config(
                text=f"✅ Nouvelle version disponible : {latest} (tu as {current})\n\n"
                     f"Notes :\n{notes}",
                fg=GREEN)
            self.upd_apply_btn.config(state="normal", bg=GREEN, fg=BG)
            self._pending_update_info = info
        else:
            self.upd_status.config(
                text=f"✅ Tu es à jour ({current}).", fg=GREEN)

    def _apply_update(self):
        info = getattr(self, "_pending_update_info", None)
        if not info:
            return
        is_git = info.get("is_git_install")
        msg = (f"Installer {info['latest']} ?\n\n"
               f"Méthode : {'git pull (recommandé)' if is_git else 'Téléchargement zip + remplacement'}\n"
               f"Un backup des .py actuels sera créé dans _backup_before_update/\n\n"
               f"Le logiciel devra être redémarré après.")
        if not messagebox.askyesno("Confirmer la mise à jour", msg):
            return

        self.upd_status.config(text="⏳ Mise à jour en cours…", fg=ACCENT)
        self.upd_apply_btn.config(state="disabled")
        def worker():
            try:
                from updater import apply_update_git, apply_update_zip
                if info.get("is_git_install"):
                    result = apply_update_git()
                else:
                    result = apply_update_zip(info["zipball_url"])
                self.root.after(0, lambda: self._on_update_applied(result))
            except Exception as e:
                self.root.after(0, lambda e=e: self.upd_status.config(
                    text=f"❌ Erreur : {e}", fg=RED))
        threading.Thread(target=worker, daemon=True).start()

    def _on_update_applied(self, result):
        if result.get("success"):
            self.upd_status.config(
                text=f"✅ {result['message']}\n\n"
                     f"➜ Ferme l'application et relance run.bat pour activer la nouvelle version.",
                fg=GREEN)
            messagebox.showinfo("Mise à jour OK",
                                  "Mise à jour appliquée. Ferme et relance l'application.")
            # Auto-refresh changelog après update réussi
            try:
                self._load_changelog()
            except Exception:
                pass
        else:
            self.upd_status.config(text=f"❌ {result['message']}", fg=RED)
            self.upd_apply_btn.config(state="normal", bg=GREEN, fg=BG)

    def _open_url(self, url):
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'URL :\n{e}")

    def _load_changelog(self):
        """Charge CHANGELOG.md depuis le disque local, fallback sur GitHub."""
        if not hasattr(self, "changelog_text"):
            return
        chg_path = Path(__file__).parent / "CHANGELOG.md"
        content = ""
        if chg_path.is_file():
            try:
                content = chg_path.read_text(encoding="utf-8")
            except Exception:
                content = ""

        # Fallback : télécharge depuis GitHub si CHANGELOG.md manquant
        if not content:
            try:
                url = "https://raw.githubusercontent.com/akalavol/LoRA-Dataset-Coach/main/CHANGELOG.md"
                with urllib.request.urlopen(url, timeout=8) as resp:
                    content = resp.read().decode("utf-8")
            except Exception as e:
                content = f"Impossible de charger CHANGELOG.md :\n{e}\n\n" \
                           f"Visite directement le repo GitHub."

        # Rendu markdown basique avec coloration
        self.changelog_text.config(state="normal")
        self.changelog_text.delete("1.0", "end")
        for line in content.splitlines():
            if line.startswith("# "):
                self.changelog_text.insert("end", line[2:] + "\n", "h1")
            elif line.startswith("## "):
                self.changelog_text.insert("end", line[3:] + "\n", "h2")
            elif line.startswith("### "):
                self.changelog_text.insert("end", line[4:] + "\n", "h3")
            elif line.startswith("- ") or line.startswith("  - "):
                self.changelog_text.insert("end", line + "\n", "bullet")
            else:
                self.changelog_text.insert("end", line + "\n")
        self.changelog_text.config(state="disabled")

    # ============================================================
    # ANALYZER : coherence dataset LoRA
    # ============================================================
    def _build_analyzer_tab(self, parent):
        frame = Frame(parent, bg=BG, padx=15, pady=12)
        parent.add(frame, text="  📊 Analyseur dataset  ")

        Label(frame, text="Analyseur de coherence dataset LoRA",
              font=FONT_H1, fg=ACCENT2, bg=BG).pack(anchor="w")
        Label(frame, text="Verifie qu'un dossier d'images est coherent pour entrainer un LoRA visage : "
                          "meme personne, visages detectes, resolutions, etc.",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG, wraplength=900, justify="left").pack(anchor="w", pady=(0, 10))

        # Selection dossier
        sel_frame = Frame(frame, bg=CARD, padx=10, pady=8)
        sel_frame.pack(fill="x", pady=4)
        Label(sel_frame, text="📂 Dossier dataset :",
              font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        self.analyzer_path = StringVar(value=self.cfg.get("datasets_dir", r"C:\AI\datasets"))
        entry = tk.Entry(sel_frame, textvariable=self.analyzer_path, font=FONT_MONO,
                         bg=BG2, fg=TEXT, insertbackground=TEXT, relief="flat")
        entry.pack(side="left", fill="x", expand=True, ipady=5, padx=8)
        make_button(sel_frame, "📂", lambda: self._pick_folder(self.analyzer_path)).pack(side="left", padx=4)
        make_button(sel_frame, "🔍 Analyser", self._run_analyzer, primary=True, width=14).pack(side="left", padx=4)
        self.analyzer_export_btn = Button(sel_frame, text="📄 Export PDF", font=FONT_BODY,
                                           bg=CARD_HI, fg=TEXT_DIM, relief="flat",
                                           padx=10, pady=6, cursor="hand2",
                                           state="disabled",
                                           command=self._export_analyzer_pdf)
        self.analyzer_export_btn.pack(side="left", padx=4)

        self.analyzer_move_btn = Button(sel_frame, text="🗑 Déplacer ratés", font=FONT_BODY,
                                         bg=CARD_HI, fg=TEXT_DIM, relief="flat",
                                         padx=10, pady=6, cursor="hand2",
                                         state="disabled",
                                         command=self._move_rejected_photos)
        self.analyzer_move_btn.pack(side="left", padx=4)

        self.analyzer_upscale_btn = Button(sel_frame, text="🔧 Floues → upscale", font=FONT_BODY,
                                            bg=CARD_HI, fg=TEXT_DIM, relief="flat",
                                            padx=10, pady=6, cursor="hand2",
                                            state="disabled",
                                            command=self._move_blurry_to_upscale)
        self.analyzer_upscale_btn.pack(side="left", padx=4)

        self.analyzer_kohya_btn = Button(sel_frame, text="🧬 Préparer Kohya", font=FONT_BODY,
                                          bg=CARD_HI, fg=TEXT_DIM, relief="flat",
                                          padx=10, pady=6, cursor="hand2",
                                          state="disabled",
                                          command=self._prepare_kohya)
        self.analyzer_kohya_btn.pack(side="left", padx=4)

        self.analyzer_mask_btn = Button(sel_frame, text="🎭 Masques sujet", font=FONT_BODY,
                                         bg=CARD_HI, fg=TEXT_DIM, relief="flat",
                                         padx=10, pady=6, cursor="hand2",
                                         state="disabled",
                                         command=self._generate_subject_masks)
        self.analyzer_mask_btn.pack(side="left", padx=4)

        self.analyzer_gen_btn = Button(sel_frame, text="✨ Générer manques", font=FONT_BODY,
                                        bg=CARD_HI, fg=TEXT_DIM, relief="flat",
                                        padx=10, pady=6, cursor="hand2",
                                        state="disabled",
                                        command=self._generate_targeted_workflows)
        self.analyzer_gen_btn.pack(side="left", padx=4)

        # ===== Photo de reference (optionnelle) =====
        ref_frame = Frame(frame, bg=CARD, padx=10, pady=8)
        ref_frame.pack(fill="x", pady=4)
        Label(ref_frame, text="📷 Photo de référence :",
              font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        self.analyzer_ref_path = StringVar(value="")
        ref_entry = tk.Entry(ref_frame, textvariable=self.analyzer_ref_path, font=FONT_MONO,
                             bg=BG2, fg=TEXT, insertbackground=TEXT, relief="flat")
        ref_entry.pack(side="left", fill="x", expand=True, ipady=5, padx=8)
        make_button(ref_frame, "📂", self._pick_ref_image).pack(side="left", padx=4)
        make_button(ref_frame, "✕", lambda: self.analyzer_ref_path.set("")).pack(side="left", padx=2)
        Label(ref_frame, text="(optionnel — vérifie que c'est la bonne personne)",
              font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(side="left", padx=8)

        # ===== Mode captioner =====
        cap_frame = Frame(frame, bg=CARD, padx=10, pady=6)
        cap_frame.pack(fill="x", pady=4)
        Label(cap_frame, text="🏷 Captions :", font=FONT_BODY, fg=TEXT, bg=CARD).pack(side="left")
        self.captioner_mode = StringVar(value="wd14")
        for val, lbl, tip in (
            ("wd14",       "WD14 tags",       "Tags booru SDXL/Kohya (~30 s, 330 Mo)"),
            ("natural",    "Florence-2",      "Caption naturelle, rapide mais hallucine sur personnes"),
            ("joycaption", "JoyCaption ⭐",   "STANDARD 2026 Flux/Wan persona (lent, 4-8 Go modèle)"),
            ("all",        "Tous",            "WD14 + Florence + JoyCaption (très lent, exhaustif)"),
        ):
            tk.Radiobutton(cap_frame, text=lbl, variable=self.captioner_mode, value=val,
                            font=FONT_BODY, fg=TEXT, bg=CARD, selectcolor=BG2,
                            activebackground=CARD, activeforeground=TEXT).pack(side="left", padx=6)
        Label(cap_frame, text="(WD14 pour SDXL · JoyCaption pour Flux/Wan)",
              font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(side="left", padx=8)

        # Barre de progression + ligne phase + ETA
        progress_outer = Frame(frame, bg=BG)
        progress_outer.pack(fill="x", pady=(8, 4))

        # Ligne 1 : phase en cours
        self.analyzer_phase = Label(progress_outer, text="⏸ En attente",
                                     font=FONT_BODY, fg=TEXT_DIM, bg=BG, anchor="w")
        self.analyzer_phase.pack(fill="x")

        # Ligne 2 : barre + pourcentage + ETA + fichier
        bar_row = Frame(progress_outer, bg=BG)
        bar_row.pack(fill="x", pady=(4, 0))

        self.analyzer_progress = ttk.Progressbar(bar_row, mode="determinate",
                                                  length=400, maximum=100)
        self.analyzer_progress.pack(side="left", fill="x", expand=True)

        self.analyzer_percent = Label(bar_row, text="0%",
                                       font=FONT_BODY, fg=ACCENT, bg=BG, width=6, anchor="e")
        self.analyzer_percent.pack(side="left", padx=(8, 4))

        self.analyzer_eta = Label(bar_row, text="",
                                   font=FONT_SMALL, fg=TEXT_DIM, bg=BG, width=18, anchor="w")
        self.analyzer_eta.pack(side="left", padx=4)

        # Ligne 3 : fichier en cours
        self.analyzer_status = Label(progress_outer, text="",
                                      font=FONT_SMALL, fg=TEXT_DIM, bg=BG, anchor="w")
        self.analyzer_status.pack(fill="x", pady=(2, 0))

        # ===== PREVIEW LIVE de la photo en cours d'analyse =====
        if _HAS_PIL:
            preview_row = Frame(frame, bg=BG)
            preview_row.pack(fill="x", pady=(8, 4))

            # Image courante (gauche)
            self.preview_current_frame = Frame(preview_row, bg=CARD, padx=6, pady=6)
            self.preview_current_frame.pack(side="left")
            Label(self.preview_current_frame, text="📷 En cours d'analyse",
                  font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack()
            self.preview_current_label = Label(self.preview_current_frame, bg=CARD,
                                                width=200, height=200)
            self.preview_current_label.pack(pady=2)
            self.preview_current_name = Label(self.preview_current_frame, text="",
                                               font=FONT_SMALL, fg=TEXT_DIM, bg=CARD,
                                               wraplength=200)
            self.preview_current_name.pack()

            # Image de reference (droite, si une ref est definie)
            self.preview_ref_frame = Frame(preview_row, bg=CARD, padx=6, pady=6)
            self.preview_ref_frame.pack(side="left", padx=12)
            Label(self.preview_ref_frame, text="📌 Référence",
                  font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack()
            self.preview_ref_label = Label(self.preview_ref_frame, bg=CARD,
                                            width=200, height=200)
            self.preview_ref_label.pack(pady=2)
            self.preview_ref_name = Label(self.preview_ref_frame, text="(aucune)",
                                           font=FONT_SMALL, fg=TEXT_DIM, bg=CARD,
                                           wraplength=200)
            self.preview_ref_name.pack()

            # Verdict live de la photo courante (a droite des 2 previews)
            self.preview_verdict_frame = Frame(preview_row, bg=CARD, padx=12, pady=6)
            self.preview_verdict_frame.pack(side="left", fill="both", expand=True, padx=4)
            Label(self.preview_verdict_frame, text="🔎 Dernier verdict",
                  font=FONT_SMALL, fg=TEXT_DIM, bg=CARD, anchor="w").pack(anchor="w")
            self.preview_verdict_text = Label(self.preview_verdict_frame, text="",
                                               font=FONT_BODY, fg=TEXT, bg=CARD,
                                               anchor="nw", justify="left", wraplength=400)
            self.preview_verdict_text.pack(anchor="nw", fill="both", expand=True, pady=4)

            # References vers les PhotoImages (Tk les garbage-collect sinon)
            self._preview_current_photo = None
            self._preview_ref_photo = None

        # Variables pour l'ETA
        self._analyzer_start_time = None
        self._analyzer_total = 0

        # Bloc verdict global (cache par defaut)
        self.analyzer_verdict_frame = Frame(frame, bg=CARD, padx=15, pady=10)
        # pas pack maintenant - apparait apres analyse

        # Resume
        self.analyzer_summary = Label(frame, text="(aucune analyse lancee)", font=FONT_BODY,
                                       fg=TEXT_DIM, bg=BG, justify="left")
        self.analyzer_summary.pack(anchor="w", pady=8)

        # Tableau resultats
        tree_frame = Frame(frame, bg=CARD)
        tree_frame.pack(fill="both", expand=True)
        self.analyzer_tree = ttk.Treeview(tree_frame,
                                          columns=("res", "sharp", "face", "size", "yaw", "expr", "sim", "body", "qual", "lora"),
                                          show="tree headings", style="Dark.Treeview")
        self.analyzer_tree.heading("#0", text="Image")
        self.analyzer_tree.heading("res", text="Resol.")
        self.analyzer_tree.heading("sharp", text="Net.")
        self.analyzer_tree.heading("face", text="Vis.")
        self.analyzer_tree.heading("size", text="%vis.")
        self.analyzer_tree.heading("yaw", text="Pose°")
        self.analyzer_tree.heading("expr", text="Expression")
        self.analyzer_tree.heading("sim", text="S.vis")
        self.analyzer_tree.heading("body", text="S.cor")
        self.analyzer_tree.heading("qual", text="Qualite")
        self.analyzer_tree.heading("lora", text="🧬 LoRA - raison")
        self.analyzer_tree.column("#0", width=160)
        self.analyzer_tree.column("res", width=75, anchor="center")
        self.analyzer_tree.column("sharp", width=50, anchor="e")
        self.analyzer_tree.column("face", width=35, anchor="center")
        self.analyzer_tree.column("size", width=55, anchor="e")
        self.analyzer_tree.column("yaw", width=50, anchor="e")
        self.analyzer_tree.column("expr", width=100, anchor="w")
        self.analyzer_tree.column("sim", width=50, anchor="e")
        self.analyzer_tree.column("body", width=50, anchor="e")
        self.analyzer_tree.column("qual", width=85)
        self.analyzer_tree.column("lora", width=230)
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.analyzer_tree.yview)
        self.analyzer_tree.configure(yscrollcommand=sb.set)
        self.analyzer_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # Tags couleur par verdict
        self.analyzer_tree.tag_configure("ok", foreground=GREEN)
        self.analyzer_tree.tag_configure("warn", foreground=YELLOW)
        self.analyzer_tree.tag_configure("err", foreground=RED)
        # Double-clic ouvre l'image en grand
        self.analyzer_tree.bind("<Double-1>", self._on_tree_double_click)
        # Clic simple : update le preview courant (rapide)
        self.analyzer_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # === Zone d'erreur (cachee par defaut, apparait sur crash) ===
        self.analyzer_error_frame = Frame(frame, bg="#3a1f24", padx=10, pady=8)
        # NE PAS pack maintenant - on l'affiche seulement sur crash

        err_header = Frame(self.analyzer_error_frame, bg="#3a1f24")
        err_header.pack(fill="x")
        Label(err_header, text="❌ CRASH DETECTE", font=FONT_H1, fg=RED, bg="#3a1f24").pack(side="left")
        Button(err_header, text="📋 Copier", font=FONT_SMALL,
               bg=CARD_HI, fg=TEXT, relief="flat", padx=8, pady=2, cursor="hand2",
               command=self._copy_analyzer_error).pack(side="right", padx=4)
        Button(err_header, text="✕ Fermer", font=FONT_SMALL,
               bg=CARD_HI, fg=TEXT, relief="flat", padx=8, pady=2, cursor="hand2",
               command=lambda: self.analyzer_error_frame.pack_forget()).pack(side="right")

        # Text widget scrollable pour le traceback complet
        err_text_frame = Frame(self.analyzer_error_frame, bg="#3a1f24")
        err_text_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.analyzer_error_text = Text(err_text_frame, height=8, font=FONT_MONO,
                                         bg=BG2, fg="#f5c2c7", insertbackground=TEXT,
                                         relief="flat", wrap="word", padx=8, pady=6)
        err_sb = ttk.Scrollbar(err_text_frame, orient="vertical", command=self.analyzer_error_text.yview)
        self.analyzer_error_text.configure(yscrollcommand=err_sb.set)
        self.analyzer_error_text.pack(side="left", fill="both", expand=True)
        err_sb.pack(side="right", fill="y")

    def _run_analyzer(self):
        folder = self.analyzer_path.get()
        if not Path(folder).is_dir():
            messagebox.showerror("Erreur", f"Dossier introuvable :\n{folder}")
            return
        if not Path(self.comfyui_py).exists():
            messagebox.showerror("Erreur", f"Python ComfyUI-future introuvable :\n{self.comfyui_py}\n\n"
                                            "L'analyseur a besoin d'insightface (installe dans ComfyUI-future).")
            return

        # Photo de reference (optionnelle)
        ref = self.analyzer_ref_path.get().strip()
        if ref and not Path(ref).is_file():
            messagebox.showerror("Erreur", f"Photo de reference introuvable :\n{ref}\n\n"
                                            "Laisse vide pour analyser sans reference.")
            return
        # Affiche la ref dans le preview (au cas ou elle a ete tapee manuellement)
        self._update_preview_ref(ref)

        import time as _time
        self.analyzer_tree.delete(*self.analyzer_tree.get_children())
        self.analyzer_summary.config(text="", fg=TEXT_DIM)
        self.analyzer_progress.config(value=0, mode="indeterminate")
        self.analyzer_progress.start(15)
        self.analyzer_phase.config(text="⚙ Chargement insightface (~5-10s)...", fg=ACCENT)
        self.analyzer_percent.config(text="")
        self.analyzer_eta.config(text="")
        self.analyzer_status.config(text="")
        self.analyzer_error_frame.pack_forget()  # Cache l'erreur si visible
        self.analyzer_error_text.delete("1.0", "end")
        self._analyzer_start_time = _time.time()
        self._analyzer_total = 0
        self._analyzer_stderr_buffer = []
        self.root.update_idletasks()

        cap_mode = self.captioner_mode.get() if hasattr(self, "captioner_mode") else "wd14"
        threading.Thread(target=self._analyze_subprocess, args=(folder, ref, cap_mode), daemon=True).start()

    def _analyze_subprocess(self, folder, ref="", cap_mode="wd14"):
        import time as _time
        script = str(Path(__file__).parent / "analyze_dataset.py")
        try:
            cmd = [self.comfyui_py, script, folder, "full"]
            # On a besoin de cap_mode en position 4, donc ref doit etre en position 3 meme vide
            cmd.append(ref if ref else "")
            cmd.append(cap_mode)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1
            )

            def fmt_eta(seconds):
                if seconds < 60:
                    return f"~{int(seconds)}s"
                m, s = divmod(int(seconds), 60)
                return f"~{m}m{s:02d}s"

            def read_progress():
                for raw_line in iter(proc.stderr.readline, ""):
                    line = raw_line.strip()
                    self._analyzer_stderr_buffer.append(raw_line)  # Capture TOUT pour crash report
                    if line.startswith("TOTAL "):
                        total = int(line.split()[1])
                        self._analyzer_total = total
                        self.root.after(0, lambda t=total: (
                            self.analyzer_progress.stop(),
                            self.analyzer_progress.config(mode="determinate", maximum=t, value=0),
                            self.analyzer_phase.config(text=f"🔍 Analyse en cours — {t} images", fg=ACCENT),
                            self.analyzer_percent.config(text="0%"),
                            self.analyzer_eta.config(text="")
                        ))
                    elif line.startswith("PROGRESS "):
                        try:
                            parts = line.split(" ", 2)
                            cur_total = parts[1].split("/")
                            cur = int(cur_total[0])
                            tot = int(cur_total[1])
                            name = parts[2] if len(parts) > 2 else ""
                            # Calcul ETA
                            elapsed = _time.time() - self._analyzer_start_time
                            pct = (cur / tot) * 100
                            if cur > 0 and elapsed > 1:
                                rate = cur / elapsed
                                remaining = (tot - cur) / rate if rate > 0 else 0
                                eta_str = f"⏱ {fmt_eta(remaining)} restant"
                            else:
                                eta_str = "⏱ calcul..."
                            self.root.after(0, lambda c=cur, t=tot, n=name, p=pct, e=eta_str: (
                                self.analyzer_progress.config(value=c),
                                self.analyzer_percent.config(text=f"{p:.0f}%"),
                                self.analyzer_eta.config(text=e),
                                self.analyzer_status.config(text=f"📷 {c}/{t} — {n[:50]}", fg=TEXT_DIM)
                            ))
                        except Exception:
                            pass
                    elif line.startswith("PREVIEW "):
                        # Path absolu de l'image en cours d'analyse
                        img_path = line[8:].strip()
                        self.root.after(0, lambda p=img_path: self._update_preview_current(p))
                    elif line.startswith("IMGINFO "):
                        # Mini-verdict JSON de l'image qu'on vient de finir
                        try:
                            mini = json.loads(line[8:])
                            self.root.after(0, lambda m=mini: self._update_preview_verdict(m))
                        except Exception:
                            pass
                    elif line.startswith("STEP "):
                        msg = line[5:]
                        self.root.after(0, lambda m=msg: self.analyzer_phase.config(text=f"⚙ {m}", fg=ACCENT))
                    elif line.startswith("PROGRESS_DONE"):
                        self.root.after(0, lambda: (
                            self.analyzer_phase.config(text="⚙ Calculs de similarite en cours...", fg=ACCENT),
                            self.analyzer_status.config(text="")
                        ))

            t = threading.Thread(target=read_progress, daemon=True)
            t.start()

            stdout_data, _ = proc.communicate(timeout=600)
            t.join(timeout=2)

            if proc.returncode != 0:
                full_stderr = "".join(self._analyzer_stderr_buffer)
                self.root.after(0, lambda: self._analyzer_done_error(
                    f"Le script Python a quitte avec le code {proc.returncode}", full_stderr))
                return

            try:
                data = json.loads(stdout_data)
            except json.JSONDecodeError as e:
                full = "".join(self._analyzer_stderr_buffer) + "\n\n--- STDOUT (debut) ---\n" + stdout_data[:3000]
                self.root.after(0, lambda: self._analyzer_done_error(
                    f"Sortie JSON invalide : {e}", full))
                return
            if "error" in data:
                full = "".join(self._analyzer_stderr_buffer) + "\n\n--- ERREUR ---\n" + str(data.get("error"))
                self.root.after(0, lambda: self._analyzer_done_error(str(data["error"]), full))
                return
            self.root.after(0, lambda: self._show_analyzer_result(data))
        except subprocess.TimeoutExpired:
            full = "".join(self._analyzer_stderr_buffer)
            self.root.after(0, lambda: self._analyzer_done_error("Timeout (>10 min)", full))
        except Exception as e:
            import traceback
            full = "".join(self._analyzer_stderr_buffer) + "\n\n--- TRACEBACK GUI ---\n" + traceback.format_exc()
            self.root.after(0, lambda: self._analyzer_done_error(str(e), full))

    def _analyzer_done_error(self, msg, full_details=""):
        self.analyzer_progress.stop()
        self.analyzer_progress.config(mode="determinate", value=0)
        self.analyzer_phase.config(text=f"❌ Echec : {msg[:100]}", fg=RED)
        self.analyzer_percent.config(text="")
        self.analyzer_eta.config(text="")
        self.analyzer_status.config(text="")

        # Affiche la zone d'erreur scrollable avec le detail complet
        if full_details:
            self.analyzer_error_text.delete("1.0", "end")
            self.analyzer_error_text.insert("1.0", full_details)
            self.analyzer_error_text.see("end")  # Scroll vers le bas (souvent l'erreur)
            self.analyzer_error_frame.pack(fill="both", expand=False, pady=(8, 0), padx=10, before=self.analyzer_tree.master)

    def _copy_analyzer_error(self):
        text = self.analyzer_error_text.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("Erreur copiee dans le presse-papier")

    def _export_analyzer_pdf(self):
        """Genere un PDF paysage A4 du compte rendu de la derniere analyse."""
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            messagebox.showinfo("Info", "Lance d'abord une analyse.")
            return

        # Nom par defaut base sur le dossier + date
        from pathlib import Path as _P
        folder = self.analyzer_path.get()
        default_name = f"rapport_dataset_{_P(folder).name}_{datetime.now():%Y%m%d_%H%M}.pdf"

        out = filedialog.asksaveasfilename(
            title="Enregistrer le rapport PDF",
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not out:
            return

        try:
            # Import a la demande
            import sys as _sys
            _sys.path.insert(0, str(_P(__file__).parent))
            import export_pdf
            export_pdf.export(data, folder, out)
            self.status_var.set(f"PDF genere : {_P(out).name}")
            # Propose d'ouvrir
            if messagebox.askyesno("PDF genere", f"Rapport sauve dans :\n{out}\n\nL'ouvrir maintenant ?"):
                os.startfile(out)
        except Exception as e:
            import traceback
            messagebox.showerror("Erreur PDF",
                f"Echec generation PDF :\n{e}\n\n{traceback.format_exc()[-400:]}")

    def _show_analyzer_result(self, data):
        import time as _time
        # Finalise la barre
        self.analyzer_progress.stop()
        try:
            maxv = int(self.analyzer_progress.cget("maximum"))
            self.analyzer_progress.config(mode="determinate", value=maxv)
        except Exception:
            self.analyzer_progress.config(value=100)

        # Calcul duree totale
        elapsed = _time.time() - (self._analyzer_start_time or _time.time())
        m, s = divmod(int(elapsed), 60)
        dur = f"{m}m{s:02d}s" if m else f"{s}s"
        n = data.get("summary", {}).get("total_images", 0)
        self.analyzer_phase.config(text=f"✅ Analyse terminee — {n} images en {dur}", fg=GREEN)
        self.analyzer_percent.config(text="100%")
        self.analyzer_eta.config(text="")
        self.analyzer_status.config(text="")

        # Garde les donnees + active les boutons
        self._last_analysis_data = data
        self.analyzer_export_btn.config(state="normal", bg=ACCENT, fg=BG)
        unusable = data.get("summary", {}).get("lora_unusable", 0)
        if unusable > 0:
            self.analyzer_move_btn.config(state="normal", bg=RED, fg=BG,
                                          text=f"🗑 Déplacer ratés ({unusable})")
        else:
            self.analyzer_move_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                          text="🗑 Aucun raté à déplacer")

        # Bouton upscale floues
        n_blurry = len(data.get("summary", {}).get("blurry_recoverable", []))
        if n_blurry > 0:
            self.analyzer_upscale_btn.config(state="normal", bg=YELLOW, fg=BG,
                                              text=f"🔧 Floues → upscale ({n_blurry})")
        else:
            self.analyzer_upscale_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                              text="🔧 Aucune floue récupérable")

        # Bouton Kohya
        n_viable = data.get("summary", {}).get("lora_viable", 0)
        n_border = data.get("summary", {}).get("lora_borderline", 0)
        if n_viable + n_border >= 10:
            self.analyzer_kohya_btn.config(state="normal", bg=GREEN, fg=BG,
                                            text=f"🧬 Préparer Kohya ({n_viable + n_border} photos)")
        else:
            self.analyzer_kohya_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                            text="🧬 Pas assez de photos viables (min 10)")

        # Bouton masques (Lot D)
        if n_viable + n_border >= 1:
            self.analyzer_mask_btn.config(state="normal", bg=ACCENT2, fg=BG,
                                          text=f"🎭 Masques sujet ({n_viable + n_border} photos)")
        else:
            self.analyzer_mask_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                          text="🎭 Pas d'image viable")

        # Bouton générer manques (Lot E)
        n_suggestions = len(data.get("summary", {}).get("next_to_generate", []))
        if n_suggestions > 0:
            self.analyzer_gen_btn.config(state="normal", bg=ACCENT, fg=BG,
                                         text=f"✨ Générer manques ({n_suggestions})")
        else:
            self.analyzer_gen_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                         text="✨ Aucun manque détecté")

        # Affiche le verdict global (avec infos ref + overfit + scores target + diversite)
        s = data.get("summary", {})
        self._show_verdict_block(s.get("verdict", {}),
                                  ref_info=s.get("reference"),
                                  ref_match=s.get("reference_match"),
                                  overfit_alerts=s.get("tag_overfit_alerts", []),
                                  top_tags=s.get("tag_frequency_top", []),
                                  target_scores=s.get("target_scores", {}),
                                  diversity=s.get("diversity", {}),
                                  ar_distribution=s.get("aspect_ratio_distribution", {}))

    def _show_verdict_block(self, verdict, ref_info=None, ref_match=None,
                              overfit_alerts=None, top_tags=None,
                              target_scores=None, diversity=None,
                              ar_distribution=None):
        # Vide le bloc precedent
        for w in self.analyzer_verdict_frame.winfo_children():
            w.destroy()

        if not verdict:
            self.analyzer_verdict_frame.pack_forget()
            return

        grade = verdict.get("grade", "?")
        desc = verdict.get("grade_desc", "")
        viable = verdict.get("viable_now", 0)
        after = verdict.get("after_cleanup", 0)
        target_min = verdict.get("target_min", 20)
        target_ideal = verdict.get("target_ideal", 30)
        actions = verdict.get("actions", [])

        # Couleur selon grade
        grade_color = {
            "A": GREEN, "B": GREEN, "B+": GREEN, "B-": YELLOW,
            "C": YELLOW, "D": RED, "F": RED,
        }.get(grade, TEXT)

        # Ligne 1 : grade enorme + description
        top = Frame(self.analyzer_verdict_frame, bg=CARD)
        top.pack(fill="x")
        Label(top, text=grade, font=("Segoe UI", 32, "bold"),
              fg=grade_color, bg=CARD).pack(side="left", padx=(0, 15))
        right = Frame(top, bg=CARD)
        right.pack(side="left", fill="x", expand=True)
        Label(right, text=f"VERDICT DATASET : {desc}",
              font=FONT_H1, fg=grade_color, bg=CARD, anchor="w").pack(fill="x")
        Label(right,
              text=f"{viable} photos viables maintenant → {after} après cleanup  (cible : {target_min}-{target_ideal})",
              font=FONT_BODY, fg=TEXT_DIM, bg=CARD, anchor="w").pack(fill="x")

        # Bloc reference (si presente)
        if ref_info and "error" not in (ref_info or {}):
            ref_box = Frame(self.analyzer_verdict_frame, bg=CARD)
            ref_box.pack(fill="x", pady=(8, 0))
            if ref_match:
                ok = ref_match.get("ok", 0)
                doubt = ref_match.get("doubt", 0)
                wrong = ref_match.get("wrong", 0)
                avg = ref_match.get("avg", 0)
                if wrong == 0 and doubt == 0:
                    color = GREEN
                    txt = f"📷 Réf : {ref_info.get('name', '?')}  →  ✅ {ok} photos correspondent (avg {avg})"
                elif wrong == 0:
                    color = YELLOW
                    txt = f"📷 Réf : {ref_info.get('name', '?')}  →  💡 {ok} OK, {doubt} douteuses (avg {avg})"
                else:
                    color = RED
                    txt = f"📷 Réf : {ref_info.get('name', '?')}  →  ❌ {wrong} mauvaise(s) personne(s), {doubt} douteuse(s), {ok} OK"
                Label(ref_box, text=txt, font=FONT_BODY, fg=color, bg=CARD,
                      anchor="w", wraplength=900, justify="left").pack(fill="x")
        elif ref_info and "error" in ref_info:
            Label(self.analyzer_verdict_frame,
                  text=f"📷 Réf : ⚠️ {ref_info['error']}",
                  font=FONT_BODY, fg=YELLOW, bg=CARD,
                  anchor="w", wraplength=900, justify="left").pack(fill="x", pady=(8, 0))

        # ===== Alertes overfit (tags accessoires trop frequents) =====
        if overfit_alerts:
            Label(self.analyzer_verdict_frame, text="⚠️ Risques d'overfit :",
                  font=FONT_H1, fg=YELLOW, bg=CARD, anchor="w").pack(fill="x", pady=(10, 4))
            # Groupe par categorie pour affichage propre
            by_cat = {}
            for a in overfit_alerts:
                by_cat.setdefault(a["category"], []).append(a)
            for cat, alerts in by_cat.items():
                alerts.sort(key=lambda x: -x["count"])
                top = alerts[:3]
                txt = ", ".join(f"« {a['tag']} » ({int(a['ratio']*100)}%)" for a in top)
                color = RED if any(a["severity"] == "ALERTE" for a in top) else YELLOW
                Label(self.analyzer_verdict_frame,
                      text=f"  • {cat.capitalize()} : {txt}",
                      font=FONT_BODY, fg=color, bg=CARD,
                      anchor="w", wraplength=900, justify="left").pack(fill="x")

        # ===== SCORES PAR TARGET (5 familles) =====
        if target_scores:
            Label(self.analyzer_verdict_frame, text="🎯 Scores par famille de target :",
                  font=FONT_H1, fg=ACCENT, bg=CARD, anchor="w").pack(fill="x", pady=(10, 4))
            for fam, sc in target_scores.items():
                grade = sc.get("grade", "?")
                score = sc.get("score", 0)
                reason = sc.get("reason", "")
                color = {"A+": GREEN, "A": GREEN, "B+": GREEN, "B": YELLOW,
                         "C": YELLOW, "D": RED, "F": RED}.get(grade, TEXT)
                row = Frame(self.analyzer_verdict_frame, bg=CARD)
                row.pack(fill="x", pady=1)
                Label(row, text=f"  {grade}", font=("Segoe UI", 12, "bold"),
                      fg=color, bg=CARD, width=4).pack(side="left")
                Label(row, text=f"{score}/100", font=FONT_SMALL, fg=color, bg=CARD,
                      width=8).pack(side="left")
                Label(row, text=fam, font=FONT_BODY, fg=TEXT, bg=CARD,
                      width=35, anchor="w").pack(side="left")
                Label(row, text=reason[:80], font=FONT_SMALL, fg=TEXT_DIM, bg=CARD,
                      anchor="w").pack(side="left", fill="x", expand=True)

        # ===== DIVERSITE =====
        if diversity and diversity.get("overall_score") is not None:
            div_score = diversity["overall_score"]
            div_color = GREEN if div_score >= 70 else (YELLOW if div_score >= 50 else RED)
            div_row = Frame(self.analyzer_verdict_frame, bg=CARD)
            div_row.pack(fill="x", pady=(10, 2))
            Label(div_row, text=f"🌈 Diversité : {int(div_score)}/100",
                  font=FONT_H1, fg=div_color, bg=CARD).pack(side="left")
            Label(div_row, text=f"  ({diversity.get('verdict', '?')})",
                  font=FONT_BODY, fg=TEXT_DIM, bg=CARD).pack(side="left", padx=8)
            if diversity.get("clip_clusters"):
                Label(div_row,
                      text=f"  · {diversity['clip_clusters']} clusters visuels CLIP",
                      font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(side="left", padx=8)

        # ===== AR DISTRIBUTION =====
        if ar_distribution and sum(ar_distribution.values()) > 0:
            total_ar = sum(ar_distribution.values())
            sq = ar_distribution.get("square", 0)
            po = ar_distribution.get("portrait", 0) + ar_distribution.get("tall_portrait", 0)
            la = ar_distribution.get("landscape", 0) + ar_distribution.get("wide_landscape", 0)
            ar_txt = (f"📐 Aspect ratio : "
                       f"⬜ {sq} carré · 📱 {po} portrait · 🖼 {la} paysage")
            Label(self.analyzer_verdict_frame, text=ar_txt,
                  font=FONT_BODY, fg=TEXT, bg=CARD, anchor="w").pack(fill="x", pady=2)

        # Ligne 2 : plan d'action
        if actions:
            Label(self.analyzer_verdict_frame, text="🎯 Plan d'action :",
                  font=FONT_H1, fg=ACCENT, bg=CARD, anchor="w").pack(fill="x", pady=(10, 4))
            for a in actions:
                Label(self.analyzer_verdict_frame, text=f"  • {a}",
                      font=FONT_BODY, fg=TEXT, bg=CARD,
                      anchor="w", wraplength=900, justify="left").pack(fill="x")

        # Pack au-dessus du resume
        self.analyzer_verdict_frame.pack(fill="x", padx=10, pady=(0, 6),
                                          before=self.analyzer_summary)

    def _generate_targeted_workflows(self):
        """Génère des workflows ComfyUI ciblés pour combler les manques détectés."""
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            return
        suggestions = data.get("summary", {}).get("next_to_generate", [])
        if not suggestions:
            messagebox.showinfo("Rien à faire", "Aucun manque détecté.")
            return

        # Dossier de sortie
        default_out = Path(self.analyzer_path.get()).parent / "targeted_workflows"
        out = filedialog.askdirectory(
            initialdir=str(default_out.parent),
            title="Dossier où exporter les workflows ciblés"
        )
        out_folder = Path(out) if out else default_out

        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import prompt_generator
            written = prompt_generator.export_workflows_for_suggestions(
                suggestions=suggestions,
                output_folder=out_folder,
                base_persona_desc="",  # à enrichir : peut être lu depuis une config
                reference_image="reference_face_1024.png",
                checkpoint="RealVisXL_V5.0_fp16.safetensors",
            )
        except Exception as e:
            import traceback
            messagebox.showerror("Erreur", f"{e}\n\n{traceback.format_exc()[-300:]}")
            return

        msg = f"✅ {len(written)} workflow(s) exporté(s) dans :\n{out_folder}\n\n"
        for w in written:
            msg += f"  • {w['category']} (x{w['count']}) → {Path(w['workflow_path']).name}\n"
        msg += "\n➜ Drag-and-drop un .json dans ComfyUI pour lancer la génération.\n"
        msg += "➜ Le batch_size est déjà réglé sur le nombre demandé."
        messagebox.showinfo("Workflows ciblés générés", msg)
        try:
            os.startfile(str(out_folder))
        except Exception:
            pass

    def _generate_subject_masks(self):
        """Genere des masques sujet (alpha BriaRMBG) a cote de chaque image viable.
        Format OneTrainer : <image>-masklabel.png noir/blanc."""
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            return

        # Popup de config
        dlg = tk.Toplevel(self.root)
        dlg.title("Générer masques sujet")
        dlg.configure(bg=BG)
        dlg.geometry("520x340")
        dlg.transient(self.root)
        dlg.grab_set()

        Label(dlg, text="🎭 Générer masques sujet (OneTrainer)",
              font=FONT_H1, fg=ACCENT, bg=BG).pack(anchor="w", padx=20, pady=(15, 4))
        Label(dlg,
              text="Génère <image>-masklabel.png à côté de chaque photo viable.\n"
                   "OneTrainer focalise la loss sur le sujet (gain qualité gros fond varié).",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG, justify="left",
              wraplength=480).pack(anchor="w", padx=20, pady=(0, 12))

        # Threshold (binarisation)
        thresh_var = tk.DoubleVar(value=0.5)
        thresh_enabled = tk.BooleanVar(value=True)

        tk.Checkbutton(dlg, text="Binariser (recommandé OneTrainer : sujet blanc pur, fond noir pur)",
                        variable=thresh_enabled, font=FONT_BODY, fg=TEXT, bg=BG,
                        selectcolor=CARD, activebackground=BG,
                        activeforeground=TEXT).pack(anchor="w", padx=20, pady=4)

        Label(dlg, text="Seuil de binarisation (0.3-0.7) :",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(anchor="w", padx=20, pady=(8, 0))
        tk.Scale(dlg, from_=0.1, to=0.9, resolution=0.05,
                  orient="horizontal", variable=thresh_var,
                  bg=CARD, fg=TEXT, troughcolor=BG2,
                  highlightthickness=0, length=400).pack(anchor="w", padx=20)

        # Viable only
        viable_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dlg, text="Uniquement photos viables/borderline",
                        variable=viable_var, font=FONT_BODY, fg=TEXT, bg=BG,
                        selectcolor=CARD, activebackground=BG,
                        activeforeground=TEXT).pack(anchor="w", padx=20, pady=6)

        # Boutons
        btn_row = Frame(dlg, bg=BG)
        btn_row.pack(side="bottom", fill="x", padx=20, pady=15)

        def go():
            th = thresh_var.get() if thresh_enabled.get() else None
            viable_only = viable_var.get()
            dlg.destroy()
            self._launch_mask_generation(data, th, viable_only)

        Button(btn_row, text="Annuler", font=FONT_BODY, bg=CARD_HI, fg=TEXT,
               relief="flat", padx=20, pady=8, command=dlg.destroy).pack(side="right", padx=4)
        Button(btn_row, text="🚀 Générer", font=FONT_BODY, bg=ACCENT2, fg=BG,
               relief="flat", padx=20, pady=8, command=go).pack(side="right", padx=4)

    def _launch_mask_generation(self, data, threshold, viable_only):
        """Lance la generation en subprocess (besoin de torch + briarmbg = ComfyUI-future Python)."""
        # Liste des images cibles
        if viable_only:
            names = [img.get("name") for img in data.get("images", [])
                     if img.get("lora_viable") in ("yes", "borderline") and img.get("name")]
        else:
            names = [img.get("name") for img in data.get("images", []) if img.get("name")]
        if not names:
            messagebox.showinfo("Rien à faire", "Aucune photo à traiter.")
            return

        folder = self.analyzer_path.get()
        self.status_var.set("Génération masques en cours...")
        self.analyzer_mask_btn.config(state="disabled", text="🎭 Génération...")

        def worker():
            import time as _time
            script = str(Path(__file__).parent / "mask_generator_local.py")
            try:
                # On invoque le module via -c pour passer la liste de noms
                # mais le plus simple : on fait un petit driver inline
                import tempfile
                # Driver script avec args
                driver = (
                    "import sys, json\n"
                    f"sys.path.insert(0, r'{Path(__file__).parent}')\n"
                    "from mask_generator_local import generate_masks_for_folder\n"
                    "data = json.loads(sys.stdin.read())\n"
                    "result = generate_masks_for_folder(\n"
                    "    folder=data['folder'],\n"
                    "    viable_names=data['names'],\n"
                    "    threshold=data['threshold'],\n"
                    "    progress_cb=lambda c, t, n: print(f'PROGRESS {c}/{t} {n}', file=sys.stderr, flush=True)\n"
                    ")\n"
                    "print(json.dumps(result))\n"
                )
                payload = json.dumps({
                    "folder": folder,
                    "names": names,
                    "threshold": threshold,
                })

                proc = subprocess.Popen(
                    [self.comfyui_py, "-c", driver],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                )

                def read_stderr():
                    for line in iter(proc.stderr.readline, ""):
                        ln = line.strip()
                        if ln.startswith("PROGRESS "):
                            try:
                                parts = ln.split(" ", 2)
                                cur, tot = parts[1].split("/")
                                fname = parts[2] if len(parts) > 2 else ""
                                self.root.after(0, lambda c=cur, t=tot, n=fname:
                                    self.status_var.set(f"Masque {c}/{t} — {n[:40]}"))
                            except Exception:
                                pass
                        elif ln.startswith("STEP "):
                            self.root.after(0, lambda m=ln[5:]:
                                self.status_var.set(f"⚙ {m}"))

                t = threading.Thread(target=read_stderr, daemon=True)
                t.start()

                stdout, _ = proc.communicate(input=payload, timeout=900)
                result = json.loads(stdout.strip()) if stdout.strip() else {"errors": ["sortie vide"]}
                self.root.after(0, lambda r=result: self._on_mask_done(r))
            except Exception as e:
                import traceback
                err = f"{e}\n\n{traceback.format_exc()[-500:]}"
                self.root.after(0, lambda e=err: messagebox.showerror("Erreur masques", e))
                self.root.after(0, lambda: self.analyzer_mask_btn.config(
                    state="normal", text="🎭 Masques sujet (réessayer)"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_mask_done(self, result):
        written = result.get("written", 0)
        skipped = result.get("skipped", 0)
        errs = result.get("errors", [])
        total = result.get("total", written + skipped)
        msg = (f"✅ {written} masque(s) généré(s)\n"
               f"⏭ {skipped} déjà existant(s)\n"
               f"📁 Format : <image>-masklabel.png à côté des photos\n\n"
               f"➜ Charge le dataset dans OneTrainer\n"
               f"➜ Concepts > Image augmentations > coche \"masked training\"\n"
               f"➜ Le LoRA ignorera le fond pendant l'entraînement (gain qualité)")
        if errs:
            msg += f"\n\n⚠️ {len(errs)} erreur(s) :\n" + "\n".join(errs[:3])
        messagebox.showinfo("Masques générés", msg)
        self.status_var.set(f"Masques : {written}/{total} générés")
        self.analyzer_mask_btn.config(
            state="normal", bg=ACCENT2, fg=BG,
            text=f"🎭 Masques sujet ({written} OK)"
        )

    def _move_blurry_to_upscale(self):
        """Deplace les photos floues mais recuperables dans _a_upscaler/ avec un README."""
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            return
        names = data.get("summary", {}).get("blurry_recoverable", []) or []
        if not names:
            messagebox.showinfo("Rien à faire", "Aucune photo floue récupérable.")
            return

        # Recupere les paths complets
        paths_by_name = {img["name"]: img["path"] for img in data.get("images", [])}
        to_move = [paths_by_name[n] for n in names if n in paths_by_name and Path(paths_by_name[n]).exists()]
        if not to_move:
            messagebox.showinfo("Rien à faire", "Photos introuvables (déjà déplacées ?)")
            return

        source_folder = Path(self.analyzer_path.get())
        upscale_folder = source_folder / "_a_upscaler"

        if not messagebox.askyesno(
            "Déplacer vers upscale",
            f"Déplacer {len(to_move)} photo(s) floue(s) dans :\n{upscale_folder}\n\n"
            f"Un README expliquant comment les upscaler dans ComfyUI (SUPIR / UltraSharp) "
            f"sera créé dans le dossier. Continuer ?"
        ):
            return

        upscale_folder.mkdir(exist_ok=True)
        moved = 0
        for src in to_move:
            try:
                src_p = Path(src)
                dst = upscale_folder / src_p.name
                if dst.exists():
                    dst = upscale_folder / f"{src_p.stem}_{moved}{src_p.suffix}"
                shutil.move(str(src_p), str(dst))
                # Deplace aussi le .txt caption s'il existe
                txt_src = src_p.with_suffix(".txt")
                if txt_src.exists():
                    try:
                        shutil.move(str(txt_src), str(dst.with_suffix(".txt")))
                    except Exception:
                        pass
                moved += 1
            except Exception as e:
                print(f"Move fail: {e}")

        # Ecrit le README
        readme_path = upscale_folder / "README.txt"
        readme_path.write_text(
            "PHOTOS FLOUES MAIS RECUPERABLES (nettete 50-100)\n"
            "=" * 60 + "\n\n"
            f"Genere le : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Nombre : {moved} photo(s)\n\n"
            "COMMENT LES SAUVER ?\n"
            "--------------------\n\n"
            "Option 1 : SUPIR (le meilleur pour les visages)\n"
            "  1. Lance ComfyUI (via AI Launcher)\n"
            "  2. Charge un workflow SUPIR (custom_nodes/ComfyUI-SUPIR)\n"
            "  3. LoadImage -> SUPIR Upscale (denoise ~0.3-0.4 pour ne pas trop changer)\n"
            "  4. Save Image -> reinjecte dans le dataset apres verification\n\n"
            "Option 2 : UltraSharp (rapide, generique)\n"
            "  1. ComfyUI -> Load image -> Upscale Image (using Model)\n"
            "  2. Modele : 4x-UltraSharp.pth (si installe dans models/upscale_models/)\n"
            "  3. Resize a 1024x1024 si necessaire\n\n"
            "Option 3 : Refusionner avec InstantID\n"
            "  Si la photo a un visage clair mais une nettete moyenne,\n"
            "  refais-la via le workflow 02 (InstantID dataset) avec la meme seed.\n\n"
            "APRES UPSCALE\n"
            "-------------\n"
            "1. Verifie visuellement (pas d'artefacts sur le visage)\n"
            "2. Remets les fichiers OK dans le dossier dataset parent\n"
            "3. Relance l'analyse (le cache reconnaitra les nouvelles versions)\n",
            encoding="utf-8"
        )

        messagebox.showinfo("Terminé",
                             f"✅ {moved} photo(s) déplacée(s) dans :\n{upscale_folder}\n\n"
                             f"Un README.txt explique comment les upscaler.")
        self.status_var.set(f"Déplacé {moved} photos vers _a_upscaler/")
        self.analyzer_upscale_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                          text="🔧 Relance l'analyse")

    def _prepare_kohya(self):
        """Dialog de selection target + preparation LoRA multi-format."""
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            return

        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import lora_prep
        except Exception as e:
            messagebox.showerror("Erreur", f"Module lora_prep introuvable :\n{e}")
            return

        # === Popup de selection target + nom + captioner ===
        dlg = tk.Toplevel(self.root)
        dlg.title("Préparer dataset LoRA")
        dlg.configure(bg=BG)
        dlg.geometry("620x520")
        dlg.transient(self.root)
        dlg.grab_set()

        Label(dlg, text="🧬 Préparer un dataset LoRA",
              font=FONT_H1, fg=ACCENT, bg=BG).pack(anchor="w", padx=20, pady=(15, 4))
        Label(dlg, text="Le crop, la résolution et le format de config s'adaptent au target choisi.",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(anchor="w", padx=20, pady=(0, 12))

        # Trigger word
        Label(dlg, text="Nom de la persona (trigger word) :",
              font=FONT_BODY, fg=TEXT, bg=BG).pack(anchor="w", padx=20)
        persona_var = StringVar(value="persona")
        persona_entry = tk.Entry(dlg, textvariable=persona_var, font=FONT_MONO,
                                  bg=CARD, fg=TEXT, insertbackground=TEXT, relief="flat")
        persona_entry.pack(fill="x", padx=20, ipady=5, pady=(2, 10))

        # Target (groupé par catégorie)
        Label(dlg, text="Trainer / modèle cible :",
              font=FONT_BODY, fg=TEXT, bg=BG).pack(anchor="w", padx=20)
        target_var = StringVar(value="sdxl_kohya")
        # Construit la liste groupee : [--- Photo réaliste ---, sdxl_kohya, ...]
        targets_by_cat = {}
        for key, _ in lora_prep.list_targets():
            cat = lora_prep.get_target_category(key)
            targets_by_cat.setdefault(cat, []).append(key)
        cat_order = ["image_photo", "image_anime", "video"]
        combo_values = []
        for cat in cat_order:
            if cat in targets_by_cat:
                combo_values.append(f"━━ {lora_prep.TARGET_CATEGORIES.get(cat, cat)} ━━")
                combo_values.extend(targets_by_cat[cat])
        # Reste (catégories inattendues)
        for cat, keys in targets_by_cat.items():
            if cat not in cat_order:
                combo_values.append(f"━━ {lora_prep.TARGET_CATEGORIES.get(cat, cat)} ━━")
                combo_values.extend(keys)
        target_combo = ttk.Combobox(dlg, textvariable=target_var, state="readonly",
                                     values=combo_values, font=FONT_BODY, height=20)
        target_combo.pack(fill="x", padx=20, pady=(2, 4))
        target_info = Label(dlg, text="", font=FONT_SMALL, fg=TEXT_DIM, bg=BG,
                             justify="left", wraplength=580)
        target_info.pack(anchor="w", padx=20, pady=(0, 12))

        def update_target_info(*_):
            sel = target_var.get()
            # Refuse les separateurs "━━ ... ━━"
            if sel.startswith("━━"):
                target_var.set("sdxl_kohya")
                sel = "sdxl_kohya"
            cfg = lora_prep.TARGETS.get(sel, {})
            label = cfg.get("label", "?")
            res = cfg.get("resolutions", [])
            cap = cfg.get("captioner", "?")
            q_prefix = cfg.get("quality_prefix")
            note = ""
            if cap == "natural":
                note = "  ⭐ JoyCaption recommandé (relance l'analyse en mode 'joycaption' ou 'all')."
            if q_prefix:
                note += f"\n   🏷 Quality tags auto-ajoutés : « {q_prefix} »"
            target_info.config(
                text=(f"→ {label}\n"
                      f"   Résolution(s) : {res}   |   Captioner conseillé : {cap}\n"
                      f"   Doc : {cfg.get('trainer_doc_url', '')}{note}")
            )
        target_combo.bind("<<ComboboxSelected>>", update_target_info)
        update_target_info()

        # Viable only
        viable_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dlg, text="N'inclure que les photos viables (recommandé)",
                        variable=viable_var, font=FONT_BODY, fg=TEXT, bg=BG,
                        selectcolor=CARD, activebackground=BG,
                        activeforeground=TEXT).pack(anchor="w", padx=20, pady=4)

        # Masques sujet (Lot D)
        masks_var = tk.BooleanVar(value=False)
        tk.Checkbutton(dlg,
                        text="🎭 Générer aussi les masques sujet (OneTrainer masked training)",
                        variable=masks_var, font=FONT_BODY, fg=TEXT, bg=BG,
                        selectcolor=CARD, activebackground=BG,
                        activeforeground=TEXT).pack(anchor="w", padx=20, pady=4)
        Label(dlg, text="   (Recommandé pour OneTrainer — fond noir = ignoré pendant le training)",
              font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(anchor="w", padx=20)

        # Boutons
        btn_row = Frame(dlg, bg=BG)
        btn_row.pack(side="bottom", fill="x", padx=20, pady=15)

        def go():
            persona = persona_var.get().strip().replace(" ", "_")
            if not persona:
                messagebox.showwarning("Manquant", "Donne un trigger word.", parent=dlg)
                return
            target = target_var.get()
            viable_only = viable_var.get()
            with_masks = masks_var.get()
            dlg.destroy()
            self._launch_lora_prep(data, persona, target, viable_only, with_masks)

        Button(btn_row, text="Annuler", font=FONT_BODY, bg=CARD_HI, fg=TEXT,
               relief="flat", padx=20, pady=8, command=dlg.destroy).pack(side="right", padx=4)
        Button(btn_row, text="🚀 Préparer", font=FONT_BODY, bg=GREEN, fg=BG,
               relief="flat", padx=20, pady=8, command=go).pack(side="right", padx=4)

    def _launch_lora_prep(self, data, persona, target, viable_only, with_masks=False):
        import lora_prep
        cfg = lora_prep.TARGETS[target]

        # Dossier de sortie
        folder_prefix = "kohya" if "kohya" in target else target.split("_")[0]
        default_out = Path(self.analyzer_path.get()).parent / f"{folder_prefix}_{persona}"
        out = filedialog.askdirectory(
            initialdir=str(default_out.parent),
            title=f"Dossier où créer {folder_prefix}_{persona}/ (annule = défaut)"
        )
        out_folder = Path(out) / f"{folder_prefix}_{persona}" if out else default_out

        self.status_var.set(f"Préparation {cfg['label']} en cours...")
        self.analyzer_kohya_btn.config(state="disabled", text="🧬 Préparation...")

        def progress_cb(cur, tot, name):
            self.root.after(0, lambda c=cur, t=tot, n=name:
                self.status_var.set(f"Prep : {c}/{t} — {n}"))

        def worker():
            try:
                result = lora_prep.prepare_lora_folder(
                    analysis_data=data,
                    source_folder=self.analyzer_path.get(),
                    output_folder=out_folder,
                    persona_name=persona,
                    target=target,
                    viable_only=viable_only,
                    progress_cb=progress_cb,
                )
                # Si demande, genere aussi les masques sujet dans le dossier final
                if with_masks:
                    self.root.after(0, lambda: self.status_var.set("Génération masques sujet…"))
                    images_subfolder = lora_prep.TARGETS[target].get("folder_naming") == "kohya"
                    masks_target = out_folder / (f"10_{persona}" if images_subfolder else "images")
                    try:
                        # Driver inline
                        driver = (
                            "import sys, json\n"
                            f"sys.path.insert(0, r'{Path(__file__).parent}')\n"
                            "from mask_generator_local import generate_masks_for_folder\n"
                            "data = json.loads(sys.stdin.read())\n"
                            "r = generate_masks_for_folder(folder=data['folder'], threshold=0.5,\n"
                            "    progress_cb=lambda c, t, n: print(f'PROGRESS {c}/{t} {n}', file=sys.stderr, flush=True))\n"
                            "print(json.dumps(r))\n"
                        )
                        payload = json.dumps({"folder": str(masks_target)})
                        proc = subprocess.Popen(
                            [self.comfyui_py, "-c", driver],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                        )
                        stdout, _ = proc.communicate(input=payload, timeout=600)
                        mask_result = json.loads(stdout.strip()) if stdout.strip() else {}
                        result["masks_generated"] = mask_result.get("written", 0)
                    except Exception as e:
                        result["masks_error"] = str(e)[:200]
                self.root.after(0, lambda r=result: self._on_kohya_done(r, out_folder))
            except Exception as e:
                import traceback
                err = f"{e}\n\n{traceback.format_exc()[-500:]}"
                self.root.after(0, lambda e=err: messagebox.showerror("Erreur prep", e))
                self.root.after(0, lambda: self.analyzer_kohya_btn.config(
                    state="normal", text="🧬 Préparer LoRA (réessayer)"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_kohya_done(self, result, out_folder):
        written = result.get("written", 0)
        skipped = result.get("skipped", 0)
        errs = result.get("errors", [])
        tgt = result.get("target", "?")
        cfg_path = result.get("config_path")
        msg = (f"✅ {written} photo(s) préparée(s) (crop + captions)\n"
               f"⏭ {skipped} exclue(s) (non viables)\n\n"
               f"📁 {out_folder}\n"
               f"🎯 Target : {tgt}\n")
        if cfg_path:
            msg += f"⚙ Config : {Path(cfg_path).name}\n"
        masks_n = result.get("masks_generated")
        if masks_n is not None:
            msg += f"🎭 Masques sujet : {masks_n} générés\n"
        if result.get("masks_error"):
            msg += f"⚠️ Erreur masques : {result['masks_error'][:100]}\n"
        msg += "\nLis le README.txt pour les instructions de lancement."
        if errs:
            msg += f"\n\n⚠️ {len(errs)} erreur(s) :\n" + "\n".join(errs[:3])
        messagebox.showinfo("Dataset LoRA prêt", msg)
        self.status_var.set(f"Prep terminé : {written} photos dans {out_folder.name}/")
        self.analyzer_kohya_btn.config(
            state="normal", bg=GREEN, fg=BG,
            text=f"🧬 Préparer LoRA ({written} OK)"
        )
        try:
            os.startfile(str(out_folder))
        except Exception:
            pass
        # Ouvre le dossier dans l'explorer
        try:
            os.startfile(str(out_folder))
        except Exception:
            pass

    def _move_rejected_photos(self):
        """Deplace toutes les photos marquees lora_viable='no' dans un sous-dossier."""
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            return
        rejected = [img["path"] for img in data.get("images", [])
                    if img.get("lora_viable") == "no" and Path(img["path"]).exists()]
        if not rejected:
            messagebox.showinfo("Rien à faire", "Aucune photo marquée à virer.")
            return

        source_folder = Path(self.analyzer_path.get())
        reject_folder = source_folder / "_rejected"

        if not messagebox.askyesno(
            "Confirmer le déplacement",
            f"Déplacer {len(rejected)} photo(s) dans :\n{reject_folder}\n\n"
            f"Les fichiers seront DEPLACES (pas copiés). Continuer ?"
        ):
            return

        reject_folder.mkdir(exist_ok=True)
        moved = 0
        errors = []
        for src in rejected:
            try:
                src_p = Path(src)
                dst = reject_folder / src_p.name
                # Si conflit, ajoute un suffixe
                if dst.exists():
                    dst = reject_folder / f"{src_p.stem}_{moved}{src_p.suffix}"
                shutil.move(str(src_p), str(dst))
                moved += 1
            except Exception as e:
                errors.append(f"{Path(src).name}: {e}")

        msg = f"✅ {moved} photo(s) déplacée(s) dans :\n{reject_folder}"
        if errors:
            msg += f"\n\n⚠️ {len(errors)} erreurs :\n" + "\n".join(errors[:5])
        messagebox.showinfo("Déplacement terminé", msg)
        self.status_var.set(f"Déplacé : {moved} photo(s) vers _rejected/")
        self.analyzer_move_btn.config(state="disabled", bg=CARD_HI, fg=TEXT_DIM,
                                      text="🗑 Relance l'analyse")

        if "error" in data:
            self.analyzer_summary.config(text=f"❌ {data['error']}", fg=RED)
            return

        s = data.get("summary", {})
        recos = data.get("recommendations", [])
        body_coh = s.get('overall_body_coherence')
        dup_count = s.get('duplicates_count', 0)
        cap_count = s.get('captions_written', 0)
        ai_count = s.get('ai_classifier_count', 0)
        art_high = s.get('artifacts_high_count', 0)
        extra = ""
        if dup_count > 0:
            extra += f"  |  🔁 {dup_count} duplicates"
        if cap_count > 0:
            extra += f"  |  🏷 {cap_count} captions"
        if ai_count > 0:
            extra += f"  |  🤖 {ai_count} IA-detected"
        if art_high > 0:
            extra += f"  |  ❌ {art_high} artefacts sévères"
        summary_text = (
            f"📊 {s.get('total_images', 0)} images  |  "
            f"✅ {s.get('lora_viable', 0)} viables  |  "
            f"⚠️ {s.get('lora_borderline', 0)} borderline  |  "
            f"❌ {s.get('lora_unusable', 0)} a virer{extra}\n"
            f"Coherence visage : {s.get('overall_face_coherence', 'N/A')}  |  "
            f"Coherence corps : {body_coh if body_coh is not None else 'N/A'}\n\n"
            + "\n".join(recos)
        )
        self.analyzer_summary.config(text=summary_text, fg=TEXT)

        # Remplir le tableau
        for img in data.get("images", []):
            name = img.get("name", "?")
            res = f"{img.get('width', '?')}x{img.get('height', '?')}"
            sharp = f"{img.get('sharpness', 0):.0f}" if img.get("sharpness") is not None else "-"
            faces = str(img.get("face_count", "?"))
            prop = f"{img.get('face_proportion', 0):.1f}%" if img.get("face_proportion") else "-"
            yaw_val = img.get("face_yaw")
            yaw = f"{yaw_val:+.0f}" if yaw_val is not None else "-"
            expr = img.get("expression") or "-"
            sim_avg = img.get("face_similarity_avg")
            body_avg = img.get("body_similarity_avg")
            qual = img.get("quality_verdict", "-")
            view_type = img.get("view_type", "both")

            sim = f"{sim_avg:.2f}" if sim_avg is not None else "-"
            body = f"{body_avg:.2f}" if body_avg is not None else "NA"

            # === Tag = verdict viabilite LoRA en priorite ===
            viable = img.get("lora_viable", "yes")
            reason = img.get("lora_reason", "OK pour LoRA")

            if viable == "no":
                tag = "err"
                lora_text = f"❌  {reason}"
            elif viable == "borderline":
                tag = "warn"
                lora_text = f"⚠  {reason}"
            else:
                tag = "ok"
                lora_text = f"✓  {reason}"

            self.analyzer_tree.insert("", "end", text=f"  {name}",
                                       values=(res, sharp, faces, prop, yaw, expr, sim, body, qual, lora_text),
                                       tags=(tag,))

    # ============================================================
    # Config helpers
    # ============================================================
    def _pick_folder(self, var):
        d = filedialog.askdirectory(initialdir=var.get())
        if d:
            var.set(d.replace("/", "\\"))

    def _pick_file(self, var):
        cur = var.get()
        initdir = str(Path(cur).parent) if cur and Path(cur).parent.exists() else None
        f = filedialog.askopenfilename(
            initialdir=initdir,
            title="Choisir python.exe de ComfyUI",
            filetypes=[("Python executable", "python*.exe"), ("Tous", "*.*")],
        )
        if f:
            var.set(f.replace("/", "\\"))

    def _load_preview_image(self, path, max_size=200):
        """Charge une image en PhotoImage Tk en gardant le ratio. Retourne None si echec."""
        if not _HAS_PIL or not path or not Path(path).is_file():
            return None
        try:
            img = PILImage.open(path)
            img.thumbnail((max_size, max_size), PILImage.LANCZOS)
            return PILImageTk.PhotoImage(img)
        except Exception:
            return None

    def _update_preview_current(self, path):
        if not _HAS_PIL:
            return
        photo = self._load_preview_image(path, max_size=200)
        if photo is None:
            return
        # Garde une ref pour eviter le GC
        self._preview_current_photo = photo
        self.preview_current_label.config(image=photo, width=200, height=200)
        self.preview_current_name.config(text=Path(path).name)

    def _update_preview_ref(self, path):
        if not _HAS_PIL:
            return
        if not path:
            # Vide
            self.preview_ref_label.config(image="", width=200, height=200)
            self.preview_ref_name.config(text="(aucune)")
            self._preview_ref_photo = None
            return
        photo = self._load_preview_image(path, max_size=200)
        if photo is None:
            self.preview_ref_name.config(text=f"⚠️ {Path(path).name}")
            return
        self._preview_ref_photo = photo
        self.preview_ref_label.config(image=photo, width=200, height=200)
        self.preview_ref_name.config(text=Path(path).name)

    def _tree_item_to_path(self, item_id):
        """Retrouve le path d'une ligne du tableau via son nom + dataset folder."""
        name = self.analyzer_tree.item(item_id, "text").strip()
        data = getattr(self, "_last_analysis_data", None)
        if not data:
            return None
        for img in data.get("images", []):
            if img.get("name") == name:
                return img.get("path")
        return None

    def _on_tree_select(self, event=None):
        """Clic simple sur une ligne -> met a jour le preview courant + verdict."""
        sel = self.analyzer_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        path = self._tree_item_to_path(item_id)
        if not path:
            return
        # Met a jour le mini-verdict avec les infos completes de l'image
        data = self._last_analysis_data
        name = self.analyzer_tree.item(item_id, "text").strip()
        for img in data.get("images", []):
            if img.get("name") == name:
                mini = {
                    "name": img.get("name"),
                    "face_count": img.get("face_count"),
                    "face_proportion": img.get("face_proportion"),
                    "face_yaw": img.get("face_yaw"),
                    "sharpness": img.get("sharpness"),
                    "expression": img.get("expression"),
                    "ref_match": img.get("ref_match"),
                    "face_similarity_to_ref": img.get("face_similarity_to_ref"),
                    "quality_verdict": img.get("quality_verdict"),
                    "wd14_tags": (img.get("wd14_tags") or "")[:300],
                }
                self._update_preview_verdict(mini)
                break
        self._update_preview_current(path)

    def _on_tree_double_click(self, event=None):
        """Double-clic -> popup avec l'image en grand + tous les details."""
        sel = self.analyzer_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        path = self._tree_item_to_path(item_id)
        if not path:
            return
        self._show_image_popup(path)

    def _show_image_popup(self, image_path):
        """Affiche une popup avec l'image en grand + toutes les metadonnees."""
        if not _HAS_PIL or not Path(image_path).is_file():
            return
        data = self._last_analysis_data or {}
        name = Path(image_path).name
        img_data = next((i for i in data.get("images", []) if i.get("name") == name), {})

        popup = tk.Toplevel(self.root)
        popup.title(f"📷 {name}")
        popup.configure(bg=BG)
        popup.geometry("1200x800")

        # Image a gauche
        left = Frame(popup, bg=BG, padx=12, pady=12)
        left.pack(side="left", fill="both", expand=True)
        try:
            pil = PILImage.open(image_path)
            pil.thumbnail((900, 750), PILImage.LANCZOS)
            photo = PILImageTk.PhotoImage(pil)
            lbl = Label(left, image=photo, bg=BG)
            lbl.image = photo  # garde la ref
            lbl.pack()
        except Exception as e:
            Label(left, text=f"Erreur affichage : {e}", fg=RED, bg=BG).pack()

        # Details a droite
        right = Frame(popup, bg=CARD, padx=14, pady=14, width=350)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        Label(right, text=name, font=FONT_H1, fg=ACCENT, bg=CARD,
              wraplength=320, justify="left").pack(anchor="w", pady=(0, 8))

        def kv(label, val, color=TEXT):
            if val is None or val == "":
                return
            row = Frame(right, bg=CARD)
            row.pack(fill="x", pady=2)
            Label(row, text=f"{label} :", font=FONT_SMALL, fg=TEXT_DIM, bg=CARD,
                  width=18, anchor="w").pack(side="left")
            Label(row, text=str(val), font=FONT_BODY, fg=color, bg=CARD,
                  anchor="w", wraplength=200, justify="left").pack(side="left", fill="x", expand=True)

        viable = img_data.get("lora_viable", "?")
        v_color = {"yes": GREEN, "borderline": YELLOW, "no": RED}.get(viable, TEXT)
        kv("Viabilité LoRA", viable.upper(), v_color)
        kv("Raison", img_data.get("lora_reason"))
        kv("", "")
        kv("Resolution", f"{img_data.get('width')}x{img_data.get('height')}")
        kv("Nettete", img_data.get("sharpness"))
        kv("Brightness", img_data.get("brightness"))
        kv("Contraste", img_data.get("contrast"))
        kv("Qualité", img_data.get("quality_verdict"))
        kv("", "")
        kv("Visages", img_data.get("face_count"))
        kv("% du cadre", f"{img_data.get('face_proportion', 0):.1f}%" if img_data.get("face_proportion") else None)
        kv("Yaw (pose)", f"{img_data.get('face_yaw'):+.1f}°" if img_data.get("face_yaw") is not None else None)
        kv("Type de plan", img_data.get("view_type"))
        kv("Expression", img_data.get("expression"))
        kv("Sim visage moy", img_data.get("face_similarity_avg"))
        kv("Sim corps moy", img_data.get("body_similarity_avg"))
        rm = img_data.get("ref_match")
        rm_color = {"OK": GREEN, "douteux": YELLOW, "mauvaise personne": RED}.get(rm, TEXT)
        kv("Match référence", f"{rm} (sim {img_data.get('face_similarity_to_ref')})" if rm else None, rm_color)
        if img_data.get("duplicate_of"):
            kv("Duplicate de", img_data["duplicate_of"], RED)
        if img_data.get("upscale_candidate"):
            kv("Upscale possible", "✓ (SUPIR/UltraSharp)", YELLOW)
        # Lot B : detection IA + artefacts
        ai_s = img_data.get("ai_score")
        if ai_s is not None:
            ai_color = RED if ai_s > 0.7 else (YELLOW if ai_s > 0.4 else GREEN)
            kv("Score IA-generated", f"{ai_s:.2f} ({img_data.get('ai_label', '?')})", ai_color)
        art_s = img_data.get("artifacts_severity")
        if art_s and art_s != "none":
            art_color = {"high": RED, "medium": YELLOW, "low": TEXT}.get(art_s, TEXT)
            cats = img_data.get("artifacts_categories") or []
            kv("Artefacts IA", f"{art_s.upper()} ({', '.join(cats)})", art_color)
        md_src = img_data.get("ai_metadata_sources")
        if md_src:
            kv("Metadata IA", f"{', '.join(md_src)} ({img_data.get('ai_metadata_confidence', '?')})", TEXT_DIM)

        # ===== Captions EDITABLES (Lot D — éditeur inline) =====
        # Chaque caption a son propre Text + bouton Sauver qui réécrit le .txt
        for label, key, sidecar_ext, color in (
            ("🏷  WD14 tags",       "wd14_tags",       ".txt",     TEXT),
            ("🌿 Florence-2",       "natural_caption", ".nat.txt", TEXT_DIM),
            ("⭐ JoyCaption (2026)", "joycaption",      ".joy.txt", ACCENT),
        ):
            val = img_data.get(key)
            if not val:
                continue
            # Header avec label + bouton
            head = Frame(right, bg=CARD)
            head.pack(fill="x", pady=(10, 2))
            Label(head, text=label, font=FONT_SMALL, fg=color, bg=CARD,
                  anchor="w").pack(side="left")
            status_lbl = Label(head, text="", font=FONT_SMALL, fg=GREEN, bg=CARD)
            status_lbl.pack(side="right")

            # Zone editable
            t = Text(right, height=5, bg=BG, fg=TEXT, font=FONT_SMALL,
                     wrap="word", relief="flat", padx=6, pady=6,
                     insertbackground=TEXT)
            t.insert("1.0", val)
            t.pack(fill="x")

            # Bouton sauver
            def make_save_handler(text_widget=t, kk=key, ext=sidecar_ext,
                                   status_widget=status_lbl, img_name=name,
                                   img_path_str=str(image_path)):
                def save():
                    new_val = text_widget.get("1.0", "end").strip()
                    self._save_caption_edit(img_name, kk, ext, new_val, img_path_str)
                    status_widget.config(text="✅ sauvegardé", fg=GREEN)
                    # Reset le statut au bout de 2s
                    self.root.after(2000, lambda: status_widget.config(text=""))
                return save

            Button(right, text=f"💾 Sauver {ext}", font=FONT_SMALL,
                   bg=CARD_HI, fg=TEXT, relief="flat", padx=8, pady=4,
                   cursor="hand2",
                   command=make_save_handler()).pack(anchor="e", pady=(2, 6))

        Button(right, text="Fermer (Esc)", font=FONT_BODY, bg=CARD_HI, fg=TEXT,
               relief="flat", padx=12, pady=6, command=popup.destroy).pack(pady=12)
        popup.bind("<Escape>", lambda e: popup.destroy())

    def _save_caption_edit(self, image_name, caption_key, sidecar_ext, new_value, image_path_str):
        """Sauve une caption editee : ecrit le .txt + maj cache + maj memoire."""
        image_path = Path(image_path_str)
        if not image_path.exists():
            messagebox.showwarning("Erreur", f"Image introuvable : {image_name}")
            return

        # 1) Reecrit le sidecar .txt / .nat.txt / .joy.txt
        sidecar = image_path.with_suffix(sidecar_ext)
        try:
            sidecar.write_text(new_value, encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'écrire {sidecar.name} :\n{e}")
            return

        # 2) Met a jour la donnee en memoire (pour les autres affichages)
        data = self._last_analysis_data
        if data:
            for img in data.get("images", []):
                if img.get("name") == image_name:
                    img[caption_key] = new_value
                    # Si on edite joycaption, met aussi a jour natural_caption
                    # (le fallback Flux/Wan utilise natural_caption)
                    if caption_key == "joycaption":
                        img["natural_caption"] = new_value
                    break

        # 3) Met a jour le cache pour que le prochain run ne reanalyse pas
        try:
            dataset_folder = Path(self.analyzer_path.get())
            cache_path = dataset_folder / ".analyzer_cache.json"
            if cache_path.is_file():
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                # Trouve l'entree par nom d'image
                for key, entry_data in cache.get("entries", {}).items():
                    if entry_data.get("entry", {}).get("name") == image_name:
                        entry_data["entry"][caption_key] = new_value
                        if caption_key == "joycaption":
                            entry_data["entry"]["natural_caption"] = new_value
                        break
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        except Exception as e:
            print(f"Avertissement : MAJ cache impossible ({e})")

        self.status_var.set(f"Caption sauvegardée : {image_name} ({sidecar_ext})")

    def _update_preview_verdict(self, mini):
        """Affiche le mini-verdict de la derniere image analysee dans le panneau live."""
        if not _HAS_PIL:
            return
        lines = []
        # Visage
        fc = mini.get("face_count")
        if fc == 0:
            lines.append("👤 Aucun visage detecte")
        elif fc == 1:
            prop = mini.get("face_proportion")
            yaw = mini.get("face_yaw")
            ystr = f", yaw {yaw:+.0f}°" if yaw is not None else ""
            pstr = f" ({prop:.1f}% du cadre)" if prop else ""
            lines.append(f"👤 1 visage{pstr}{ystr}")
        elif fc and fc > 1:
            lines.append(f"⚠️ {fc} visages detectes")

        # Match a la reference
        sim_ref = mini.get("face_similarity_to_ref")
        rm = mini.get("ref_match")
        if rm is not None and sim_ref is not None:
            icon = {"OK": "✅", "douteux": "⚠️", "mauvaise personne": "❌"}.get(rm, "•")
            lines.append(f"{icon} Match ref : {rm} (sim {sim_ref:.2f})")

        # Qualite
        sharp = mini.get("sharpness")
        qv = mini.get("quality_verdict")
        if sharp is not None:
            sq = "✅" if sharp >= 200 else ("⚠️" if sharp >= 100 else "❌")
            lines.append(f"{sq} Nettete {sharp:.0f}  →  {qv}")

        # Expression
        expr = mini.get("expression")
        if expr:
            lines.append(f"😶 Expression : {expr}")

        # Tags WD14
        tags = mini.get("wd14_tags")
        if tags:
            lines.append(f"🏷  {tags}")

        # JoyCaption (preferred quand disponible)
        joy = mini.get("joycaption")
        if joy:
            lines.append(f"⭐ {joy}")

        # Detection IA + artefacts (Lot B)
        ai_score = mini.get("ai_score")
        if ai_score is not None:
            icon = "🤖" if ai_score > 0.5 else "📷"
            lines.append(f"{icon} IA-score : {ai_score:.2f} ({'généré IA' if ai_score > 0.5 else 'photo'})")
        art_sev = mini.get("artifacts_severity")
        if art_sev and art_sev != "none":
            cats = mini.get("artifacts_categories") or []
            icon = {"high": "❌", "medium": "⚠️", "low": "💡"}.get(art_sev, "•")
            lines.append(f"{icon} Artefacts ({art_sev}) : {', '.join(cats)}")
        md_sources = mini.get("ai_metadata_sources") or []
        if md_sources:
            lines.append(f"📋 Metadata IA : {', '.join(md_sources)}")

        self.preview_verdict_text.config(text="\n".join(lines))

    def _pick_ref_image(self):
        # Demarre dans le dossier dataset si possible
        cur = self.analyzer_ref_path.get().strip()
        initdir = str(Path(cur).parent) if cur and Path(cur).exists() else self.analyzer_path.get()
        f = filedialog.askopenfilename(
            initialdir=initdir,
            title="Choisir une photo de reference (un visage clair)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("Tous", "*.*")]
        )
        if f:
            self.analyzer_ref_path.set(f.replace("/", "\\"))
            self._update_preview_ref(f)

    def _save_cfg(self):
        for k, v in self.entries.items():
            self.cfg[k] = v.get()
        save_config(self.cfg)
        # Applique a chaud le nouveau chemin Python ComfyUI
        self.comfyui_py = self.cfg.get("comfyui_python") or COMFYUI_FUTURE_PY
        # Met a jour le dossier datasets par defaut dans l'analyseur (si vide)
        if hasattr(self, "analyzer_path") and self.cfg.get("datasets_dir"):
            if not self.analyzer_path.get().strip():
                self.analyzer_path.set(self.cfg["datasets_dir"])
        messagebox.showinfo("Sauvegarde", "Config enregistrée.")

    def _reset_cfg(self):
        for k, v in DEFAULT_CONFIG.items():
            if k in self.entries:
                self.entries[k].set(v)


if __name__ == "__main__":
    cfg = load_config()
    # Pre-cree le dossier datasets par defaut s'il n'existe pas
    try:
        d = cfg.get("datasets_dir")
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    root = Tk()
    app = App(root)
    # Argument --tab <name> pour ouvrir directement sur un onglet
    if len(sys.argv) >= 3 and sys.argv[1] == "--tab":
        app.select_tab(sys.argv[2])
    root.mainloop()
