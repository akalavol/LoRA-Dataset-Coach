# Analyseur de dataset LoRA — Documentation complète

Application Tkinter ciblée avec **3 onglets** — 📊 Analyseur dataset, 📊 Évaluer LoRA, ⚙ Config — qui prend un dossier d'images et te dit, en quelques secondes par image, **si ton dataset est prêt à entraîner un LoRA persona** — et te prépare directement le dossier d'entraînement pour le trainer de ton choix parmi **19 cibles** (Kohya / Flux / Wan / Hunyuan / LTX / CogVideoX / Mochi / Open-Sora / ai-toolkit / OneTrainer / etc.).

**Note actuelle vs état de l'art 2026 : A (9.5/10)** après Lots A + B + C + D + E.

> 🌍 English version : [DOCUMENTATION.md](DOCUMENTATION.md)
> 📦 GitHub : https://github.com/akalavol/LoRA-Dataset-Coach

---

## 1. Vue d'ensemble

```
[dossier dataset]
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  ANALYSE PAR IMAGE                                            │
│  • détection visage (insightface antelopev2)                  │
│  • embedding visage → cosine sim contre photo de référence    │
│  • orientation (yaw) via keypoints                            │
│  • qualité (nettete Laplacien, brightness, contraste, MP)    │
│  • expression via CLIP zero-shot                              │
│  • plan (face_only / both / body_only) via face_proportion    │
│  • pHash perceptuel (détection duplicates)                    │
│  • captioning : WD14 / Florence-2 / JoyCaption / tous         │
│  • détection IA-generated (sdxl-detector ViT)                 │
│  • détection artefacts (mains/yeux/membres via WD14+caption)  │
│  • lecture metadata IA (C2PA / EXIF / PNG text / filename)    │
│  • cache (.analyzer_cache.json) — skip si déjà analysée       │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  ANALYSE GLOBALE                                              │
│  • matrice cosine sim (cohérence interne visage / corps)      │
│  • détection duplicates (pHash hamming < 5 OU face sim > 0.96)│
│  • clustering CLIP Union-Find (diversité visuelle)            │
│  • distribution angles / plans / expressions / aspect-ratios  │
│  • analyse fréquence tags WD14 → alertes overfit par catégorie│
│  • verdict viability par image (yes / borderline / no)        │
│  • verdict global A / B / C / D / F + plan d'action concret   │
│  • scores par famille de target (SDXL/Flux/Wan/Vidéo)         │
│  • suggestions de génération ciblée                           │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  ACTIONS                                                      │
│  🗑 Déplacer ratés       → _rejected/                         │
│  🔧 Floues → upscale     → _a_upscaler/ + README SUPIR        │
│  🧬 Préparer LoRA        → 19 targets disponibles             │
│  📄 Export PDF           → rapport paysage A4                 │
│  🖱 Double-clic ligne    → popup détaillée                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Modules

| Fichier | Rôle | Lot |
|---------|------|-----|
| `manager.py` | GUI Tkinter (onglet Analyseur) | base |
| `analyze_dataset.py` | Moteur d'analyse (subprocess Python ComfyUI-future) | base |
| `wd14_local.py` | Tagger WD14-MOAT standalone (ONNX) | base |
| `florence_local.py` | Captioner Florence-2 (deprecated 2026 pour persona) | base |
| `joycaption_local.py` | **Captioner JoyCaption Beta One** (standard 2026) | **A** |
| `ai_detector_local.py` | **Détection IA-generated** (Organika/sdxl-detector ViT) | **B** |
| `artifact_detector_local.py` | **Détection artefacts** (mains/yeux/membres via WD14+caption) | **B** |
| `metadata_ai.py` | **Lecture metadata IA** (C2PA / EXIF / PNG text / filename) | **B** |
| `lora_prep.py` | Préparation multi-target (19 targets, crops + configs) | base+ |
| `mask_generator_local.py` | **Masques sujet BriaRMBG-1.4** pour OneTrainer masked training | **D** |
| `export_pdf.py` | Génération PDF paysage A4 | base |

Python d'exécution :
- **GUI** : Python système (Python 3.11 ou 3.12 — celui qui a `tkinter` + `PIL/ImageTk`)
- **Analyse** : `C:\AI\ComfyUI-future\ComfyUI_windows_portable\python_embeded\python.exe` (celui qui a `insightface`, `onnxruntime`, `transformers`, `torch` CUDA)

---

## 3. L'interface, ligne par ligne

### 3.1 Sélection du dataset

```
📂 Dossier dataset : [ C:\AI\datasets\ma_persona ] [📂] [🔍 Analyser]
                                                                [📄 Export PDF]
                                                                [🗑 Déplacer ratés (N)]
                                                                [🔧 Floues → upscale (N)]
                                                                [🧬 Préparer LoRA (N)]
                                                                [🎭 Masques sujet (N)]   ← Lot D
```

Les 4 derniers boutons sont **désactivés** tant qu'une analyse n'a pas tourné. Après analyse, leur libellé indique le nombre d'éléments concernés.

### 3.2 Photo de référence (optionnel)

```
📷 Photo de référence : [ C:\AI\datasets\ref.png ] [📂] [✕]   (optionnel)
```

Si fournie, l'analyseur calcule l'embedding de cette photo une fois, puis pour chaque image du dataset il calcule `cos_sim(emb_image, emb_ref)`. Trois verdicts :

| Score | Verdict | Effet sur la viabilité |
|-------|---------|------------------------|
| `≥ 0.50` | ✅ OK | aucun |
| `0.35 - 0.50` | ⚠ douteux | passe en `borderline` si `yes` |
| `< 0.35` | ❌ mauvaise personne | passe en `no` |

**Pourquoi c'est crucial** : sans référence, l'analyseur ne fait que de la **cohérence interne**. Si InstantID a généré 20 photos de quelqu'un d'autre qui se ressemblent entre elles, l'analyseur passera à côté. Avec une référence, chaque photo est comparée à la **vérité**.

### 3.3 Captioner (4 radio buttons — Lot A)

```
🏷 Captions : ( ) WD14 tags    ( ) Florence-2    ( ) JoyCaption ⭐    ( ) Tous
              (WD14 pour SDXL · JoyCaption pour Flux/Wan)
```

| Mode | Tagger | Sortie | Pour | Coût modèle |
|------|--------|--------|------|-------------|
| `wd14` | WD14-MOAT (ONNX) | `image.txt` (booru) | SDXL / Kohya / OneTrainer | 330 Mo |
| `natural` | Florence-2 | `image.nat.txt` | **Fallback uniquement** (hallucine sur personnes) | 540 Mo |
| `joycaption` | **JoyCaption Beta One** (LLaVA fine-tuné) | `image.joy.txt` | **Flux / Wan / Hunyuan / SD 3.5** | 4 Go INT4 / 8 Go BF16 |
| `all` | Les 3 | les 3 fichiers | Si tu hésites encore sur le target | tous |

**JoyCaption Beta One est le standard communautaire 2025-2026** pour LoRA persona. Florence-2 est gardé en fallback parce qu'il marche sur CPU sans CUDA, mais le consensus est qu'**il hallucine emotions/vêtements/contextes** sur les personnes.

### 3.4 Progression

```
⚙ Chargement insightface (~5-10s)…    ← phase courante
[████████████░░░░░░░░░░░░] 53%   ⏱ ~12s restant
📷 16/30 — IMG_1234.png                ← fichier en cours
```

Phases typiques (ordre de chargement) :
1. `Chargement insightface`
2. `Chargement CLIP pour analyse corps + expressions`
3. `Chargement sdxl-detector (détection images IA)` (Lot B)
4. `Chargement WD14 tagger` (si mode ∈ {wd14, both, all})
5. `Chargement Florence-2` (si mode ∈ {natural, both, all})
6. `Chargement JoyCaption Beta One` (si mode ∈ {joycaption, all})
7. `Analyse en cours — N images`
8. `Cache : X réutilisées, Y nouvellement analysées`
9. `Calcul matrice de similarité`
10. `Detection des duplicates`
11. `Cache mis a jour`

### 3.5 Preview live (3 zones)

```
┌──────────────────┐ ┌──────────────────┐ ┌────────────────────────┐
│ 📷 En cours      │ │ 📌 Référence     │ │ 🔎 Dernier verdict     │
│  [image 200x200] │ │  [image 200x200] │ │ 👤 1 visage (12.3%)    │
│                  │ │                  │ │ ✅ Match ref : OK (0.78)│
│ IMG_0042.png     │ │ ref.png          │ │ ✅ Nettete 312 → OK     │
└──────────────────┘ └──────────────────┘ │ 😶 slight smile         │
                                          │ 🏷 brown hair, indoors  │
                                          │ ⭐ A young woman with…  │
                                          │ 🤖 IA-score : 0.97      │
                                          │ ❌ Artefacts (high) :   │
                                          │    mains, yeux          │
                                          │ 📋 Metadata IA : exif   │
                                          └────────────────────────┘
```

### 3.6 Bloc verdict global (enrichi Lots A + B + C)

```
┌──────────────────────────────────────────────────────────────┐
│  A    VERDICT DATASET : EXCELLENT                            │
│       28 photos viables → 28 après cleanup (cible 20-30)     │
│                                                              │
│  📷 Réf : ref.png → ✅ 28 photos correspondent (avg 0.74)    │
│                                                              │
│  ⚠️ Risques d'overfit :                  (Lot A)            │
│    • Vetement : « white shirt » (87%), « jeans » (60%)      │
│    • Fond     : « indoors » (90%)                            │
│                                                              │
│  🎯 Scores par famille de target :       (Lot C)            │
│    B    72/100   SDXL classique                              │
│    B+   78/100   SDXL anime                                  │
│    A    88/100   Flux                                        │
│    C    55/100   Wan vidéo                                   │
│    C    50/100   Vidéo (Hunyuan/Mochi/...)                   │
│                                                              │
│  🌈 Diversité : 67/100  (Diversité correcte)  · 22 clusters  │
│  📐 AR : ⬜ 26 carré · 📱 2 portrait · 🖼 0 paysage          │
│                                                              │
│  🎯 Plan d'action :                                          │
│    • Genere 2-3 photos de profil (yaw 60-80°)                │
│    • Varie les vêtements (white shirt à 87%)                 │
└──────────────────────────────────────────────────────────────┘
```

Échelle de notes globales :
| Grade | Condition | Couleur |
|-------|-----------|---------|
| `A` | ≥ 30 viables + cohérence > 0.6 | vert |
| `B+` | 20–29 viables | vert |
| `B` | défaut | vert |
| `B-` | 15–19 viables | jaune |
| `C` | 10–14 viables | jaune |
| `D` | 1–9 viables | rouge |
| `F` | 0 viable | rouge |

### 3.7 Tableau détaillé

Une ligne par image, 10 colonnes : **Image · Resol · Net · Vis · %vis · Pose° · Expression · S.vis · S.cor · Qualite · 🧬 LoRA - raison**

Couleur de la ligne :
- 🟢 `ok` : viable
- 🟡 `warn` : borderline
- 🔴 `err` : à virer

**Clic simple** : met à jour le preview courant + dernier verdict.

**Double-clic** : ouvre une popup 1200×800 avec l'image en grand + **toutes les métadonnées** dans le panneau droit, dont :
- Viabilité (vert/jaune/rouge) + raison
- Résolution, qualité, % cadre, yaw, expression
- Match référence (couleur selon sim)
- Duplicate de / Upscale possible
- **Score IA-generated** (rouge > 0.7, jaune > 0.4, vert sinon) [Lot B]
- **Artefacts IA** (severity + catégories) [Lot B]
- **Metadata IA** (sources + confidence) [Lot B]
- Les **3 captions côte à côte ÉDITABLES** (WD14 / Florence / **JoyCaption** ⭐) — bouton **💾 Sauver** sous chaque caption qui réécrit le sidecar `.txt` / `.nat.txt` / `.joy.txt` + met à jour le cache + la mémoire (Lots A + **D**)

### 3.8 Résumé final

```
📊 30 images | ✅ 28 viables | ⚠️ 2 borderline | ❌ 0 a virer
| 🔁 0 duplicates | 🏷 30 captions | 🤖 30 IA-detected | ❌ 0 artefacts sévères
Coherence visage : 0.78 | Coherence corps : 0.81

🎯 Meilleur target pour ce dataset : « Flux » (A, 88/100)
   — captions JoyCaption présentes, ratios variés, diversité OK
✅ Identite verifiee (vs ref) : 28 photos correspondent (avg 0.74)
🌈 Diversité : 67/100 — Diversité correcte
📐 AR : ✅ Mix sain de ratios → idéal Flux/Wan multi-bucket
💡 30 captions Kohya generee(s)
🎯 À générer ensuite pour compléter le dataset :
   → Genere 2-3 photos de profil (yaw 60-80°)
```

---

## 4. Actions

### 4.1 🗑 Déplacer ratés
- Identifie toutes les images marquées `lora_viable == "no"`
- Crée `<dataset>/_rejected/`
- Déplace les fichiers **avec leur caption** `.txt` / `.nat.txt` / `.joy.txt`

### 4.2 🔧 Floues → upscale
- Identifie `sharpness ∈ [50, 100]` + visage détecté
- Crée `<dataset>/_a_upscaler/` + README.txt SUPIR/UltraSharp
- Récupère aussi les captions

### 4.3 🎭 Masques sujet (Lot D)

Génère un fichier `<image>-masklabel.png` à côté de chaque photo viable. C'est le format **OneTrainer masked training** : la loss du LoRA est focalisée sur le sujet (blanc dans le masque), le fond (noir) est ignoré. Gain qualité significatif quand les arrière-plans varient beaucoup ou polluent l'apprentissage.

Popup de config :
- **Binariser** (recommandé OneTrainer, défaut activé) : seuil 0.5 — sujet pur blanc, fond pur noir, plus de gradients alpha
- **Seuil** (0.1-0.9, slider) : ajuste si le masque coupe mal (essaie 0.4 si le sujet est sous-segmenté, 0.6 si le fond bave)
- **Viable only** : ne traite que les photos `yes` + `borderline`

Modèle utilisé : **BriaRMBG-1.4** (~176 Mo téléchargé au 1er run, CUDA 1 s/image, CPU 8 s/image). Réutilise le custom node `ComfyUI-BRIA_AI-RMBG` si déjà installé.

Après lancement OneTrainer :
- Charge le dataset
- Concepts → Image augmentations → coche **"masked training"**
- Train

### 4.4 🧬 Préparer LoRA (19 targets — multi-format)

Popup avec :
- Trigger word (`persona_name`)
- **Dropdown groupé par catégorie** :
  ```
  ━━ 📸 Photo réaliste ━━     ━━ 🎨 Anime/Style ━━     ━━ 🎬 Vidéo ━━
  sdxl_kohya                    pony_kohya               wan21_musubi
  sd15_kohya                    illustrious_kohya        wan22_musubi
  sd35_kohya                    noobai_kohya             hunyuan_diffpipe
  hunyuan_dit_kohya                                       ltx_video_diffpipe
  sana_diffpipe                                           cogvideox_diffpipe
  flux_aitoolkit                                          mochi_diffpipe
  flux_kohya                                              open_sora_diffpipe
  chroma_aitoolkit
  onetrainer_sdxl
  ```
- Checkbox "viable only" (recommandé)
- Info live affiche : label, résolutions, captioner conseillé, lien doc, **quality tags auto-ajoutés** (Pony/Illustrious/NoobAI)

### 4.5 📄 Export PDF
- Format A4 paysage, thème Catppuccin
- En-tête : cartouche verdict (note A-F géante + plan d'action)
- Bloc référence si présente
- Statistiques globales, recommandations, tableau de toutes les images

---

## 5. Cache d'analyse

Fichier : `<dataset>/.analyzer_cache.json`

```json
{
  "version": 2,
  "entries": {
    "IMG_0042.png|123456|1748685432": {
      "entry": { ... tout le résultat de l'analyse ... },
      "face_emb": [512 floats],
      "body_emb": [512 floats],
      "phash": "1234567890123456789"
    }
  }
}
```

**Clé** = `nom | taille_octets | mtime` → invalidation automatique si tu remplaces un fichier.

**Comportement** :
1. Cache hit → tout est récupéré, **aucune** détection ré-exécutée (insightface/CLIP/WD14/JoyCaption/sdxl-detector)
2. Cache miss → analyse complète + ajout au cache
3. Si la **référence change**, `face_similarity_to_ref` et `ref_match` sont **recalculés** depuis l'embedding caché
4. Matrices de sim, duplicates, **clustering CLIP**, **distribution AR**, **scores par target** sont **toujours recalculés** sur l'ensemble

**Conséquence pratique** : tu ajoutes 5 photos → seulement les 5 nouvelles sont analysées (~30 s), le reste est instantané. Très utile pour les cycles d'itération InstantID.

**Pour reset le cache** : supprime `.analyzer_cache.json`.

---

## 6. Fichiers générés à côté des images

| Fichier | Quand | Contenu |
|---------|-------|---------|
| `image.txt` | captioner ∈ {wd14, both, all} | tags booru WD14 (format Kohya) |
| `image.nat.txt` | captioner ∈ {natural, both, all} | Florence-2 (fallback CPU) |
| `image.joy.txt` | captioner ∈ {joycaption, all} | **JoyCaption Beta One** (standard 2026) |

Au moment du **🧬 Préparer LoRA**, le caption final est reconstruit selon le target :
- `wd14` → reprend `image.txt` ou `wd14_tags` du résultat
- `natural` → **priorité JoyCaption** > Florence-2 > reconstruction depuis WD14
- Quality prefix injecté avant le trigger word pour SDXL forks (Pony/Illustrious/NoobAI)

---

## 7. Catalogue des 19 targets supportés

### 📸 Photo réaliste (9 targets)

| Clé | Trainer | Résolution(s) | Crop | Captioner |
|-----|---------|---------------|------|-----------|
| `sdxl_kohya` | Kohya SS GUI | 1024² | carré visage | WD14 |
| `sd15_kohya` | Kohya SS GUI | 512² | carré visage | WD14 |
| `sd35_kohya` | Kohya branche sd3 | 1024² | carré visage | naturel |
| `hunyuan_dit_kohya` | Tencent HunyuanDiT | 1024² | carré visage | naturel |
| `sana_diffpipe` | diffusion-pipe (NVIDIA Sana) | 1024² + 512×1024 + 1024×512 | buckets | naturel |
| `flux_aitoolkit` | ai-toolkit (ostris) | 1024² + 1024×768 + 768×1024 | buckets | naturel |
| `flux_kohya` | Kohya branche sd3 | 1024² | carré visage | naturel |
| `chroma_aitoolkit` | ai-toolkit (Chroma = Flux variant uncensored) | 1024² + 1024×768 + 768×1024 | buckets | naturel |
| `onetrainer_sdxl` | OneTrainer | 1024² | carré visage | WD14 |

### 🎨 Anime/Style — SDXL forks (3 targets)

Tous basent sur SDXL mais ont des **quality tags obligatoires** injectés automatiquement en tête de caption :

| Clé | Quality prefix automatique |
|-----|--------------------------|
| `pony_kohya` | `score_9, score_8_up, score_7_up, source_photo` |
| `illustrious_kohya` | `masterpiece, best quality, very aesthetic, absurdres` |
| `noobai_kohya` | `masterpiece, best quality, newest, absurdres, highres` |

### 🎬 Vidéo (7 targets)

| Clé | Trainer | Résolution(s) | Spécificités |
|-----|---------|---------------|-------------|
| `wan21_musubi` | musubi-tuner | 832×480 + 480×832 + 720² | I2V/T2V, .bat lancement généré |
| `wan22_musubi` | musubi-tuner | 832×480 + 480×832 + 720² | Wan 2.2, .bat lancement généré |
| `hunyuan_diffpipe` | diffusion-pipe | 832×480 + 480×832 + 720² | Hunyuan video |
| `ltx_video_diffpipe` | diffusion-pipe | 768×512 + 512×768 + 704² | LTX-Video temps réel, frame_buckets |
| `cogvideox_diffpipe` | cogvideox-factory | 720×480 + 480×720 | CogVideoX 5B, network_dim 64, lr 1e-3 |
| `mochi_diffpipe` | diffusion-pipe | 848×480 + 480×848 | Mochi 1 (Genmo) |
| `open_sora_diffpipe` | diffusion-pipe | 720² + 1280×720 + 720×1280 | Open-Sora 2.0 (HPC-AI) |

### Stratégies de crop

- **`square_face`** : crop carré centré sur le visage, marge configurable (60 % par défaut = headshot avec épaules), resize à la résolution cible.
- **`bucket_face`** : choisit le bucket dont le ratio est le plus proche de l'image source, puis crop au bon ratio centré sur le visage. **Garde plus de contexte** qu'un crop carré. Recommandé pour Flux / Wan / Hunyuan / LTX qui supportent les ratios variés.

### Structure générée

**Kohya** (`folder_naming = "kohya"`) :
```
kohya_persona/
  10_persona/
    persona_001.png   (1024×1024)
    persona_001.txt   ("persona, tag1, tag2..." ou avec quality_prefix pour Pony/Illustrious/NoobAI)
    ...
  kohya_config.toml         (ou kohya_sd35_config.toml / kohya_hunyuan_dit_config.toml / kohya_flux_config.toml)
  README.txt
```

**Flat** (`folder_naming = "flat"` : Flux ai-toolkit / Sana / Wan musubi / Hunyuan / LTX / CogVideoX / Mochi / Open-Sora / OneTrainer) :
```
flux_persona/
  images/
    persona_001.png
    persona_001.txt
    ...
  ai_toolkit_config.yaml   (ou musubi_dataset.toml + launch_musubi.bat,
                            ou diffusion_pipe_*.toml,
                            ou cogvideox_factory_config.yaml,
                            ou rien si OneTrainer)
  README.txt
```

---

## 8. Détection IA + artefacts + metadata (Lot B)

### sdxl-detector — `ai_detector_local.py`
- Modèle `Organika/sdxl-detector` (ViT, **99.6 % accuracy**)
- ~350 Mo téléchargés au 1er run
- Sortie par image : `ai_score` (0-1) + `is_ai_classifier` (bool) + `ai_label` ("artificial" / "human")
- Stats summary : `ai_classifier_count`, `ai_score_avg`

### Artefacts anatomiques — `artifact_detector_local.py`
Approche **HADM-light** sans Detectron2 lourd :
- **Parse les tags WD14** déjà calculés pour les keywords d'artefacts :
  - `mains` (severity high) : bad hands, extra fingers, six fingers, mutated hands, fused fingers
  - `yeux` (severity high) : asymmetric eyes, deformed eyes, crossed eyes, lazy eye
  - `membres` (severity medium) : extra arms/legs, missing limbs
  - `anatomie` (severity medium) : bad anatomy, disfigured, deformed, mutation
  - `qualite` (severity low) : lowres, jpeg artifacts
- **Cross-check caption naturelle** (Florence-2 / JoyCaption) avec regex : "deformed hand", "asymmetric eyes", etc.
- **Impact sur viability** :
  - `high` → `lora_viable = "no"` (les mains/yeux pourris polluent le LoRA)
  - `medium` → bascule en `borderline`

HADM full (Detectron2 + Distortion-5K, paper arXiv 2411.13842) reste documenté pour les usages avancés, mais **le détecteur via tags couvre 80 % des cas en pratique** : si WD14 voit l'artefact, on le voit aussi.

### Metadata IA — `metadata_ai.py`
4 sources combinées :
1. **C2PA v2.2** : lit le manifest signé si présent (optionnel, nécessite `c2pa-python`)
2. **EXIF Software field** : matche "Stable Diffusion", "Midjourney", "DALL-E", "ComfyUI", "Fooocus", "InvokeAI", "Leonardo.ai", "Playground", "Ideogram", "Krea"
3. **PNG tEXt chunks** : parameters / prompt / workflow / comment (signatures ComfyUI / A1111)
4. **Filename heuristics** : `ComfyUI_00042_.png`, `MJ_xxx.png`, `_flux_`, `instantid_`, `_lora_`, `dalle_`, etc.

Sortie : `ai_metadata_sources` + `ai_metadata_confidence` (high si C2PA/EXIF, medium si filename seul).

### Recommandations auto (résumé final)
- `❌ N photo(s) avec artefacts IA severes (mains/yeux pourris) — a virer absolument`
- `⚠️ N photo(s) avec artefacts moyens — a verifier`
- `💡 30/30 (100%) photos detectees IA-generated — normal si dataset InstantID/Flux`
- `⚠️ N photo(s) detectee(s) IA-generated dans un dataset suppose reel — verifie via double-clic`
- `📋 N photo(s) avec metadata IA explicite (C2PA/EXIF/filename)`

---

## 9. Diversité + scores par target + AR (Lot C)

### Distribution aspect-ratio
5 buckets pour le verdict :
| Bucket | Ratio | Compte pour |
|--------|-------|-------------|
| `square` | 0.9–1.1 | SDXL / SD1.5 / Kohya idéal |
| `portrait` | 0.5–0.9 | Flux portrait / Wan portrait |
| `tall_portrait` | < 0.5 | **⚠️ Casse bucketing Flux/Wan** |
| `landscape` | 1.1–2.0 | Flux paysage / Wan paysage |
| `wide_landscape` | > 2.0 | **⚠️ Casse bucketing** |

Recommandations selon le mix :
- 95 %+ carré → "idéal SDXL, multi-bucket Flux sous-utilisé"
- Mix sain (sq + po + la présents) → "idéal Flux/Wan multi-bucket"
- Dominante portrait → "Wan I2V/T2V portrait, Flux portrait OK"

### Diversité CLIP + tags
**Clustering Union-Find** sur les `body_embeddings` CLIP :
- Seuil de fusion : `sim > 0.92` → même cluster
- **Nombre de clusters** = mesure de variance visuelle
- Détection des **gros clusters** (≥ 3 photos quasi-identiques) → reportés par nom

**Score global** = moyenne pondérée :
- `clip_score` (ratio clusters/total × 100)
- **Pénalité -15 par catégorie d'overfit** détectée (vetement/fond/lumiere/accessoire/couleur_fond)

Verdicts :
| Score | Verdict |
|-------|---------|
| 75+ | Excellente diversité |
| 55–74 | Diversité correcte |
| 35–54 | Diversité limitée |
| < 35 | Dataset trop homogène |

### Scores par famille de target

**5 familles** au lieu d'une note unique. Chacune avec ses critères et son grade A+/A/B+/B/C/D/F.

| Famille | Critères pondérés |
|---------|------------------|
| **SDXL classique** | Vol 25 / Res 1024+ 20 / WD14 15 / Diversité 20 / AR carré 10 / Pas d'artefacts 10 |
| **SDXL anime** (Pony/Illustrious/NoobAI) | Idem mais **+8 sur artefacts** (style anime + tolérant) |
| **Flux** | Vol 25 / Res 1024+ 20 / **Captions naturelles 20** (JoyCaption préféré) / **AR variété 15** / Diversité 15 / Artefacts 5 |
| **Wan vidéo** | Vol 25 / Res 512+ 15 / **Captions ≥ 100 char 20** (T5 adore les longues) / AR variété 15 / **Mix po+la 10** / Diversité 15 |
| **Vidéo (Hunyuan/Mochi/LTX/CogVideoX)** | Idem Wan -5 (clips MP4 recommandés en plus) |

Recommandation auto en tête : `🎯 Meilleur target pour ce dataset : « Flux » (A, 88/100) — raisons concises`.

---

## 10. Analyse fréquence tags + overfit (Lot A)

Catégorise chaque tag WD14 fréquent dans 5 catégories d'**attributs accessoires** (= ce qu'on **ne veut pas** que le LoRA mémorise comme la persona) :

| Catégorie | Patterns détectés |
|-----------|-------------------|
| `vetement` | shirt, dress, jacket, coat, blouse, sweater, hoodie, t-shirt, top, skirt, pants, jeans, shorts, bra, bikini, uniform |
| `fond` | indoors, outdoors, background, wall, room, kitchen, bedroom, studio, street, park, beach, forest, sky, garden, office, cafe |
| `lumiere` | lighting, sunlight, backlight, shadow, dim, bright, dark, neon, golden hour, natural light |
| `accessoire` | earrings, necklace, glasses, sunglasses, hat, scarf, watch, ring, bracelet, tie, bag, headphones, mask |
| `couleur_fond` | white background, black background, grey background, blue background, plain background, simple background |

**Seuils** :
- ≥ 75 % des photos viables → **ALERTE rouge** (overfit quasi-garanti)
- 50–75 % → **ATTENTION jaune**

**Tags d'identité ignorés intentionnellement** : `woman`, `long hair`, `brown hair` — c'est ce qu'on **veut** que le LoRA apprenne.

Affichage dans le bloc verdict :
```
⚠️ Risques d'overfit :
  • Vetement : « white shirt » (87%), « jeans » (60%)
  • Fond     : « indoors » (90%), « plain background » (73%)
```

---

## 11. Format du résultat JSON (stdout de `analyze_dataset.py`)

```json
{
  "summary": {
    "total_images": 30,
    "with_face": 28,
    "no_face": 2,
    "multiple_faces": 0,
    "overall_face_coherence": 0.78,
    "overall_body_coherence": 0.81,
    "resolution_min": "832x1216",
    "resolution_max": "1024x1024",
    "lora_viable": 28, "lora_borderline": 2, "lora_unusable": 0,
    "duplicates_count": 0, "duplicates_groups": [],
    "captions_written": 30,

    "ai_classifier_count": 30, "ai_metadata_count": 0,
    "ai_score_avg": 0.94,
    "artifacts_high_count": 0, "artifacts_medium_count": 1,

    "expressions": {"slight smile": 14, "neutral expression": 10},
    "verdict": {
      "grade": "A", "grade_desc": "EXCELLENT",
      "actions": ["Dataset suffisant : tu peux lancer le training Kohya SS"],
      "viable_now": 28, "after_cleanup": 28,
      "target_min": 20, "target_ideal": 30
    },
    "reference": {"name": "ref.png", "path": "...", "face_count_in_ref": 1},
    "reference_match": {"avg": 0.74, "min": 0.62, "max": 0.85, "ok": 28, "doubt": 0, "wrong": 0},
    "distribution": {
      "angles": {"front": 22, "three_quarter": 6, "profil": 0},
      "plans":  {"face_only": 18, "both": 10, "body_only": 0},
      "expressions": {"slight smile": 14, "neutral expression": 10}
    },
    "next_to_generate": ["Genere 2-3 photos de profil (yaw 60-80°)"],
    "blurry_recoverable": [],

    "tag_frequency_top": [["brown hair", 28, 1.0], ["long hair", 27, 0.96], ...],
    "tag_overfit_alerts": [
      {"tag": "white shirt", "category": "vetement", "count": 26, "ratio": 0.93, "severity": "ALERTE"}
    ],

    "aspect_ratio_distribution": {"square": 26, "portrait": 2, "tall_portrait": 0, "landscape": 0, "wide_landscape": 0},
    "aspect_ratio_alerts": [],
    "aspect_ratio_target_recos": ["✅ Dominante carré → parfait SDXL..."],

    "diversity": {
      "clip_clusters": 22, "clip_score": 78,
      "tag_distinct_top30": 28,
      "dominant_attributes": ["vetement", "fond"],
      "overall_score": 67,
      "verdict": "Diversité correcte",
      "big_clusters": [{"size": 3, "names": ["IMG_001.png", "IMG_002.png", "IMG_003.png"]}]
    },

    "target_scores": {
      "SDXL classique": {"score": 72, "grade": "B", "reason": "tout en ordre",
                          "applies_to": ["sdxl_kohya", "onetrainer_sdxl"]},
      "SDXL anime":     {"score": 78, "grade": "B+", "reason": "...",
                          "applies_to": ["pony_kohya", "illustrious_kohya", "noobai_kohya"]},
      "Flux":           {"score": 88, "grade": "A", "reason": "tout en ordre",
                          "applies_to": ["flux_aitoolkit", "flux_kohya", "chroma_aitoolkit"]},
      "Wan vidéo":      {"score": 55, "grade": "C", "reason": "WD14 trop court pour Wan",
                          "applies_to": ["wan21_musubi", "wan22_musubi"]},
      "Vidéo (Hunyuan/Mochi/LTX/CogVideoX)": {"score": 50, "grade": "C", "reason": "...",
                          "applies_to": ["hunyuan_diffpipe", "ltx_video_diffpipe", ...]}
    }
  },
  "recommendations": [
    "🎯 Meilleur target pour ce dataset : « Flux » (A, 88/100) — ...",
    "✅ Identite verifiee (vs ref) : 28 photos correspondent",
    "🌈 Diversité : 67/100 — Diversité correcte",
    "📐 AR : Dominante carré → parfait SDXL",
    "💡 30 captions Kohya generee(s)"
  ],
  "images": [
    {
      "name": "IMG_0042.png",
      "path": "C:\\AI\\datasets\\ma_persona\\IMG_0042.png",
      "width": 832, "height": 1216,
      "sharpness": 312.4, "brightness": 128.3, "contrast": 64.2, "quality_verdict": "OK",
      "phash": "1234567890",
      "face_count": 1, "face_proportion": 12.3,
      "view_type": "face_only", "face_yaw": -3.5,
      "_face_bbox": [120.1, 95.4, 412.7, 521.8],
      "expression": "slight smile",
      "face_similarity_avg": 0.78,
      "face_similarity_to_ref": 0.78, "ref_match": "OK",
      "body_similarity_avg": 0.81, "body_verdict": "NA (close-up)",

      "wd14_tags": "brown hair, indoors, white shirt, smile",
      "natural_caption": "...",
      "joycaption": "A young woman with long dark brown hair, captured...",

      "ai_score": 0.97, "ai_label": "artificial", "is_ai_classifier": true,
      "artifacts_severity": "none", "artifacts_categories": [],
      "ai_metadata_sources": ["filename"], "ai_metadata_confidence": "medium",

      "lora_viable": "yes", "lora_reason": "OK pour LoRA",
      "verdict": "OK"
    }
  ],
  "cache_stats": {"reused": 25, "new": 5}
}
```

---

## 12. Auto-légendage : WD14 vs Florence-2 vs JoyCaption

### 12.1 État du marché 2025-2026

**WD14 (tags booru)** — format historique :
```
brown hair, long hair, indoors, white shirt, smile, looking at viewer
```
- Recommandé par la doc Kohya pour SDXL/SD1.5
- Excellent quand le modèle de base a été entraîné sur datasets booru (Pony, Illustrious, NoobAI)
- Rapide (ONNX), ~30 s pour 30 photos

**Florence-2 (caption naturelle Microsoft)** — **déprécié 2026** pour personnes :
```
a young woman with long brown hair, wearing a white shirt, smiling indoors
```
- **Hallucine emotions/vêtements/contextes sur les personnes** (consensus communautaire)
- Reste utile pour les images générales (objets, paysages) ou en fallback CPU
- ~1 min pour 30 photos sur CPU, ~10 s sur CUDA

**JoyCaption Beta One (LLaVA fine-tuné)** — **standard 2026** :
```
A young woman with long dark brown hair, captured in a three-quarter view portrait.
She wears a beige knit sweater and has a slight smile, looking directly at the camera.
The lighting is soft and diffuse, suggesting an indoor environment with natural window light.
```
- **Standard CivitAI / HuggingFace 2026** pour LoRA persona
- Uncensored, full-sentence, descriptions précises sans hallucination
- Modèle 4 Go INT4 / 8 Go BF16 (CUDA fortement recommandé)
- ~30 s/image sur CUDA, ~2 min/image sur CPU

### 12.2 Quand utiliser quoi

| Tu vises | Captioner |
|----------|-----------|
| SDXL / SD 1.5 / Kohya / OneTrainer | **WD14** |
| Pony / Illustrious / NoobAI | **WD14** (+ quality_prefix auto) |
| Flux / Chroma / SD 3.5 / HunyuanDiT / Sana | **JoyCaption** ⭐ |
| Wan 2.x / HunyuanVideo / LTX-Video / CogVideoX / Mochi / Open-Sora | **JoyCaption** ⭐ |
| Tu hésites encore | **Tous** (les 3 sont stockés, `lora_prep` prend le bon selon le target) |

### 12.3 Conseils pratiques

- **Trigger word toujours en tête** : `persona_alpha, a young woman with...`
- Pour SDXL forks : **quality_prefix injecté automatiquement avant le trigger** : `score_9, score_8_up, score_7_up, source_photo, persona_alpha, ...`
- **Caption dropout 5–10 %** en config trainer pour éviter overfit sur attributs accessoires
- Relis 3-4 `.txt` avant entraînement : si JoyCaption hallucine systématiquement un attribut récurrent (rare mais possible), édite-les. **L'éditeur inline arrive au Lot D.**

---

## 13. Pipeline complet : du clic à l'entraînement

```
1. ComfyUI workflow InstantID (02 ou 12)
       │ génère 30 photos InstantID dans output/
       ▼
2. Copie/déplace les 30 photos vers C:\AI\datasets\ma_persona\
       │
       ▼
3. File Manager → onglet 📊 Analyseur dataset
   - Dossier : C:\AI\datasets\ma_persona
   - Référence : la photo d'ancrage qui a servi à InstantID
   - Captions :
     • SDXL/Pony/Illustrious/NoobAI/OneTrainer → "WD14 tags"
     • Flux / Wan / Hunyuan / SD 3.5 / HunyuanDiT / LTX / CogVideoX / Mochi / Open-Sora → "JoyCaption ⭐"
     • Hésitation → "Tous"
   - 🔍 Analyser
       │ Lots A+B+C : preview live + verdict global + scores par target
       ▼
4. Lecture des résultats :
   - Verdict A/B/C/D/F + meilleur target suggéré automatiquement
   - Scores par famille (SDXL / SDXL anime / Flux / Wan / Vidéo autre)
   - Score diversité + clusters CLIP
   - Distribution AR + alertes buckets
   - Alertes overfit (tags accessoires > 70 %)
   - Détection IA + artefacts + metadata
   - Plan d'action concret
       │
       ▼
5. Nettoyage (dans cet ordre) :
   a. 🗑 Déplacer ratés → _rejected/
   b. 🔧 Floues → upscale → ComfyUI SUPIR → réinjecter
   c. Si "À générer ensuite" → repasse dans InstantID
      Tu ajoutes les nouvelles → relances 🔍 Analyser (cache !)
       │
       ▼
6. 🧬 Préparer LoRA
   - Trigger word
   - Target : suggéré automatiquement par le verdict, sinon choix manuel
   - "Viable only" coché
       │
       ▼
7. Dossier final ouvert dans l'explorer
   - README.txt avec instructions précises pour le trainer choisi
   - Config trainer prête (.toml / .yaml / .bat)
   - Images croppées + captions (trigger word + quality_prefix si SDXL fork)
       │
       ▼
8. Lance le trainer (voir Section 7 pour chaque target)
       │
       ▼
9. LoRA sortant dans output/<persona>_lora.safetensors
       │
       ▼
10. Copie dans C:\AI\ComfyUI-future\ComfyUI_windows_portable\ComfyUI\models\loras\
    Charge dans ComfyUI avec LoraLoader
    Trigger word dans le prompt
```

---

## 14. Troubleshooting

| Symptôme | Cause probable | Fix |
|---------|--------------|-----|
| `Modeles antelopev2 introuvables` | Pas encore utilisé InstantID | Lance le workflow 02 une fois pour télécharger antelopev2 |
| `WD14 indispo` au démarrage | Pas d'accès réseau au 1er run | Téléchargement échoue, l'analyse continue sans tags. Vérifie connexion |
| `Florence-2 indispo (transformers/torch manquant)` | Le venv ComfyUI-future a un torch cassé | Réinstalle torch dans ce venv |
| `JoyCaption indispo` | Pas assez de VRAM (LLaVA 8 Go BF16) | Active INT4 (auto si CUDA dispo). Sur CPU, prends Florence-2 |
| `sdxl-detector indispo` | transformers < 4.40 ou problème HF | Update transformers, vérifie accès HF |
| L'analyse semble bloquée à 0 % | Chargement insightface (CPU) | Normal au 1er run, ~10 s |
| Le tableau reste vide | Erreur silencieuse | Onglet → zone rouge en bas avec traceback complet |
| Cache pas pris en compte | Tu as copié-collé les fichiers, taille a changé | Normal : si taille/mtime change, cache invalide. Reset : supprime `.analyzer_cache.json` |
| JoyCaption ultra lent | CPU sans CUDA | Normal sur CPU (~2 min/image). Sur CUDA INT4 → 30 s/image |
| "Pas assez de photos viables" pour Kohya | < 10 viables | Soit tu génères plus, soit tu désactives "viable only" (mais LoRA souffrira) |
| Crop coupe le menton | margin_ratio trop petit | Édite `lora_prep.py` → `margin_ratio=0.8` ou 1.0 |
| Score IA = 0.97 partout | Dataset 100 % généré InstantID | Normal et attendu, la reco le précise |
| Artefacts severity high partout | Modèle de base hallucinant (Flux Schnell par ex.) | Recommence la génération avec InstantID + SDXL, ou Flux dev |
| AR alerts "tall_portrait" | Tes refs sont 9:21 ou plus extrêmes | Recrop avant import, ou évite Flux multi-bucket |

---

## 15. Performance

Sur CPU 6-cœurs, dataset de 30 photos 832×1216 :

| Phase | Temps |
|-------|-------|
| Chargement insightface (1er run) | 8 s |
| Chargement insightface (suivants) | 2 s |
| Chargement CLIP (1er run) | 6 s |
| Chargement CLIP (suivants) | 1 s |
| Chargement WD14 (1er run, inclut DL 330 Mo) | 60 s |
| Chargement WD14 (suivants) | 3 s |
| Chargement Florence-2 (1er run, DL 540 Mo) | 90 s |
| Chargement Florence-2 (suivants) | 8 s |
| Chargement JoyCaption (1er run, DL 4-8 Go) | 5-10 min |
| Chargement JoyCaption (suivants, CUDA) | 15 s |
| Chargement sdxl-detector (1er run, DL 350 Mo) | 30 s |
| Chargement sdxl-detector (suivants) | 3 s |
| Analyse par image (WD14) | 1.2 s |
| Analyse par image (Florence-2 CPU) | 4 s |
| Analyse par image (Florence-2 CUDA) | 0.5 s |
| Analyse par image (JoyCaption CPU) | 120 s |
| Analyse par image (JoyCaption CUDA INT4) | 30 s |
| Analyse par image (JoyCaption CUDA BF16) | 5 s |
| Détection IA (sdxl-detector) | 0.3 s CPU, 0.05 s CUDA |
| Détection artefacts (regex sur tags) | 0.001 s |
| Lecture metadata IA | 0.01 s |
| Matrices sim + duplicates + clustering + verdict | 1 s |

**Cache hit** : 0.05 s par image. Sur un dataset 50 photos déjà cachées avec 3 nouvelles : 13 s total.

**Recommandation** : pour Flux/Wan, fais l'analyse une fois en mode `joycaption` ou `all` (long mais complet), puis le cache te ramène à 0.05 s par image pour toutes les itérations suivantes.

---

## 16. Roadmap (manquant pour 10/10)

État au 2026-05-31 (après Lots A + B + C + D) :

| Lot | Features | Statut |
|-----|----------|--------|
| **A** | JoyCaption + fréquence tags + alertes overfit | ✅ |
| **B** | sdxl-detector + HADM-light + C2PA/EXIF/PNG metadata | ✅ |
| **C** | Diversité CLIP + scores par target + distribution AR | ✅ |
| **D** | Éditeur de captions inline + masques sujet OneTrainer (BriaRMBG-1.4) | ✅ |
| **E long terme** | Génération auto images manquantes, MirrorMetrics post-train, benchmarks | ⏳ |

---

*Documentation mise à jour 2026-05-31 — Lots A + B + C + D livrés.*
