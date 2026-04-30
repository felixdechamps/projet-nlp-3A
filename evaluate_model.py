"""
evaluate_model.py
=================
Évaluation quantitative du modèle Llama-3.2-1B fine-tuné (QLoRA) vs modèle de base.

Métriques calculées :
  1. Perplexity sur le test set (fine-tuné vs base)
  2. ROUGE-1 / ROUGE-2 / ROUGE-L  (générations vs textes réels)
  3. BERTScore (générations vs textes réels)
  4. Distinctiveness politique : classification linéaire sur TF-IDF des textes générés
  5. Lexical overlap intra-parti vs inter-parti (Jaccard sur bigrammes)

Usage:
    python evaluate_model.py --model_id fdechamps/Llama-1B-Archelec-XXXXXXXX-XXXX \
                             --base_id  meta-llama/Llama-3.2-1B-Instruct \
                             --data_dir data/

Sorties (dossier ./eval_results/) :
    - perplexity_comparison.png
    - rouge_scores.png
    - bertscore_distribution.png
    - political_classifier_confusion.png
    - eval_summary.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)

PARTIES_ORDER = [
    "Extreme_Droite", "Droite", "Centre",
    "Socialiste", "Communiste_et_Extreme_Gauche",
    "Ecologiste", "Regionaliste",
]

# 8 profils tests (un par famille politique) – identiques à ceux du notebook
CANDIDATS_TEST = [
    {"famille": "Extreme_Droite",              "prenom": "Martial",      "nom": "Delaborde",
     "profession": "Artisan commerçant",        "soutien": "Front national",
     "departement": "Bouches-du-Rhône",         "date": "1988", "tour": "1"},
    {"famille": "Droite",                       "prenom": "Charles-Henri","nom": "de Courcelles",
     "profession": "Chef d'entreprise",         "soutien": "Rassemblement pour la République",
     "departement": "Hauts-de-Seine",           "date": "1988", "tour": "1"},
    {"famille": "Centre",                       "prenom": "François",     "nom": "Lemaire",
     "profession": "Médecin généraliste",       "soutien": "Union pour la démocratie française",
     "departement": "Calvados",                 "date": "1988", "tour": "1"},
    {"famille": "Socialiste",                   "prenom": "Alain",        "nom": "Mignot",
     "profession": "Professeur de lycée",       "soutien": "Parti socialiste",
     "departement": "Nord",                     "date": "1988", "tour": "1"},
    {"famille": "Communiste_et_Extreme_Gauche", "prenom": "Marcel",       "nom": "Roussillon",
     "profession": "Ouvrier métallurgiste",     "soutien": "Parti communiste français",
     "departement": "Seine-Saint-Denis",        "date": "1988", "tour": "1"},
    {"famille": "Ecologiste",                   "prenom": "Brigitte",     "nom": "Valette",
     "profession": "Chercheuse en biologie",    "soutien": "Verts",
     "departement": "Isère",                    "date": "1988", "tour": "1"},
    {"famille": "Regionaliste",                 "prenom": "Yannick",      "nom": "Le Goff",
     "profession": "Agriculteur",               "soutien": "Union démocratique bretonne",
     "departement": "Finistère",                "date": "1988", "tour": "1"},
]

# ── 1. HELPERS ────────────────────────────────────────────────────────────────

def normalize_party(soutien) -> str:
    if pd.isna(soutien) or str(soutien) == "non mentionné":
        return "A_EXCLURE"
    s = str(soutien).lower().split(';')[0].strip()
    if any(x in s for x in ["front national","fn","extrême droite"]):
        return "Extreme_Droite"
    if any(x in s for x in ["communiste","pcf","lutte ouvrière","lcr","psu"]):
        return "Communiste_et_Extreme_Gauche"
    if any(x in s for x in ["écolog","ecolog","vert","biosphère"]) and "chasse" not in s:
        return "Ecologiste"
    if any(x in s for x in ["socialiste","ps","mrg","radicaux de gauche"]):
        return "Socialiste"
    if any(x in s for x in ["rpr","gaulliste","parti républicain","droite"]):
        return "Droite"
    if any(x in s for x in ["udf","centre","cds","parti radical"]):
        return "Centre"
    if any(x in s for x in ["corse","breton","catalan","indépendantiste"]):
        return "Regionaliste"
    if any(x in s for x in ["sans étiquette","indépendant"]):
        return "Sans_Etiquette"
    return "A_EXCLURE"


def build_prompt(row: dict, tokenizer) -> str:
    system_msg = (
        "Tu es un expert en rhétorique politique française et un archiviste spécialisé "
        "dans l'histoire de la Ve République. Ta tâche est de rédiger une profession de "
        "foi électorale historique et convaincante."
    )
    user_msg = (
        "Rédige la profession de foi à partir des caractéristiques suivantes :\n"
        f"- Candidat : {row['prenom']} {row['nom']}\n"
        f"- Profession : {row['profession']}\n"
        f"- Soutien politique : {row['soutien']}\n"
        f"- Élection : Élections législatives, Tour {row['tour']}\n"
        f"- Date : {row['date']}\n"
        f"- Département : {row['departement']}"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_text(model, tokenizer, prompt: str, max_new_tokens=512, device="cuda") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # On ne décode que les tokens générés
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── 2. PERPLEXITY ─────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, texts: list[str], device="cuda",
                        max_length=1024, batch_size=4) -> float:
    """
    Calcule la perplexité moyenne du modèle sur une liste de textes.
    Retourne la perplexité (exp(mean NLL)).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100  # Ignore padding

        with torch.no_grad():
            out = model(**enc, labels=labels)
        # out.loss est la NLL moyenne sur les tokens non-maskés
        n_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    mean_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    return float(np.exp(mean_nll))


# ── 3. ROUGE ──────────────────────────────────────────────────────────────────

def compute_rouge(hypotheses: list[str], references: list[str]) -> dict:
    """Calcule ROUGE-1, ROUGE-2, ROUGE-L."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("⚠  rouge_score non installé. `pip install rouge-score`")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
    results = defaultdict(list)
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for k, v in scores.items():
            results[k].append(v.fmeasure)
    return {k: float(np.mean(v)) for k, v in results.items()}


# ── 4. BERTSCORE ──────────────────────────────────────────────────────────────

def compute_bertscore(hypotheses: list[str], references: list[str],
                       lang="fr") -> dict:
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("⚠  bert_score non installé. `pip install bert-score`")
        return {}

    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return {
        "precision": float(P.mean()),
        "recall":    float(R.mean()),
        "f1":        float(F1.mean()),
        "f1_per_doc": F1.tolist(),
    }


# ── 5. DISTINCTIVITÉ POLITIQUE ────────────────────────────────────────────────

def political_distinctiveness(generated_texts: list[str],
                               generated_labels: list[str],
                               real_texts: list[str],
                               real_labels: list[str]) -> dict:
    """
    Entraîne un classificateur linéaire (LR sur TF-IDF) sur les vrais textes
    et prédit les étiquettes des textes générés.
    Retourne l'accuracy et la matrice de confusion.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import seaborn as sns

    # Entraînement sur les vrais textes
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zà-ÿ]{3,}\b")
    X_train = vec.fit_transform([t.lower() for t in real_texts])
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_train, real_labels)

    # Prédiction sur les textes générés
    X_gen = vec.transform([t.lower() for t in generated_texts])
    preds = clf.predict(X_gen)
    acc   = accuracy_score(generated_labels, preds)

    print(f"\n=== Distinctiveness politique ===")
    print(f"Accuracy (prédiction étiquette sur générations) : {acc:.3f}")
    print(classification_report(generated_labels, preds, zero_division=0))

    # Matrice de confusion
    labels_unique = sorted(set(generated_labels))
    cm = confusion_matrix(generated_labels, preds, labels=labels_unique)
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels_unique, yticklabels=labels_unique, ax=ax)
    except ImportError:
        ax.imshow(cm, cmap="Blues")
    ax.set_title("Matrice de confusion : familles politiques (générations)", fontsize=12)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "political_classifier_confusion.png", dpi=150)
    plt.close()
    print("✓ political_classifier_confusion.png sauvegardée")

    return {"accuracy": acc, "predictions": preds.tolist()}


# ── 6. PLOT SUMMARY ───────────────────────────────────────────────────────────

def plot_perplexity_comparison(ppl_base: float, ppl_finetuned: float):
    fig, ax = plt.subplots(figsize=(5, 4))
    models  = ["Llama-3.2-1B\n(base)", "Llama-3.2-1B\n(fine-tuné)"]
    values  = [ppl_base, ppl_finetuned]
    colors  = ["#999999", "#2196F3"]
    bars = ax.bar(models, values, color=colors, edgecolor="white", width=0.4)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Perplexité sur le test set Archelec", fontsize=12)
    ax.set_ylabel("Perplexité (↓ meilleure)")
    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "perplexity_comparison.png", dpi=150)
    plt.close()
    print("✓ perplexity_comparison.png sauvegardée")


def plot_rouge_scores(rouge_base: dict, rouge_ft: dict):
    if not rouge_base or not rouge_ft:
        return
    metrics = list(rouge_base.keys())
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, [rouge_base[m] for m in metrics], w,
           label="Base", color="#999", edgecolor="white")
    ax.bar(x + w/2, [rouge_ft[m]   for m in metrics], w,
           label="Fine-tuné", color="#2196F3", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("F1 score")
    ax.set_title("Scores ROUGE : base vs fine-tuné (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rouge_scores.png", dpi=150)
    plt.close()
    print("✓ rouge_scores.png sauvegardée")


def plot_bertscore_distribution(f1_base: list, f1_ft: list):
    if not f1_base or not f1_ft:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(f1_base, bins=30, alpha=0.6, label="Base", color="#999")
    ax.hist(f1_ft,   bins=30, alpha=0.6, label="Fine-tuné", color="#2196F3")
    ax.axvline(np.mean(f1_base), color="gray", lw=2, ls="--",
               label=f"Moy. base {np.mean(f1_base):.3f}")
    ax.axvline(np.mean(f1_ft),   color="#0D47A1", lw=2, ls="--",
               label=f"Moy. FT {np.mean(f1_ft):.3f}")
    ax.set_title("Distribution BERTScore F1 : base vs fine-tuné")
    ax.set_xlabel("BERTScore F1")
    ax.set_ylabel("Fréquence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bertscore_distribution.png", dpi=150)
    plt.close()
    print("✓ bertscore_distribution.png sauvegardée")


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",   required=True,
                        help="HuggingFace model ID du modèle fine-tuné")
    parser.add_argument("--base_id",    default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model ID du modèle de base")
    parser.add_argument("--data_dir",   default="data/",
                        help="Dossier racine des données Archelec")
    parser.add_argument("--max_gen",    type=int, default=512)
    parser.add_argument("--n_test",     type=int, default=100,
                        help="Nombre de textes du test set à évaluer (perplexité)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # --- Chargement des modèles ---
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    print("\nChargement du modèle fine-tuné...")
    model_ft, tokenizer = FastLanguageModel.from_pretrained(
        args.model_id, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model_ft)

    print("Chargement du modèle de base...")
    model_base, _ = FastLanguageModel.from_pretrained(
        args.base_id, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model_base)

    # --- Chargement du test set ---
    print("\nChargement des données de test...")
    # On réutilise la même logique que le notebook pour reconstituer le test set
    from datasets import load_dataset
    import pandas as pd

    PATTERNS = [
        f"{args.data_dir}/1981/legislatives/*PF*.txt",
        f"{args.data_dir}/1988/legislatives/*PF*.txt",
        f"{args.data_dir}/1993/legislatives/*PF*.txt",
    ]
    files = [p for pat in PATTERNS for p in sorted(Path().glob(pat))]
    dataset = load_dataset("text", data_files=PATTERNS, split="train",
                           sample_by="document")
    meta = pd.read_csv(f"{args.data_dir}/archelect_search.csv")
    COLS = ["id","titulaire-prenom","titulaire-nom","titulaire-profession",
            "titulaire-soutien","contexte-tour","date","departement-nom"]
    meta_dict = meta[[c for c in COLS if c in meta.columns]].set_index("id").to_dict("index")

    def add_meta(ex, idx):
        p = files[idx]
        ex["id"] = p.stem
        m = meta_dict.get(p.stem, {})
        for k, col in [("prenom","titulaire-prenom"),("nom","titulaire-nom"),
                        ("profession","titulaire-profession"),("soutien","titulaire-soutien"),
                        ("tour","contexte-tour"),("date","date"),("departement","departement-nom")]:
            ex[k] = m.get(col, "non mentionné")
        return ex

    dataset = dataset.map(add_meta, with_indices=True)
    df = dataset.to_pandas()
    df["famille"] = df["soutien"].apply(normalize_party)
    df = df[df["famille"] != "A_EXCLURE"]
    test_df = df.sample(frac=0.1, random_state=42).head(args.n_test)
    print(f"  {len(test_df)} documents dans le test set.")

    # ── A. PERPLEXITÉ ──────────────────────────────────────────────────────────
    print("\n=== A. Perplexité ===")
    test_texts = test_df["text"].tolist()
    ppl_ft   = compute_perplexity(model_ft,   tokenizer, test_texts, device)
    ppl_base = compute_perplexity(model_base, tokenizer, test_texts, device)
    print(f"  Perplexité base     : {ppl_base:.2f}")
    print(f"  Perplexité fine-tuné: {ppl_ft:.2f}")
    plot_perplexity_comparison(ppl_base, ppl_ft)

    # ── B. ROUGE & BERTSCORE ───────────────────────────────────────────────────
    print("\n=== B. ROUGE & BERTScore ===")
    print("  Génération des réponses sur le test set (peut prendre du temps)...")
    hypotheses_ft, hypotheses_base, references = [], [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = build_prompt(row.to_dict(), tokenizer)
        hypotheses_ft.append(  generate_text(model_ft,   tokenizer, prompt, args.max_gen, device))
        hypotheses_base.append(generate_text(model_base, tokenizer, prompt, args.max_gen, device))
        references.append(row["text"])

    rouge_ft   = compute_rouge(hypotheses_ft,   references)
    rouge_base = compute_rouge(hypotheses_base, references)
    print(f"  ROUGE fine-tuné : {rouge_ft}")
    print(f"  ROUGE base      : {rouge_base}")
    plot_rouge_scores(rouge_base, rouge_ft)

    bs_ft   = compute_bertscore(hypotheses_ft,   references)
    bs_base = compute_bertscore(hypotheses_base, references)
    print(f"  BERTScore F1 fine-tuné : {bs_ft.get('f1', 'N/A'):.3f}")
    print(f"  BERTScore F1 base      : {bs_base.get('f1', 'N/A'):.3f}")
    plot_bertscore_distribution(bs_base.get("f1_per_doc",[]),
                                bs_ft.get("f1_per_doc",[]))

    # ── C. DISTINCTIVITÉ POLITIQUE ─────────────────────────────────────────────
    print("\n=== C. Distinctivité politique ===")
    # Générer un texte par candidat de test
    gen_texts, gen_labels = [], []
    for c in CANDIDATS_TEST:
        prompt = build_prompt(c, tokenizer)
        gen_texts.append(generate_text(model_ft, tokenizer, prompt, args.max_gen, device))
        gen_labels.append(c["famille"])

    # Générer davantage de textes si souhaité (répéter chaque profil 3x)
    for c in CANDIDATS_TEST * 2:
        prompt = build_prompt(c, tokenizer)
        gen_texts.append(generate_text(model_ft, tokenizer, prompt, args.max_gen, device))
        gen_labels.append(c["famille"])

    distinctiveness = political_distinctiveness(
        gen_texts, gen_labels,
        test_df["text"].tolist(), test_df["famille"].tolist()
    )

    # ── D. SAUVEGARDE JSON ─────────────────────────────────────────────────────
    summary = {
        "perplexity_base":       ppl_base,
        "perplexity_finetuned":  ppl_ft,
        "rouge_base":            rouge_base,
        "rouge_finetuned":       rouge_ft,
        "bertscore_base":        {k: v for k, v in bs_base.items() if k != "f1_per_doc"},
        "bertscore_finetuned":   {k: v for k, v in bs_ft.items()   if k != "f1_per_doc"},
        "political_accuracy":    distinctiveness.get("accuracy"),
    }
    (OUT_DIR / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n✓ Résultats sauvegardés dans {OUT_DIR}/eval_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
