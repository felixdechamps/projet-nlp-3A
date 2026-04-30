"""
analyse_corpus.py
=================
Analyse exploratoire du corpus Archelec pour le projet NLP ENSAE 2026.
Génère toutes les figures et statistiques nécessaires pour la section
"Data Analysis" du rapport.

Usage:
    python analyse_corpus.py

Sorties (dossier ./figures/) :
    - fig1_docs_par_annee.png
    - fig2_docs_par_parti.png
    - fig3_distribution_longueurs.png
    - fig4_tfidf_top_terms.png
    - fig5_wordcloud_par_parti.png   (optionnel, nécessite wordcloud)
    - fig6_lexical_richness.png
    - corpus_stats.txt               (tableau récapitulatif)
"""

import re
import os
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────

PATTERNS = [
    "data/1981/legislatives/*PF*.txt",
    "data/1988/legislatives/*PF*.txt",
    "data/1993/legislatives/*PF*.txt",
]
META_CSV = "data/archelect_search.csv"
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

PARTIES_ORDER = [
    "Extreme_Droite", "Droite", "Centre",
    "Socialiste", "Communiste_et_Extreme_Gauche",
    "Ecologiste", "Regionaliste", "Sans_Etiquette",
]
PARTY_COLORS = {
    "Extreme_Droite":              "#1a0000",
    "Droite":                      "#003189",
    "Centre":                      "#FF7F0E",
    "Socialiste":                  "#e03030",
    "Communiste_et_Extreme_Gauche":"#8B0000",
    "Ecologiste":                  "#2ca02c",
    "Regionaliste":                "#9467BD",
    "Sans_Etiquette":              "#7f7f7f",
}

# Stopwords français minimaux (complétez si besoin)
STOPWORDS_FR = set("""
le la les un une des de du en et à au aux pour par sur avec dans
est sont était ont que qui ne pas plus mais ou bien tout aussi plus très
cette ces ce il elle ils elles nous vous je tu se ses son sa nos vos leur leurs
me te lui y en dont où car or ni donc
""".split())

# ── 1. HELPERS ────────────────────────────────────────────────────────────────

_RE_HYPHENATION = re.compile(r'([A-Za-zÀ-ÿ]+)-\s+([A-Za-zÀ-ÿ]+)')
_RE_WATERMARK   = re.compile(r'Sciences Po / fonds CEVIPOF|[☐☒@¥]')
_RE_MULTILINE   = re.compile(r'\n{3,}')
_RE_MULTSPACE   = re.compile(r' {2,}')


def clean_ocr(text: str) -> str:
    text = _RE_HYPHENATION.sub(r'\1\2', text)
    text = _RE_WATERMARK.sub("", text)
    text = _RE_MULTILINE.sub('\n\n', text)
    text = _RE_MULTSPACE.sub(' ', text)
    return text.strip()


def normalize_party(soutien) -> str:
    if pd.isna(soutien) or soutien == "non mentionné":
        return "A_EXCLURE"
    s = str(soutien).lower().split(';')[0].strip()
    if any(x in s for x in ["front national","fn","extrême droite","trop d'immigrés",
                              "nationaliste","royaliste","action française","forces nouvelles"]):
        return "Extreme_Droite"
    if any(x in s for x in ["communiste","pcf","lutte ouvrière","lcr","marxiste",
                              "trotskyste","parti des travailleurs","psu","combat ouvrier"]):
        return "Communiste_et_Extreme_Gauche"
    if any(x in s for x in ["écolog","ecolog","vert","environnement","amis de la terre",
                              "biosphère"]) and "chasse" not in s:
        return "Ecologiste"
    if any(x in s for x in ["socialiste","ps","mrg","radicaux de gauche",
                              "majorité présidentielle","gauche progressiste"]):
        return "Socialiste"
    if any(x in s for x in ["rpr","rassemblement pour la république","gaulliste",
                              "cni","parti républicain","droite","républicain indépendant"]):
        return "Droite"
    if any(x in s for x in ["udf","union pour la démocratie française","centre","cds",
                              "centriste","démocratie chrétienne","parti radical"]):
        return "Centre"
    if any(x in s for x in ["chasse","cpnt","rurale"]):
        return "Droite"
    if any(x in s for x in ["corse","kanak","breton","emgann","occitan","catalan",
                              "guadeloup","indépendantiste","abertzale"]):
        return "Regionaliste"
    if any(x in s for x in ["sans étiquette","apolitique","indépendant","libre"]):
        return "Sans_Etiquette"
    return "A_EXCLURE"


def tokenize_fr(text: str) -> list[str]:
    """Tokenisation simple, retrait des stopwords et des tokens < 3 chars."""
    tokens = re.findall(r"[a-zà-ÿ]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS_FR and len(t) >= 3]

# ── 2. CHARGEMENT DES DONNÉES ─────────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    rows = []
    for pattern in PATTERNS:
        for p in sorted(Path().glob(pattern)):
            annee = p.parts[-3]
            text  = clean_ocr(p.read_text(encoding="utf-8", errors="replace"))
            rows.append({"id": p.stem, "annee": annee, "text": text, "path": str(p)})

    df = pd.DataFrame(rows)
    meta = pd.read_csv(META_CSV)
    COLS = ["id","titulaire-prenom","titulaire-nom","titulaire-profession",
            "titulaire-soutien","contexte-tour","date","departement-nom",
            "titulaire-sexe","titulaire-age"]
    meta = meta[[c for c in COLS if c in meta.columns]]
    df = df.merge(meta, on="id", how="left")
    df["famille_politique"] = df["titulaire-soutien"].apply(normalize_party)
    df = df[df["famille_politique"] != "A_EXCLURE"].copy()
    df["nb_chars"]  = df["text"].str.len()
    df["nb_tokens"] = df["text"].apply(lambda t: len(tokenize_fr(t)))
    df["nb_words"]  = df["text"].apply(lambda t: len(t.split()))
    return df


# ── 3. FIGURES ────────────────────────────────────────────────────────────────

def fig1_docs_par_annee(df):
    counts = df.groupby(["annee","famille_politique"]).size().unstack(fill_value=0)
    # Réordonner les colonnes
    cols = [p for p in PARTIES_ORDER if p in counts.columns]
    counts = counts[cols]
    colors = [PARTY_COLORS[c] for c in cols]

    ax = counts.plot(kind="bar", stacked=True, figsize=(9, 5), color=colors,
                     edgecolor="white", linewidth=0.4)
    ax.set_title("Nombre de professions de foi par année et famille politique", fontsize=13, pad=12)
    ax.set_xlabel("Année d'élection")
    ax.set_ylabel("Nombre de documents")
    ax.legend(title="Famille politique", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_xticklabels(counts.index, rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_docs_par_annee.png", dpi=150)
    plt.close()
    print("✓ fig1 sauvegardée")


def fig2_docs_par_parti(df):
    counts = df["famille_politique"].value_counts().reindex(PARTIES_ORDER).dropna()
    colors = [PARTY_COLORS[p] for p in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_title("Distribution des documents par famille politique (corpus total)", fontsize=12)
    ax.set_xlabel("Nombre de professions de foi")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_docs_par_parti.png", dpi=150)
    plt.close()
    print("✓ fig2 sauvegardée")


def fig3_distribution_longueurs(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogramme global
    axes[0].hist(df["nb_chars"], bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["nb_chars"].median(), color="red", lw=2,
                    label=f"Médiane : {int(df['nb_chars'].median())} chars")
    axes[0].set_title("Distribution des longueurs (caractères)")
    axes[0].set_xlabel("Nombre de caractères")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()

    # Box-plot par famille politique
    data_by_party = [df[df["famille_politique"]==p]["nb_tokens"].values
                     for p in PARTIES_ORDER if p in df["famille_politique"].values]
    labels = [p.replace("_et_Extreme_Gauche", "+\nExt.G").replace("_", " ")
              for p in PARTIES_ORDER if p in df["famille_politique"].values]
    bp = axes[1].boxplot(data_by_party, vert=True, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 2})
    for patch, party in zip(bp["boxes"], [p for p in PARTIES_ORDER if p in df["famille_politique"].values]):
        patch.set_facecolor(PARTY_COLORS[party])
        patch.set_alpha(0.7)
    axes[1].set_xticks(range(1, len(labels)+1))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=15)
    axes[1].set_title("Longueur (tokens) par famille politique")
    axes[1].set_ylabel("Nombre de tokens")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_distribution_longueurs.png", dpi=150)
    plt.close()
    print("✓ fig3 sauvegardée")


def fig4_tfidf_top_terms(df, top_n=12):
    """Pour chaque famille politique, affiche les top_n termes TF-IDF distinctifs."""
    parties = [p for p in PARTIES_ORDER if p in df["famille_politique"].values]
    n_cols = 4
    n_rows = (len(parties) + n_cols - 1) // n_cols

    # Calcul TF-IDF global
    vectorizer = TfidfVectorizer(
        max_features=8000,
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zà-ÿ]{3,}\b",
    )
    corpus_texts = df["text"].str.lower().tolist()
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    # Stop-words supplémentaires
    extra_stop = set(STOPWORDS_FR) | {"france","français","nationale","politique",
                                       "candidat","élection","législatives","vote"}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()

    for i, party in enumerate(parties):
        mask   = df["famille_politique"] == party
        scores = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        # Filtrer les stop-words
        valid  = np.array([v not in extra_stop for v in vocab])
        scores_filtered = scores * valid

        idx      = np.argsort(scores_filtered)[::-1][:top_n]
        words    = vocab[idx]
        values   = scores_filtered[idx]

        color = PARTY_COLORS.get(party, "#888")
        axes[i].barh(words[::-1], values[::-1], color=color, alpha=0.75, edgecolor="white")
        axes[i].set_title(party.replace("_", " "), fontsize=9, fontweight="bold")
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].set_xlabel("TF-IDF moyen", fontsize=7)

    # Masquer les axes vides
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top termes distinctifs par famille politique (TF-IDF)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_tfidf_top_terms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig4 sauvegardée")


def fig5_tsne_projection(df):
    """Projection t-SNE des documents dans l'espace TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=2000, min_df=5, sublinear_tf=True,
                                  token_pattern=r"(?u)\b[a-zà-ÿ]{3,}\b")
    X = vectorizer.fit_transform(df["text"].str.lower())

    # PCA d'abord pour réduire le bruit avant t-SNE
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(10, 7))
    for party in PARTIES_ORDER:
        mask = df["famille_politique"] == party
        if mask.sum() == 0:
            continue
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=PARTY_COLORS[party], label=party.replace("_"," "),
                   s=12, alpha=0.55, linewidths=0)

    ax.legend(title="Famille politique", markerscale=2, fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Projection t-SNE des professions de foi (espace TF-IDF, 2 000 features)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig5 sauvegardée")


def fig6_lexical_richness(df):
    """Type-Token Ratio (TTR) et richesse lexicale par famille politique et par année."""
    def ttr(text):
        tokens = tokenize_fr(text)
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    df["ttr"] = df["text"].apply(ttr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # TTR par famille politique
    parties = [p for p in PARTIES_ORDER if p in df["famille_politique"].values]
    means   = [df[df["famille_politique"]==p]["ttr"].mean() for p in parties]
    colors  = [PARTY_COLORS[p] for p in parties]
    axes[0].bar(range(len(parties)), means, color=colors, edgecolor="white")
    axes[0].set_xticks(range(len(parties)))
    axes[0].set_xticklabels([p.replace("_"," ").replace(" et "," +\n") for p in parties],
                              rotation=15, ha="right", fontsize=8)
    axes[0].set_title("Richesse lexicale (TTR) par famille politique")
    axes[0].set_ylabel("Type-Token Ratio moyen")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # TTR par année
    ttr_by_year = df.groupby("annee")["ttr"].mean()
    axes[1].plot(ttr_by_year.index, ttr_by_year.values, marker="o", linewidth=2, color="#4C72B0")
    axes[1].set_title("Évolution du TTR moyen au fil des élections")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("TTR moyen")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_lexical_richness.png", dpi=150)
    plt.close()
    print("✓ fig6 sauvegardée")


def save_stats(df):
    """Sauvegarde un résumé statistique complet dans corpus_stats.txt."""
    lines = []
    lines.append("=" * 60)
    lines.append("STATISTIQUES CORPUS ARCHELEC — Résumé")
    lines.append("=" * 60)
    lines.append(f"\nTotal documents : {len(df)}")
    lines.append(f"Années couvertes : {sorted(df['annee'].unique())}")
    lines.append(f"\n--- Documents par année ---")
    lines.append(df["annee"].value_counts().sort_index().to_string())
    lines.append(f"\n--- Documents par famille politique ---")
    lines.append(df["famille_politique"].value_counts().to_string())
    lines.append(f"\n--- Longueur (caractères) ---")
    lines.append(df["nb_chars"].describe().round(1).to_string())
    lines.append(f"\n--- Longueur (mots) ---")
    lines.append(df["nb_words"].describe().round(1).to_string())

    if "titulaire-sexe" in df.columns:
        lines.append(f"\n--- Genre des candidats ---")
        lines.append(df["titulaire-sexe"].value_counts().to_string())

    lines.append(f"\n--- Longueur médiane par famille politique ---")
    med = df.groupby("famille_politique")["nb_words"].median().sort_values(ascending=False)
    lines.append(med.to_string())

    output = "\n".join(lines)
    (OUT_DIR / "corpus_stats.txt").write_text(output, encoding="utf-8")
    print("✓ corpus_stats.txt sauvegardé")
    print("\n" + output)
    return df


# ── 4. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chargement du corpus...")
    df = load_corpus()
    print(f"  {len(df)} documents chargés.\n")

    print("Génération des figures...")
    fig1_docs_par_annee(df)
    fig2_docs_par_parti(df)
    fig3_distribution_longueurs(df)
    fig4_tfidf_top_terms(df)
    fig5_tsne_projection(df)
    fig6_lexical_richness(df)
    save_stats(df)

    print("\nTerminé. Figures dans ./figures/")
