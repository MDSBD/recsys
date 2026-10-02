"""TP 3 : recommandation de contenu, TF-IDF et reranking MMR.

Préparer : python -m pip install numpy pandas scikit-learn
Exécuter : python TP_Chapitre_3.py --catalogue catalogue_chapitre3.csv
Le catalogue pédagogique est supposé entièrement disponible à la date initiale.
Aucune connexion n'est nécessaire après installation des dépendances.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def build_index(catalogue):
    if catalogue["item_id"].duplicated().any():
        raise ValueError("Les identifiants d'items doivent être uniques.")
    texts = (catalogue["title"].fillna("") + " " +
             catalogue["description"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        smooth_idf=True, sublinear_tf=False, norm="l2", min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def make_profile(catalogue, X, likes):
    """likes = dictionnaire {item_id: poids strictement positif}."""
    positions = {item: i for i, item in enumerate(catalogue["item_id"])}
    if not likes:
        raise ValueError("Profil vide : demander des préférences ou utiliser une baseline.")
    unknown = set(likes) - set(positions)
    if unknown:
        raise ValueError(f"Items inconnus : {sorted(unknown)}")
    weights = np.asarray(list(likes.values()), dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Les poids doivent être finis et strictement positifs.")
    indices = [positions[item] for item in likes]
    profile = np.asarray(X[indices].multiply(weights[:, None]).sum(axis=0)).ravel()
    norm = np.linalg.norm(profile)
    if norm <= 1e-12:
        raise ValueError("Profil nul : les préférences ne sont pas représentées.")
    return profile / norm


def recommend(catalogue, vectorizer, X, likes, k=5, lam=1.0, language="fr"):
    if k < 1 or not 0 <= lam <= 1:
        raise ValueError("k doit être positif et lambda dans [0, 1].")
    profile = make_profile(catalogue, X, likes)
    scores = linear_kernel(X, profile.reshape(1, -1)).ravel()
    eligible = ((catalogue["language"] == language)
                & (catalogue["available"] == 1)
                & ~catalogue["item_id"].isin(likes))
    # Dans ce TP, on ne complète pas artificiellement avec des scores nuls.
    remaining = [int(i) for i in np.flatnonzero(eligible.to_numpy()) if scores[i] > 1e-12]
    chosen, rows = [], []
    names = vectorizer.get_feature_names_out()
    for _ in range(min(k, len(remaining))):
        if not chosen:
            marginal = {i: float(scores[i]) for i in remaining}
        else:
            sim = linear_kernel(X[remaining], X[chosen])
            marginal = {i: float(lam * scores[i] - (1 - lam) * sim[pos].max())
                        for pos, i in enumerate(remaining)}
        best = max(remaining, key=lambda i: (marginal[i], scores[i], -i))
        contributions = X[best].multiply(profile).tocoo()
        pairs = sorted(zip(contributions.col, contributions.data), key=lambda p: -p[1])
        terms = ", ".join(names[j] for j, value in pairs[:4] if value > 0)
        rows.append({"item_id": catalogue.iloc[best]["item_id"],
                     "title": catalogue.iloc[best]["title"],
                     "cosine": float(scores[best]), "selection_score": marginal[best],
                     "shared_terms": terms})
        chosen.append(best)
        remaining.remove(best)
    return pd.DataFrame(rows, columns=["item_id", "title", "cosine", "selection_score", "shared_terms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path,
                        default=Path(__file__).with_name("catalogue_chapitre3.csv"))
    parser.add_argument("--likes", nargs="+", default=["R01", "R02"])
    parser.add_argument("--lambda-mmr", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    catalogue = pd.read_csv(args.catalogue).reset_index(drop=True)
    required = {"item_id", "title", "description", "language", "available"}
    if not required.issubset(catalogue.columns):
        raise ValueError(f"Colonnes manquantes : {required - set(catalogue.columns)}")
    vectorizer, X = build_index(catalogue)
    result = recommend(catalogue, vectorizer, X, {i: 1.0 for i in args.likes},
                       k=args.k, lam=args.lambda_mmr)
    print(f"Catalogue fictif : {len(catalogue)} items ; {X.shape[1]} dimensions TF-IDF.")
    print(result.to_string(index=False) if not result.empty else "Aucun candidat avec score positif.")
    result.to_csv("recommandations_chapitre3.csv", index=False)
    print("Résultats : recommandations_chapitre3.csv")


if __name__ == "__main__":
    main()
