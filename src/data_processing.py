"""
data_processing.py
==================
Ce fichier contient toutes les fonctions de traitement des données.
Il est utilisé par train_model.py et le notebook eda.ipynb.

Pipeline :
    load_data() → optimize_memory() → prepare_features() → split_data() → normalize_data()
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Charge le dataset CSV depuis le chemin donné.

    Args:
        path (str): Chemin vers le fichier CSV

    Returns:
        pd.DataFrame: Dataset chargé

    Example:
        df = load_data('../data/heart_failure_clinical_records_dataset.csv')
    """
    df = pd.read_csv(path)

    print(f"✅ Dataset chargé : {df.shape[0]} patients, {df.shape[1]} colonnes")

    return df


# ============================================================
# 2. OPTIMISATION DE LA MÉMOIRE
# ============================================================

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit la mémoire utilisée par le DataFrame en convertissant
    les types de données vers des types moins lourds :
        - float64 (8 octets) → float32 (4 octets)
        - int64   (8 octets) → int32   (4 octets)

    Args:
        df (pd.DataFrame): DataFrame original

    Returns:
        pd.DataFrame: DataFrame optimisé (~50% moins de mémoire)

    Example:
        df_optimized = optimize_memory(df)
    """
    df = df.copy()

    # Mémoire avant optimisation
    before_kb = df.memory_usage(deep=True).sum() / 1024

    # Convertir float64 → float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Convertir int64 → int32
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # Mémoire après optimisation
    after_kb = df.memory_usage(deep=True).sum() / 1024
    reduction = ((before_kb - after_kb) / before_kb) * 100

    print(f"✅ Mémoire AVANT : {before_kb:.2f} KB")
    print(f"✅ Mémoire APRÈS : {after_kb:.2f} KB")
    print(f"✅ Réduction     : {reduction:.1f}%")

    return df


# ============================================================
# 3. SÉPARATION DES FEATURES ET DE LA TARGET
# ============================================================

def prepare_features(df: pd.DataFrame):
    """
    Sépare le DataFrame en :
        - X : features (variables d'entrée)
        - y : target   (variable à prédire = DEATH_EVENT)

    DEATH_EVENT :
        0 = patient survivant
        1 = patient décédé

    Args:
        df (pd.DataFrame): Dataset complet

    Returns:
        X (pd.DataFrame): Features (12 colonnes)
        y (pd.Series)   : Target (DEATH_EVENT)

    Example:
        X, y = prepare_features(df)
    """
    # X = toutes les colonnes SAUF DEATH_EVENT
    X = df.drop('DEATH_EVENT', axis=1)

    # y = seulement la colonne DEATH_EVENT
    y = df['DEATH_EVENT']

    print(f"✅ Features X : {X.shape}")
    print(f"✅ Target y   : {y.shape}")
    print(f"   Survivants (0) : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Décédés    (1) : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    return X, y


# ============================================================
# 4. DIVISION TRAIN / TEST
# ============================================================

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divise les données en ensemble d'entraînement et de test.

    - Train (80%) : le modèle apprend sur ces données
    - Test  (20%) : on évalue le modèle sur ces données

    IMPORTANT :
        - stratify=y garantit la même proportion 68/32 dans train ET test
        - Le balance (class_weight) se fera dans train_model.py
          UNIQUEMENT sur les données d'entraînement

    Args:
        X            : Features
        y            : Target
        test_size    : Proportion du test (défaut 20%)
        random_state : Graine aléatoire pour reproductibilité

    Returns:
        X_train, X_test, y_train, y_test

    Example:
        X_train, X_test, y_train, y_test = split_data(X, y)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # garde la proportion 68/32 dans train et test
    )

    print(f"✅ Train : {X_train.shape[0]} patients (80%)")
    print(f"✅ Test  : {X_test.shape[0]} patients (20%)")
    print(f"   Train - Survivants : {(y_train == 0).sum()} | Décédés : {(y_train == 1).sum()}")
    print(f"   Test  - Survivants : {(y_test == 0).sum()}  | Décédés : {(y_test == 1).sum()}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 5. NORMALISATION DES DONNÉES
# ============================================================

def normalize_data(X_train, X_test):
    """
    Normalise les features avec StandardScaler.

    StandardScaler transforme chaque feature pour avoir :
        - Moyenne = 0
        - Écart-type = 1

    IMPORTANT :
        - fit_transform sur X_train : apprend ET transforme
        - transform sur X_test      : transforme SEULEMENT
        (on n'apprend pas sur le test pour éviter le data leakage)

    Exemple de transformation :
        age avant  : 40, 55, 75, 95
        age après  : -1.5, -0.3, 0.8, 1.9

    Args:
        X_train : Features d'entraînement
        X_test  : Features de test

    Returns:
        X_train_scaled : Features train normalisées
        X_test_scaled  : Features test normalisées
        scaler         : L'objet scaler (sauvegardé pour l'app)

    Example:
        X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    """
    scaler = StandardScaler()

    # Apprend les paramètres sur train ET transforme
    X_train_scaled = scaler.fit_transform(X_train)

    # Transforme test avec les mêmes paramètres que train
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Normalisation appliquée")
    print(f"   Moyenne train (ex age) : {X_train_scaled[:, 0].mean():.4f}")
    print(f"   Std train    (ex age)  : {X_train_scaled[:, 0].std():.4f}")

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# 6. PIPELINE COMPLÈTE
# ============================================================

def run_pipeline(path: str):
    """
    Lance toute la pipeline de traitement des données en une seule fois.

    Étapes :
        1. Charger les données
        2. Optimiser la mémoire
        3. Préparer les features
        4. Diviser train/test
        5. Normaliser

    Args:
        path (str): Chemin vers le fichier CSV

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler

    Example:
        X_train, X_test, y_train, y_test, scaler = run_pipeline('../data/heart_failure_clinical_records_dataset.csv')
    """
    print("=" * 50)
    print("🚀 PIPELINE DE TRAITEMENT DES DONNÉES")
    print("=" * 50)

    # Étape 1 : Charger
    print("\n📂 Étape 1 : Chargement des données")
    df = load_data(path)

    # Étape 2 : Optimiser
    print("\n💾 Étape 2 : Optimisation mémoire")
    df = optimize_memory(df)

    # Étape 3 : Préparer
    print("\n🔧 Étape 3 : Préparation des features")
    X, y = prepare_features(df)

    # Étape 4 : Diviser
    print("\n✂️  Étape 4 : Division Train/Test")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Étape 5 : Normaliser
    print("\n📏 Étape 5 : Normalisation")
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)

    print("\n" + "=" * 50)
    print("✅ PIPELINE TERMINÉE — Données prêtes pour le ML !")
    print("=" * 50)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    """
    Lance ce fichier directement pour tester la pipeline :
        python src/data_processing.py
    """
    X_train, X_test, y_train, y_test, scaler = run_pipeline(
        '../data/heart_failure_clinical_records_dataset.csv'
    )