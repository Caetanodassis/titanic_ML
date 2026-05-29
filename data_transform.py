"""
data_transform.py
=================
Transformadores customizados do sklearn para o pipeline de pré-processamento
do dataset Titanic.

Cada classe herda de BaseEstimator + TransformerMixin, garantindo
compatibilidade total com Pipeline, GridSearchCV e cross_val_score.

Transformadores disponíveis:
    - ColumnDropper              → Remove colunas desnecessárias
    - CategorizerFeatures        → Discretiza Age e Fare em faixas categóricas
    - OrdinalEncoderTransformer  → Codificação ordinal para colunas binárias (ex: Sex)
    - OneHotEmbarkedAndPclassEncoder → OneHot para variáveis nominais/ordinais
    - LogTransformer             → Aplica log1p em colunas numéricas com skewness
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# ---------------------------------------------------------------------------
# Importações para uso nos notebooks (pode importar tudo a partir daqui)
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# 1. ColumnDropper
# ---------------------------------------------------------------------------
class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Remove as colunas especificadas do DataFrame.

    Parâmetros
    ----------
    columns_to_drop : list of str
        Nomes das colunas a serem removidas.
        Colunas inexistentes são ignoradas (errors='ignore').

    Exemplo
    -------
    >>> dropper = ColumnDropper(columns_to_drop=['Cabin', 'Name', 'Ticket'])
    >>> X_clean = dropper.fit_transform(X)
    """

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy().drop(columns=self.columns_to_drop, errors='ignore')


# ---------------------------------------------------------------------------
# 2. CategorizerFeatures
# ---------------------------------------------------------------------------
class CategorizerFeatures(BaseEstimator, TransformerMixin):
    """
    Discretiza as colunas contínuas Age e Fare em faixas categóricas.

    Age → faixa_etaria:
        [0, 2)       → recém-nascido
        [2, 12)      → criança
        [12, 18)     → adolescente
        [18, 30)     → jovem
        [30, 60)     → adulto
        [60, 120)    → idoso
        NaN          → Não_Informado

    Fare → Fare_categoria (quartis do treino):
        [0, 7.91]    → muito_baixa
        (7.91, 14.45]→ baixa
        (14.45, 31.0]→ média
        (31.0, max]  → alta

    Nota
    ----
    O `fit` armazena o máximo de Fare do treino para evitar data leakage
    ao transformar o conjunto de teste.
    """

    def fit(self, X, y=None):
        self.fare_max_ = X['Fare'].max()
        return self

    def transform(self, X):
        X = X.copy()

        # — Age → faixa_etaria
        age_bins   = [0, 2, 12, 18, 30, 60, 120, np.inf]
        age_labels = ['recém-nascido', 'criança', 'adolescente',
                      'jovem', 'adulto', 'idoso', 'Não_Informado']
        X['faixa_etaria'] = pd.cut(
            X['Age'], bins=age_bins, labels=age_labels, right=False
        )
        X['faixa_etaria'] = X['faixa_etaria'].cat.add_categories(['Não_Informado']) \
            if 'Não_Informado' not in X['faixa_etaria'].cat.categories \
            else X['faixa_etaria']
        X['faixa_etaria'] = X['faixa_etaria'].fillna('Não_Informado')
        X.drop(columns=['Age'], inplace=True)

        # — Fare → Fare_categoria
        fare_bins   = [0, 7.91, 14.45, 31.00, self.fare_max_]
        fare_labels = ['muito_baixa', 'baixa', 'média', 'alta']
        X['Fare_categoria'] = pd.cut(
            X['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True
        )
        X.drop(columns=['Fare'], inplace=True)

        return X


# ---------------------------------------------------------------------------
# 3. OrdinalEncoderTransformer
# ---------------------------------------------------------------------------
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica OrdinalEncoder do sklearn nas colunas especificadas.

    Parâmetros
    ----------
    columns : list of str
        Colunas que receberão codificação ordinal (ex: ['Sex']).

    Exemplo
    -------
    >>> enc = OrdinalEncoderTransformer(columns=['Sex'])
    >>> X_enc = enc.fit_transform(X)
    # 'female' → 0.0 | 'male' → 1.0
    """

    def __init__(self, columns=None):
        self.columns = columns or []
        self.encoder_ = None

    def fit(self, X, y=None):
        self.encoder_ = OrdinalEncoder()
        self.encoder_.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.encoder_.transform(X[self.columns])
        return X


# ---------------------------------------------------------------------------
# 4. OneHotEmbarkedAndPclassEncoder
# ---------------------------------------------------------------------------
class OneHotEmbarkedAndPclassEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica OneHotEncoding nas colunas categóricas/nominais:
        Embarked, Pclass, faixa_etaria, Fare_categoria.

    Usa sparse_output=False para retornar um DataFrame denso e
    handle_unknown='ignore' para lidar com categorias novas no teste.

    As colunas originais são removidas e as dummies são concatenadas.
    """

    COLUMNS = ['Embarked', 'Pclass', 'faixa_etaria', 'Fare_categoria']

    def __init__(self):
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.encoder_.fit(X[self.COLUMNS])
        self.feature_names_ = self.encoder_.get_feature_names_out(self.COLUMNS)
        return self

    def transform(self, X):
        X = X.copy()
        encoded     = self.encoder_.transform(X[self.COLUMNS])
        X_dummies   = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        X           = pd.concat([X.drop(columns=self.COLUMNS), X_dummies], axis=1)
        return X


# ---------------------------------------------------------------------------
# 5. LogTransformer
# ---------------------------------------------------------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica log1p nas colunas numéricas especificadas para reduzir skewness.

    Valores nulos ou negativos são mantidos sem transformação.

    Parâmetros
    ----------
    columns : list of str
        Colunas que receberão log1p.

    Exemplo
    -------
    >>> lt = LogTransformer(columns=['Fare'])
    >>> X_log = lt.fit_transform(X)
    """

    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].apply(
                lambda x: np.log1p(x) if pd.notnull(x) and x >= 0 else x
            )
        return X


# ---------------------------------------------------------------------------
# Pipeline padrão — reutilizável em todos os notebooks de modelagem
# ---------------------------------------------------------------------------
def build_preprocessing_pipeline():
    """
    Retorna o pipeline de pré-processamento padrão do projeto.

    Etapas
    ------
    1. ColumnDropper          → remove Cabin, Name, Ticket
    2. CategorizerFeatures    → discretiza Age e Fare
    3. OrdinalEncoderTransformer → codifica Sex
    4. OneHotEmbarkedAndPclassEncoder → one-hot em variáveis categóricas
    5. SimpleImputer          → preenche nulos restantes com moda
    6. StandardScaler         → normaliza features numéricas

    Retorna
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ('drop_columns',  ColumnDropper(columns_to_drop=['Cabin', 'Name', 'Ticket'])),
        ('categorize',    CategorizerFeatures()),
        ('ordinal_enc',   OrdinalEncoderTransformer(columns=['Sex'])),
        ('onehot_enc',    OneHotEmbarkedAndPclassEncoder()),
        ('imputer',       SimpleImputer(strategy='most_frequent')),
        ('scaler',        StandardScaler()),
    ])
