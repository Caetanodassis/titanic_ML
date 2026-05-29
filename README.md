# 🚢 Titanic — Classificação de Sobrevivência

> Projeto de Machine Learning para prever quais passageiros do Titanic sobreviviram,
> usando **Regressão Logística**, **Random Forest** e um **Voting Classifier** ensemble.

---

## 📁 Estrutura do Projeto

```
titanic_project/
│
├── dataset/
│   └── titanicc/
│       ├── train.csv              # Dados de treino (com target Survived)
│       ├── test.csv               # Dados de teste (sem target)
│       └── gender_submission.csv  # Labels reais do conjunto de teste
│
├── data_transform.py              # Transformadores customizados do sklearn
│
├── 01_EDA.ipynb                   # Análise Exploratória de Dados
├── 02_Logistic_Regression.ipynb   # Modelo 1: Regressão Logística
├── 03_Random_Forest.ipynb         # Modelo 2: Random Forest
├── 04_Voting_Classifier.ipynb     # Modelo 3: Ensemble (Hard + Soft Voting)
│
└── README.md
```

---

## 📋 Sobre o Dataset

O dataset é o clássico [Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic) do Kaggle.

| Coluna | Tipo | Descrição |
|---|---|---|
| `PassengerId` | int | ID único do passageiro |
| `Survived` | int | Target: 0 = Morreu, 1 = Sobreviveu |
| `Pclass` | int | Classe da passagem (1, 2 ou 3) |
| `Name` | str | Nome completo |
| `Sex` | str | Sexo (`male` / `female`) |
| `Age` | float | Idade em anos |
| `SibSp` | int | Nº de irmãos/cônjuges a bordo |
| `Parch` | int | Nº de pais/filhos a bordo |
| `Ticket` | str | Número do bilhete |
| `Fare` | float | Tarifa paga |
| `Cabin` | str | Cabine (~77% ausente) |
| `Embarked` | str | Porto de embarque: S, C ou Q |

---

## 🔄 Pipeline de Pré-processamento

Todos os notebooks de modelagem compartilham o mesmo pipeline, definido em `data_transform.py` e instanciado por `build_preprocessing_pipeline()`.

```
Entrada (raw features)
        │
        ▼
┌─────────────────────────────────────────────┐
│ 1. ColumnDropper                            │
│    Remove: Cabin, Name, Ticket              │
├─────────────────────────────────────────────┤
│ 2. CategorizerFeatures                      │
│    Age  → faixa_etaria (7 categorias)       │
│    Fare → Fare_categoria (4 quartis)        │
├─────────────────────────────────────────────┤
│ 3. OrdinalEncoderTransformer                │
│    Sex: female=0 | male=1                   │
├─────────────────────────────────────────────┤
│ 4. OneHotEmbarkedAndPclassEncoder           │
│    OneHot: Embarked, Pclass,                │
│            faixa_etaria, Fare_categoria     │
├─────────────────────────────────────────────┤
│ 5. SimpleImputer (strategy=most_frequent)   │
│    Preenche nulos remanescentes             │
├─────────────────────────────────────────────┤
│ 6. StandardScaler                           │
│    Normaliza features numéricas             │
└─────────────────────────────────────────────┘
        │
        ▼
  Features prontas para o modelo
```

### Por que discretizar Age e Fare?

- **`Age`** tem ~20% de valores ausentes. Em vez de imputar um valor numérico (que introduziria viés), a coluna é convertida em faixas etárias com uma categoria explícita `Não_Informado` para os nulos — preservando o sinal da informação ausente.

- **`Fare`** tem distribuição fortemente assimétrica (skewed). A discretização em quartis remove outliers de forma natural e torna o modelo menos sensível a valores extremos.

---

## 🧠 Modelos

### 1. Regressão Logística (`02_Logistic_Regression.ipynb`)

Modelo linear com regularização. O `GridSearchCV` busca a melhor combinação de:

| Parâmetro | Valores testados |
|---|---|
| `penalty` | `l1`, `l2` |
| `solver` | `liblinear`, `saga` |
| `C` | `0.01`, `0.1`, `1`, `10` |

**Por que usar:** Rápido, interpretável e bom baseline. A regularização L1 também tem efeito de seleção de features.

---

### 2. Random Forest (`03_Random_Forest.ipynb`)

Ensemble de árvores de decisão. O `GridSearchCV` busca:

| Parâmetro | Valores testados |
|---|---|
| `n_estimators` | `25`, `50`, `75`, `100` |
| `min_samples_split` | `2`, `5`, `10` |
| `min_samples_leaf` | `1`, `2`, `4` |

**Por que usar:** Robusto a outliers, captura não-linearidades e fornece importância das features.

---

### 3. Voting Classifier (`04_Voting_Classifier.ipynb`)

Combina Logistic Regression + Random Forest + KNN com dois modos de votação:

| Modo | Como funciona |
|---|---|
| **Hard** | Cada modelo vota (0 ou 1) → ganha o mais votado |
| **Soft** | Média das probabilidades de cada modelo → maior prob. vence |

**Por que usar:** Reduz a variância ao combinar modelos com erros distintos. O Soft Voting tende a ser mais estável porque usa a confiança de cada modelo em vez de votos binários.

---

## 🏗️ `data_transform.py` — Transformadores Customizados

Todos os transformadores herdam de `BaseEstimator + TransformerMixin`, garantindo:
- Compatibilidade total com `Pipeline`, `GridSearchCV` e `cross_val_score`
- Método `.fit()` aprende parâmetros do treino (sem data leakage no teste)
- Método `.transform()` aplica a transformação

| Classe | Função |
|---|---|
| `ColumnDropper` | Remove colunas irrelevantes |
| `CategorizerFeatures` | Discretiza Age e Fare em faixas |
| `OrdinalEncoderTransformer` | Codificação ordinal (Sex) |
| `OneHotEmbarkedAndPclassEncoder` | OneHot para variáveis nominais |
| `LogTransformer` | log1p para reduzir skewness |
| `build_preprocessing_pipeline()` | Retorna o pipeline completo pronto para uso |

---

## 🚀 Como executar

**1. Clone o repositório e instale as dependências:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

**2. Adicione os datasets na pasta correta:**
```
dataset/titanicc/train.csv
dataset/titanicc/test.csv
dataset/titanicc/gender_submission.csv
```

**3. Execute os notebooks na ordem:**
```
01_EDA.ipynb
02_Logistic_Regression.ipynb
03_Random_Forest.ipynb
04_Voting_Classifier.ipynb
```

> ⚠️ Os notebooks de modelagem importam `data_transform.py` via `from data_transform import ...`. Certifique-se de que o arquivo esteja na mesma pasta dos notebooks ao executar.

---

## 📊 Principais Achados da EDA

| Achado | Impacto |
|---|---|
| Mulheres sobreviveram muito mais (~74% vs ~19%) | `Sex` é a feature mais discriminante |
| 1ª classe sobreviveu mais (~63%) | `Pclass` é feature importante |
| Crianças tiveram maior sobrevivência | Discretizar `Age` agrega valor |
| Tarifas altas correlacionam com sobrevivência | Discretizar `Fare` em quartis |
| `Cabin` tem ~77% nulo | Remover a coluna |
| `Name`, `Ticket`, `PassengerId` | Sem valor preditivo → remover |

---

*Dataset: Titanic — Machine Learning from Disaster (Kaggle) | Modelagem com scikit-learn*
