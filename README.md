<div align="center">

# 🚢 Titanic — Classificação de Sobrevivência

**Projeto completo de Machine Learning** para prever a sobrevivência de passageiros do Titanic utilizando um pipeline de pré-processamento customizado e um ensemble de modelos.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **Acurácia final (Soft Voting):** `~82%` no conjunto de teste  
> **Dataset:** [Titanic — Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)

<br/>

[🔍 Ver EDA](#-análise-exploratória-eda) · [🧠 Modelos](#-modelos) · [🚀 Como executar](#-como-executar) · [🌐 Demo online](#-demo-streamlit)

</div>

---

## 📋 Índice

- [Sobre o projeto](#-sobre-o-projeto)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Dataset](#-sobre-o-dataset)
- [Pipeline de pré-processamento](#-pipeline-de-pré-processamento)
- [Análise Exploratória (EDA)](#-análise-exploratória-eda)
- [Modelos](#-modelos)
- [Resultados](#-resultados)
- [Demo Streamlit](#-demo-streamlit)
- [Como executar](#-como-executar)
- [Dependências](#-dependências)
- [Melhorias futuras](#-melhorias-futuras)

---

## 🎯 Sobre o projeto

Este projeto aplica um fluxo completo de ciência de dados — da exploração dos dados ao deploy de uma aplicação interativa — usando o clássico dataset do Titanic como base.

O objetivo é prever se um passageiro sobreviveria ao naufrágio com base em suas características (sexo, idade, classe social, tarifa, porto de embarque, etc.).

**O que foi feito:**

- ✅ Análise exploratória profunda (EDA) com visualizações interativas
- ✅ Pipeline de pré-processamento reutilizável com transformadores customizados do `sklearn`
- ✅ Otimização de hiperparâmetros via `GridSearchCV` para dois modelos base
- ✅ Ensemble final com `VotingClassifier` (Hard + Soft Voting)
- ✅ Aplicação web interativa com `Streamlit` para previsões em tempo real

---

## 📁 Estrutura do repositório

```
titanic_project/
│
├── 📓 Notebooks (executar em ordem)
│   ├── 01_EDA.ipynb                    # Análise Exploratória de Dados
│   ├── 02_Logistic_Regression.ipynb    # Modelo 1 — Regressão Logística + GridSearchCV
│   ├── 03_Random_Forest.ipynb          # Modelo 2 — Random Forest + GridSearchCV
│   └── 04_Voting_Classifier.ipynb      # Modelo 3 — Ensemble Hard + Soft Voting
│
├── 🐍 Scripts Python
│   ├── data_transform.py               # Transformadores customizados do sklearn
│   ├── treinar_modelo.py               # Script para treinar e salvar o modelo .pkl
│   └── graficos.py                     # Aplicação Streamlit (interface web)
│
├── 📊 Dados
│   └── Titanic-Dataset.csv             # Dataset principal (Kaggle)
│
├── 🤖 Modelo treinado
│   └── modelo_titanic_voting.pkl       # Pipeline completo serializado (joblib)
│
└── 📄 docs/
    └── index.html                      # GitHub Pages — documentação visual
```

---

## 📋 Sobre o Dataset

O dataset é o clássico [Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic) do Kaggle, com **891 passageiros** no conjunto de treino.

| Coluna | Tipo | Descrição |
|---|---|---|
| `PassengerId` | `int` | ID único do passageiro |
| `Survived` | `int` | **Target:** 0 = Não sobreviveu · 1 = Sobreviveu |
| `Pclass` | `int` | Classe da passagem: 1ª, 2ª ou 3ª |
| `Name` | `str` | Nome completo |
| `Sex` | `str` | Sexo: `male` / `female` |
| `Age` | `float` | Idade em anos (~20% ausente) |
| `SibSp` | `int` | Nº de irmãos/cônjuges a bordo |
| `Parch` | `int` | Nº de pais/filhos a bordo |
| `Ticket` | `str` | Número do bilhete |
| `Fare` | `float` | Tarifa paga |
| `Cabin` | `str` | Cabine (~77% ausente) |
| `Embarked` | `str` | Porto de embarque: `S` = Southampton · `C` = Cherbourg · `Q` = Queenstown |

---

## 🔄 Pipeline de Pré-processamento

Todos os notebooks de modelagem compartilham o mesmo pipeline, definido em `data_transform.py` e instanciado por `build_preprocessing_pipeline()`.

```
Entrada (features brutas do CSV)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│ 1. ColumnDropper                                        │
│    Remove: Cabin (~77% nulo), Name, Ticket              │
│    → sem valor preditivo ou dados insuficientes         │
├─────────────────────────────────────────────────────────┤
│ 2. CategorizerFeatures                                  │
│    Age  → faixa_etaria (7 categorias + Não_Informado)   │
│    Fare → Fare_categoria (4 quartis: muito_baixa→alta)  │
│    → elimina outliers; nulos viram categoria explícita  │
├─────────────────────────────────────────────────────────┤
│ 3. OrdinalEncoderTransformer                            │
│    Sex: female = 0.0 | male = 1.0                       │
├─────────────────────────────────────────────────────────┤
│ 4. OneHotEmbarkedAndPclassEncoder                       │
│    OneHot: Embarked, Pclass, faixa_etaria, Fare_cat.    │
│    → handle_unknown='ignore' para categorias novas      │
├─────────────────────────────────────────────────────────┤
│ 5. SimpleImputer (strategy = most_frequent)             │
│    Preenche nulos remanescentes (ex: Embarked: 2 nulos) │
├─────────────────────────────────────────────────────────┤
│ 6. StandardScaler                                       │
│    Normaliza features numéricas (SibSp, Parch, Sex)     │
└─────────────────────────────────────────────────────────┘
          │
          ▼
  Features prontas para o classificador
```

### Por que discretizar Age e Fare?

| Feature | Problema | Solução |
|---|---|---|
| `Age` | ~20% de nulos. Imputar média/mediana introduz viés e perde o sinal da ausência. | Discretizar em 7 faixas; nulos viram `Não_Informado` — a ausência vira informação. |
| `Fare` | Distribuição fortemente assimétrica (skewed right). Outliers distorcem modelos lineares. | Quartis removem outliers naturalmente e tornam o modelo robusto. |

---

## 🔍 Análise Exploratória (EDA)

> Notebook: `01_EDA.ipynb`

### Principais achados

| Achado | Taxa | Impacto no modelo |
|---|---|---|
| Mulheres sobreviveram muito mais | 74% vs 19% (homens) | `Sex` = feature mais discriminante |
| 1ª classe sobreviveu mais | 63% vs 47% (2ª) vs 24% (3ª) | `Pclass` = feature importante |
| Crianças (<12 anos) tiveram mais sobrevivência | ~58% | Discretizar `Age` agrega valor |
| Tarifas altas correlacionam com sobrevivência | Quartil alto: ~65% | Discretizar `Fare` em quartis |
| `Cabin` tem 77% de nulos | — | Remover a coluna |
| Cherbourg teve mais sobreviventes | 55% vs 34% (S) | `Embarked` contribui marginalmente |

### Taxa de sobrevivência geral

```
Total de passageiros : 891
Sobreviventes        : 342  (38.4%)
Mortos               : 549  (61.6%)
```

---

## 🧠 Modelos

### Modelo 1 — Regressão Logística
> Notebook: `02_Logistic_Regression.ipynb`

Modelo linear com regularização. Serve como baseline interpretável e rápido.

**GridSearchCV** (cv=5, scoring=accuracy):

| Parâmetro | Valores testados | Melhor valor |
|---|---|---|
| `penalty` | `l1`, `l2` | `l2` |
| `solver` | `liblinear`, `saga` | `liblinear` |
| `C` | `0.01`, `0.1`, `1`, `10` | `1` |

**Por que usar:** Treinamento instantâneo, coeficientes interpretáveis, regularização L1 tem efeito de seleção de features.

---

### Modelo 2 — Random Forest
> Notebook: `03_Random_Forest.ipynb`

Ensemble de 75 árvores de decisão com bootstrap e `max_features='sqrt'`.

**GridSearchCV** (cv=5, n_jobs=-1):

| Parâmetro | Valores testados | Melhor valor |
|---|---|---|
| `n_estimators` | `25`, `50`, `75`, `100` | `75` |
| `min_samples_split` | `2`, `5`, `10` | `10` |
| `min_samples_leaf` | `1`, `2`, `4` | `2` |

**Por que usar:** Robusto a outliers, captura relações não-lineares, fornece importância das features sem necessidade de escalonamento.

---

### Modelo 3 — Voting Classifier (Ensemble)
> Notebook: `04_Voting_Classifier.ipynb`

Combina os três melhores modelos: **Logistic Regression + Random Forest + KNN**.

| Modelo | Configuração final |
|---|---|
| Logistic Regression | `C=1, penalty='l2', solver='liblinear'` |
| Random Forest | `n_estimators=75, min_samples_leaf=2, min_samples_split=10` |
| KNN | `n_neighbors=5, metric='manhattan', weights='uniform'` |

**Hard Voting** → cada modelo emite um voto binário (0 ou 1); vence a maioria.  
**Soft Voting** → média das probabilidades de cada modelo; vence a maior probabilidade média.

> O Soft Voting tende a superar o Hard porque usa a *confiança* de cada modelo, não apenas o voto.

---

## 📊 Resultados

| Modelo | Acurácia | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Regressão Logística | ~80% | — | — | — |
| Random Forest | ~81% | — | — | — |
| **Soft Voting** ⭐ | **~82%** | — | — | — |
| Hard Voting | ~81% | — | — | — |

> Os valores exatos dependem do split treino/teste e podem variar ligeiramente a cada execução.

---

## 🌐 Demo Streamlit

A aplicação `graficos.py` é uma interface web construída com **Streamlit** que permite prever a sobrevivência de qualquer passageiro em tempo real.

### Como funciona a página

```
┌─────────────────────────────────────────────────────────┐
│  MENU LATERAL (sidebar)                                 │
│  ─────────────────                                      │
│  • Classe da Passagem  [1 / 2 / 3]                      │
│  • Sexo                [male / female]                  │
│  • Idade               [slider 1–100]                   │
│  • SibSp               [irmãos/cônjuges a bordo]        │
│  • Parch               [pais/filhos a bordo]            │
│  • Fare                [tarifa paga]                    │
│  • Porto de Embarque   [S / C / Q]                      │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼  Clica "Calcular Previsão"
┌─────────────────────────────────────────────────────────┐
│  ÁREA PRINCIPAL                                         │
│  ─────────────                                          │
│  1. Carrega modelo_titanic_voting.pkl                   │
│  2. Monta DataFrame com estrutura idêntica ao CSV       │
│  3. modelo.predict() → 0 ou 1                           │
│  4. modelo.predict_proba() → probabilidades             │
│  5. Exibe resultado:                                    │
│     🎉 "Sobreviveria!" + % de sobrevivência             │
│     😟 "Não sobreviveria." + % de falecimento           │
└─────────────────────────────────────────────────────────┘
```

**O pipeline completo** (pré-processamento + modelo) está encapsulado no `.pkl`, então a aplicação simplesmente passa os dados brutos e recebe a previsão — sem precisar reaplicar transformações manualmente.

### Rodar localmente

```bash
streamlit run graficos.py
```

Acesse em: `http://localhost:8501`

---

## 🚀 Como executar

### Pré-requisitos

- Python 3.9+
- pip

### 1. Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/titanic-ml.git
cd titanic-ml
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute os notebooks (opcional — exploração)

```bash
jupyter notebook
```

Abra e execute na ordem:
```
01_EDA.ipynb
02_Logistic_Regression.ipynb
03_Random_Forest.ipynb
04_Voting_Classifier.ipynb
```

> ⚠️ Certifique-se de que `data_transform.py` está na mesma pasta dos notebooks.

### 4. Treine e salve o modelo

```bash
python treinar_modelo.py
```

Isso cria o arquivo `modelo_titanic_voting.pkl` com o pipeline completo treinado.

### 5. Rode a aplicação web

```bash
streamlit run graficos.py
```

Acesse em: `http://localhost:8501`

---

## 📦 Dependências

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
streamlit>=1.30.0
joblib>=1.3.0
jupyter>=1.0.0
```

Instale tudo de uma vez:
```bash
pip install -r requirements.txt
```

---

## 🔧 `data_transform.py` — Transformadores Customizados

Todos os transformadores herdam de `BaseEstimator + TransformerMixin`, garantindo compatibilidade total com `Pipeline`, `GridSearchCV` e `cross_val_score`.

| Classe | Função |
|---|---|
| `ColumnDropper` | Remove colunas irrelevantes (Cabin, Name, Ticket) |
| `CategorizerFeatures` | Discretiza Age e Fare em faixas categóricas |
| `OrdinalEncoderTransformer` | Codificação ordinal para Sex |
| `OneHotEmbarkedAndPclassEncoder` | OneHot para variáveis nominais/ordinais |
| `LogTransformer` | Aplica log1p para reduzir skewness em features numéricas |
| `build_preprocessing_pipeline()` | Retorna o pipeline completo pronto para uso |

**Uso rápido:**
```python
from data_transform import build_preprocessing_pipeline

pipeline = build_preprocessing_pipeline()
X_transformed = pipeline.fit_transform(X_train)
```

---

## 💡 Melhorias Futuras

- [ ] Feature engineering: extrair título (`Mr.`, `Mrs.`, `Miss.`) do campo `Name`
- [ ] Testar `XGBoost` e `LightGBM` como estimadores adicionais no ensemble
- [ ] Adicionar gráfico de importância de features na aplicação Streamlit
- [ ] Deploy no Streamlit Community Cloud
- [ ] Adicionar testes unitários para os transformadores customizados
- [ ] Exportar previsões em CSV diretamente pela interface

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<div align="center">

Feito com ❤️ e muito `fit_transform()`

⭐ Se este projeto foi útil, considere dar uma estrela!

</div>
