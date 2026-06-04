import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Importa o pipeline que você já estruturou no seu ficheiro original
from data_transform import build_preprocessing_pipeline

print("A carregar os dados e a treinar o modelo final...")

# 1. Carregar os dados originais
df = pd.read_csv('Titanic-Dataset.csv')

X = df.drop(columns=['Survived'])
y = df['Survived']

# 2. Dividir os dados mantendo a consistência do seu projeto
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Construir o pré-processamento existente
preprocessor = build_preprocessing_pipeline()

# 4. Configurar os três modelos com os hiperparâmetros que descobriu nos notebooks
lr = LogisticRegression(C=1, penalty='l2', solver='liblinear', random_state=42)
rf = RandomForestClassifier(n_estimators=75, min_samples_leaf=2, min_samples_split=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='uniform')

# 5. Criar o Voting Classifier (Soft Voting) igual ao notebook 04
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('knn', knn)],
    voting='soft'
)

# 6. Unir o Pré-processamento ao Classificador num único Pipeline
# Isto é crucial para que o Streamlit consiga transformar os novos dados automaticamente!
modelo_completo = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

# 7. Treinar o modelo
modelo_completo.fit(X_train, y_train)

# 8. Guardar o modelo treinado num ficheiro físico .pkl
joblib.dump(modelo_completo, 'modelo_titanic_voting.pkl')
print("✅ Ficheiro 'modelo_titanic_voting.pkl' gerado com sucesso!")