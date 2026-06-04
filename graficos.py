import streamlit as st
import pandas as pd
import joblib
import os

# Configuração da interface do Streamlit
st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Previsão de Sobrevivência — Titanic")
st.markdown("""
Insira as características do passageiro no menu lateral para prever se ele sobreviveria ao naufrágio.
O modelo utiliza um **Voting Classifier Ensemble** idêntico ao criado nos notebooks.
""")

# Formulário no menu lateral para recolha de dados
st.sidebar.header("Características do Passageiro")

pclass = st.sidebar.selectbox("Classe da Passagem (Pclass)", [1, 2, 3], index=2, 
                              help="1 = Primeira Classe; 2 = Segunda Classe; 3 = Terceira Classe")

sex = st.sidebar.selectbox("Sexo (Sex)", ["male", "female"], index=0)

age = st.sidebar.slider("Idade (Age)", min_value=1, max_value=100, value=28)

# Variáveis padronizadas em minúsculas para evitar erros de compilação
sibsp = st.sidebar.number_input("Irmãos/Cônjuges a bordo (SibSp)", min_value=0, max_value=10, value=0)

parch = st.sidebar.number_input("Pais/Filhos a bordo (Parch)", min_value=0, max_value=10, value=0)

fare = st.sidebar.number_input("Preço da Tarifa (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=5.0)

embarked = st.sidebar.selectbox("Porto de Embarque (Embarked)", ["S", "C", "Q"], index=0,
                                help="S = Southampton; C = Cherbourg; Q = Queenstown")

# Botão para executar a previsão na área principal
st.subheader("Análise do Passageiro")
if st.button("Calcular Previsão do Modelo"):
    try:
        # Resolução para Deploy: Encontra o caminho correto do arquivo na nuvem de forma dinâmica
        caminho_atual = os.path.dirname(__file__)
        caminho_modelo = os.path.join(caminho_atual, 'modelo_titanic_voting.pkl')
        
        # 1. Carrega o modelo completo (Pipeline + Classificador)
        modelo = joblib.load(caminho_modelo)
        
        # 2. Cria o DataFrame com as colunas EXATAMENTE iguais às do formato original do Titanic (Kaggle)
        dados_usuario = pd.DataFrame([{
            'PassengerId': 0,        
            'Pclass': pclass,
            'Name': 'Test',
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,          
            'Parch': parch,          
            'Ticket': '123',
            'Fare': fare,
            'Cabin': None,
            'Embarked': embarked
        }])
        
        # 3. Realiza as previsões usando o pipeline
        predicao = modelo.predict(dados_usuario)
        probabilidade = modelo.predict_proba(dados_usuario)
        
        st.write("---")
        if predicao[0] == 1:
            st.success(f"🎉 **Resultado: O passageiro SOBREVIVERIA!**")
            st.metric(label="Probabilidade de Sobrevivência", value=f"{probabilidade[0][1] * 100:.2f}%")
        else:
            st.error(f"😟 **Resultado: O passageiro NÃO SOBREVIVERIA.**")
            st.metric(label="Probabilidade de Falecimento", value=f"{probabilidade[0][0] * 100:.2f}%")
            
    except FileNotFoundError:
        st.error("❌ O ficheiro 'modelo_titanic_voting.pkl' não foi encontrado. "
                 "Certifique-se de que fez o upload do arquivo .pkl para o seu repositório do GitHub.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
