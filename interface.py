"""
interface.py
===========
Aplicação Streamlit para previsão de sobrevivência no Titanic.
Usa o pipeline completo serializado em modelo_titanic_voting.pkl.

Execute com:
    streamlit run interface.py
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Configuração da página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos customizados ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle   { color: #666; font-size: 1rem; margin-top: 0; }
    .result-box { padding: 1.5rem; border-radius: 12px; text-align: center; }
    .result-survived { background: #d4edda; border: 1px solid #c3e6cb; }
    .result-died     { background: #f8d7da; border: 1px solid #f5c6cb; }
    .big-prob { font-size: 3rem; font-weight: 800; line-height: 1; }
    .section-divider { border: none; border-top: 1px solid #eee; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar — Inputs ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧳 Dados do Passageiro")
    st.caption("Preencha as características para obter a previsão.")
    st.divider()

    pclass = st.selectbox(
        "Classe da Passagem",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: f"{x}ª Classe {'(Luxo)' if x==1 else '(Intermediária)' if x==2 else '(Econômica)'}",
        help="1 = Primeira Classe · 2 = Segunda Classe · 3 = Terceira Classe"
    )

    sex = st.selectbox(
        "Sexo",
        options=["female", "male"],
        format_func=lambda x: "Feminino" if x == "female" else "Masculino"
    )

    age = st.slider("Idade", min_value=1, max_value=80, value=28)

    col1, col2 = st.columns(2)
    with col1:
        sibsp = st.number_input("SibSp", min_value=0, max_value=8, value=0,
                                help="Irmãos/Cônjuges a bordo")
    with col2:
        parch = st.number_input("Parch", min_value=0, max_value=6, value=0,
                                help="Pais/Filhos a bordo")

    fare = st.number_input(
        "Tarifa (Fare — £)",
        min_value=0.0, max_value=600.0, value=32.0, step=5.0,
        format="%.2f"
    )

    embarked = st.selectbox(
        "Porto de Embarque",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
    )

    st.divider()
    predict_btn = st.button("🔍  Calcular Previsão", use_container_width=True, type="primary")


# ─── Área principal ────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🚢 Titanic — Previsão de Sobrevivência</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Voting Classifier Ensemble · Logistic Regression + Random Forest + KNN</p>', unsafe_allow_html=True)
st.divider()

if not predict_btn:
    # Estado inicial — resumo informativo
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Passageiros no dataset", "891")
    col_b.metric("Taxa de sobrevivência", "38.4%")
    col_c.metric("Acurácia do modelo", "~82%")
    col_d.metric("Modelos no ensemble", "3")

    st.info("👈 **Preencha os dados no menu lateral** e clique em **Calcular Previsão** para ver o resultado.")

    # Insights rápidos
    with st.expander("📊 Fatores que mais influenciaram a sobrevivência (EDA)"):
        st.markdown("""
| Fator | Detalhe |
|---|---|
| 🚺 **Sexo** | Mulheres: 74% de sobrevivência · Homens: 19% |
| 🎩 **Classe** | 1ª classe: 63% · 2ª: 47% · 3ª: 24% |
| 👶 **Idade** | Crianças (<12 anos): ~58% de sobrevivência |
| 💰 **Tarifa** | Quartil mais alto: ~65% de sobrevivência |
| 🚢 **Porto** | Cherbourg: 55% · Southampton: 34% |
        """)

else:
    # ─── Previsão ────────────────────────────────────────────────────────────
    try:
        modelo = joblib.load('modelo_titanic_voting.pkl')

        dados_usuario = pd.DataFrame([{
            'PassengerId': 0,
            'Pclass': pclass,
            'Name': 'Test, Mr. Prediction',
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Ticket': '000000',
            'Fare': fare,
            'Cabin': None,
            'Embarked': embarked
        }])

        predicao    = modelo.predict(dados_usuario)[0]
        proba       = modelo.predict_proba(dados_usuario)[0]
        prob_surv   = proba[1] * 100
        prob_died   = proba[0] * 100

        # ─── Resultado principal ───────────────────────────────────────────
        col_result, col_chart = st.columns([1, 1], gap="large")

        with col_result:
            if predicao == 1:
                st.success(f"### 🎉 Sobreviveria!")
                cor_principal = "#28a745"
            else:
                st.error(f"### 😟 Não sobreviveria.")
                cor_principal = "#dc3545"

            st.metric(
                label="Probabilidade de Sobrevivência",
                value=f"{prob_surv:.1f}%",
                delta=f"{prob_surv - 50:+.1f}% vs base de 50%"
            )
            st.metric(
                label="Probabilidade de Falecimento",
                value=f"{prob_died:.1f}%",
            )

            # Resumo do passageiro
            st.divider()
            st.markdown("**Resumo do passageiro**")
            st.markdown(f"""
- **Sexo:** {'Feminino' if sex == 'female' else 'Masculino'}
- **Idade:** {age} anos
- **Classe:** {pclass}ª
- **Tarifa:** £{fare:.2f}
- **Familiares a bordo:** {sibsp + parch}
- **Porto:** { {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}.get(embarked) }
""")

        with col_chart:
            # Gráfico de probabilidades
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            bars = ax.barh(
                ['Não sobreviveu', 'Sobreviveu'],
                [prob_died, prob_surv],
                color=['#dc3545' if predicao == 0 else '#f5a5ae',
                       '#28a745' if predicao == 1 else '#a5d6b0'],
                height=0.55, edgecolor='none'
            )
            # Destaca a previsão
            bars[predicao].set_color(cor_principal)

            for bar, val in zip(bars, [prob_died, prob_surv]):
                ax.text(
                    bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', ha='left', fontsize=14, fontweight='bold',
                    color='#333'
                )

            ax.set_xlim(0, 110)
            ax.set_xlabel('Probabilidade (%)', fontsize=10, color='#666')
            ax.set_title('Probabilidades do modelo (Soft Voting)', fontsize=11, color='#444', pad=12)
            ax.axvline(50, color='#ccc', linewidth=1, linestyle='--')
            ax.text(50.5, -0.55, '50%', fontsize=8, color='#aaa')
            ax.tick_params(colors='#666')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False)

            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ─── Contexto histórico ───────────────────────────────────────────
        st.divider()
        st.markdown("#### 📈 Contexto: perfis similares no dataset real")

        # Calcula estatísticas para perfil similar
        sex_label   = 'female' if sex == 'female' else 'male'
        pclass_label = pclass
        context_texto = ""

        if sex == 'female' and pclass == 1:
            context_texto = "Mulheres de 1ª classe tiveram **97%** de taxa de sobrevivência — o grupo mais privilegiado do naufrágio."
        elif sex == 'female' and pclass == 2:
            context_texto = "Mulheres de 2ª classe sobreviveram em **92%** dos casos — prioridade alta nos botes salva-vidas."
        elif sex == 'female' and pclass == 3:
            context_texto = "Mulheres de 3ª classe sobreviveram em **50%** dos casos — dificuldade de acesso aos botes foi um fator decisivo."
        elif sex == 'male' and pclass == 1:
            context_texto = "Homens de 1ª classe sobreviveram em apenas **37%** dos casos — mesmo o status elevado não garantiu sobrevivência."
        elif sex == 'male' and pclass == 2:
            context_texto = "Homens de 2ª classe sobreviveram em apenas **16%** dos casos — um dos grupos mais afetados."
        else:
            context_texto = "Homens de 3ª classe sobreviveram em apenas **13%** dos casos — o grupo com menor taxa de sobrevivência."

        st.info(context_texto)

    except FileNotFoundError:
        st.error("❌ Arquivo 'modelo_titanic_voting.pkl' não encontrado. Certifique-se de que o modelo está no diretório correto.")
    except Exception as e:
        st.error(f"❌ Erro ao processar previsão: {str(e)}")
