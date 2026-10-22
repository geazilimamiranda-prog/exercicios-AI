import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="🔧",
    layout="centered"
)

st.title("🔧 Controle de Qualidade de Peças de Usinagem")
st.markdown("Classificação de peças com **SVM** (Support Vector Machine)")

# -------------------------------------------------
# 1. Geração de dados sintéticos e treinamento do modelo
# -------------------------------------------------
@st.cache_resource
def treinar_modelo():
    # Gera dados sintéticos com 2 features e 3 classes
    X, y = make_classification(
        n_samples=600,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        class_sep=1.5,
        random_state=42
    )

    # Ajusta as features para escalas realistas
    # Peso_gramas: aproximadamente 50 a 150 g
    # Espessura_mm: aproximadamente 1.0 a 5.0 mm
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (50, 150))
    X[:, 1] = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), (1.0, 5.0))

    # Mapeamento das classes
    classes = {0: "Aprovada", 1: "Retrabalho", 2: "Refugo"}
    y_labels = np.array([classes[i] for i in y])

    # Divide treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Treina o SVM
    modelo = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    modelo.fit(X_train, y_train)

    # Predições para matriz de confusão
    y_pred = modelo.predict(X_test)

    return modelo, X_test, y_test, y_pred, classes

modelo, X_test, y_test, y_pred, classes = treinar_modelo()

# -------------------------------------------------
# 2. Interface de inspeção
# -------------------------------------------------
st.subheader("📋 Inspeção da Peça")

col1, col2 = st.columns(2)

with col1:
    peso = st.slider(
        "Peso (gramas)",
        min_value=50.0,
        max_value=150.0,
        value=100.0,
        step=0.5,
        help="Peso da peça em gramas"
    )

with col2:
    espessura = st.slider(
        "Espessura (mm)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.05,
        help="Espessura da peça em milímetros"
    )

# Botão de avaliação
if st.button("Avaliar Peça", type="primary", use_container_width=True):
    # Predição
    entrada = np.array([[peso, espessura]])
    predicao = modelo.predict(entrada)[0]
    classe_predita = classes[predicao]
    
    # Probabilidades (opcional, para dar mais confiança)
    probs = modelo.predict_proba(entrada)[0]
    confianca = probs[predicao] * 100

    st.markdown("---")
    st.subheader("Resultado da Classificação")

    if classe_predita == "Aprovada":
        st.success(f"✅ **Peça APROVADA**  \nConfiança: {confianca:.1f}%")
        st.markdown(
            f"""
            <div style="background-color:#d4edda; padding:20px; border-radius:10px; 
                        border-left:8px solid #28a745; color:#155724;">
                <h3 style="margin:0;">🟢 Peça Aprovada</h3>
                <p style="margin:5px 0 0 0;">A peça está dentro dos padrões de qualidade.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif classe_predita == "Retrabalho":
        st.warning(f"⚠️ **Peça com RETRABALHO**  \nConfiança: {confianca:.1f}%")
        st.markdown(
            f"""
            <div style="background-color:#fff3cd; padding:20px; border-radius:10px; 
                        border-left:8px solid #ffc107; color:#856404;">
                <h3 style="margin:0;">🟡 Retrabalho Necessário</h3>
                <p style="margin:5px 0 0 0;">A peça precisa de ajustes antes de ser aprovada.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:  # Refugo
        st.error(f"❌ **Peça REFUGADA**  \nConfiança: {confianca:.1f}%")
        st.markdown(
            f"""
            <div style="background-color:#f8d7da; padding:20px; border-radius:10px; 
                        border-left:8px solid #dc3545; color:#721c24;">
                <h3 style="margin:0;">🔴 Peça Refugada</h3>
                <p style="margin:5px 0 0 0;">A peça está fora dos padrões e deve ser descartada.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# 3. Matriz de Confusão
# -------------------------------------------------
st.markdown("---")
st.subheader("📊 Matriz de Confusão do Modelo")

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Aprovada", "Retrabalho", "Refugo"]
)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Matriz de Confusão - SVM")
st.pyplot(fig)

# Informações adicionais
with st.expander("ℹ️ Sobre o modelo"):
    st.markdown("""
    - **Algoritmo**: Support Vector Machine (SVM) com kernel RBF  
    - **Features**: Peso (g) e Espessura (mm)  
    - **Classes**: Aprovada · Retrabalho · Refugo  
    - Dados gerados de forma sintética com `sklearn.datasets.make_classification`  
    - O modelo foi treinado com 75% dos dados e avaliado com 25%
    """)
