import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# Configuração da página
# ==========================================================
st.set_page_config(
    page_title="Controle de Qualidade - Peças Usinadas",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Controle de Qualidade de Peças Usinadas")
st.markdown(
    "Insira as características da peça inspecionada e clique em **Avaliar Peça** "
    "para que o modelo SVM classifique o resultado."
)

# ==========================================================
# Geração dos dados sintéticos + treino do modelo (cacheados)
# ==========================================================
@st.cache_data
def gerar_dados():
    """
    Gera dados sintéticos com 3 classes usando make_classification.
    As features geradas (X) são escalonadas para faixas realistas:
      - Peso_gramas   : ~ 200 a 800 g
      - Espessura_mm  : ~ 2.0 a 20.0 mm
    """
    # 3 classes, 2 features informativas
    X, y = make_classification(
        n_samples=1200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.6,
        flip_y=0.03,
        random_state=42,
    )

    # Re-escalonamento para faixas "reais" de peças usinadas
    peso = 200 + (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 600
    espessura = 2 + (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 18

    df = pd.DataFrame({
        "Peso_gramas": peso,
        "Espessura_mm": espessura,
        "Classe": y,
    })

    nomes = {0: "Aprovada", 1: "Retrabalho", 2: "Refugo"}
    df["Classe_Nome"] = df["Classe"].map(nomes)
    return df


@st.cache_resource
def treinar_modelo(df):
    """Treina um SVM com kernel RBF e retorna modelo, scaler e métricas."""
    X = df[["Peso_gramas", "Espessura_mm"]].values
    y = df["Classe"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelo = SVC(kernel="rbf", C=1.5, gamma="scale", random_state=42)
    modelo.fit(X_train_s, y_train)

    y_pred = modelo.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=["Aprovada", "Retrabalho", "Refugo"],
        output_dict=True,
    )
    acc = modelo.score(X_test_s, y_test)
    return modelo, scaler, cm, report, acc


df = gerar_dados()
modelo, scaler, cm, report, acc = treinar_modelo(df)

# ==========================================================
# Sidebar — Sliders de inspeção
# ==========================================================
st.sidebar.header("🔍 Inspeção da Peça")
peso = st.sidebar.slider(
    "Peso da peça (gramas)",
    min_value=200.0,
    max_value=800.0,
    value=500.0,
    step=1.0,
)
espessura = st.sidebar.slider(
    "Espessura da peça (mm)",
    min_value=2.0,
    max_value=20.0,
    value=10.0,
    step=0.1,
)

avaliar = st.sidebar.button("✅ Avaliar Peça", use_container_width=True)

# ==========================================================
# Resultado da classificação
# ==========================================================
st.subheader("📋 Resultado da Inspeção")

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Peso inserido", f"{peso:.1f} g")
col_info2.metric("Espessura inserida", f"{espessura:.2f} mm")
col_info3.metric("Acurácia do modelo (teste)", f"{acc*100:.2f}%")

if avaliar:
    entrada = np.array([[peso, espessura]])
    entrada_s = scaler.transform(entrada)
    pred = modelo.predict(entrada_s)[0]
    proba = modelo.predict_proba(entrada_s)[0]

    rotulos = {
        0: ("APROVADA", "#2e7d32", "✅", "Peça dentro das especificações."),
        1: ("RETRABALHO", "#f9a825", "⚠️", "Peça necessita de retrabalho."),
        2: ("REFUGO", "#c62828", "❌", "Peça deve ser descartada."),
    }
    nome, cor, icone, msg = rotulos[pred]

    # Cartão colorido
    st.markdown(
        f"""
        <div style="
            background-color:{cor};
            padding:22px;
            border-radius:12px;
            color:white;
            text-align:center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom:16px;
        ">
            <h2 style="margin:0;">{icone} {nome}</h2>
            <p style="margin:6px 0 0 0; font-size:16px;">{msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probabilidades
    st.markdown("**Probabilidades estimadas pelo modelo:**")
    prob_df = pd.DataFrame({
        "Classe": ["Aprovada", "Retrabalho", "Refugo"],
        "Probabilidade": proba,
    })
    st.dataframe(
        prob_df.style.format({"Probabilidade": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(prob_df.set_index("Classe"))

else:
    st.info("ℹ️ Ajuste os sliders na barra lateral e clique em **Avaliar Peça**.")

# ==========================================================
# Matriz de confusão e métricas
# ==========================================================
st.divider()
st.subheader("📊 Matriz de Confusão do Modelo SVM")

col_cm, col_metrics = st.columns([1.3, 1])

with col_cm:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Aprovada", "Retrabalho", "Refugo"],
        yticklabels=["Aprovada", "Retrabalho", "Refugo"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão (conjunto de teste)")
    plt.tight_layout()
    st.pyplot(fig)

with col_metrics:
    st.markdown("**Relatório de classificação:**")
    metrics_df = pd.DataFrame(report).T
    metrics_df = metrics_df.loc[
        ["Aprovada", "Retrabalho", "Refugo"], ["precision", "recall", "f1-score"]
    ]
    st.dataframe(
        metrics_df.style.format("{:.3f}"),
        use_container_width=True,
    )

# ==========================================================
# Visualização dos dados sintéticos (opcional)
# ==========================================================
with st.expander("🌐 Ver distribuição dos dados sintéticos"):
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    cores = {"Aprovada": "#2e7d32", "Retrabalho": "#f9a825", "Refugo": "#c62828"}
    for nome_classe, cor in cores.items():
        sub = df[df["Classe_Nome"] == nome_classe]
        ax2.scatter(
            sub["Peso_gramas"], sub["Espessura_mm"],
            s=18, alpha=0.6, label=nome_classe, color=cor,
        )
    ax2.scatter([peso], [espessura], marker="X", s=220,
                color="black", edgecolor="white", linewidth=1.2,
                label="Peça atual")
    ax2.set_xlabel("Peso (g)")
    ax2.set_ylabel("Espessura (mm)")
    ax2.set_title("Distribuição dos dados sintéticos")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
