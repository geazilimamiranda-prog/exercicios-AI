import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

CLASSES = ["Aprovada", "Retrabalho", "Refugo"]
PESO_RANGE = (400.0, 600.0)      # gramas
ESPESSURA_RANGE = (8.0, 14.0)    # mm

st.set_page_config(page_title="Controle de Qualidade - Usinagem", page_icon="🔩", layout="wide")


# ----------------------------------------------------------------------
# Dados sintéticos + treino do modelo
# ----------------------------------------------------------------------
@st.cache_resource
def treinar_modelo():
    # 1) Dados sintéticos com 3 classes e 2 features
    X, y = make_classification(
        n_samples=900,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.4,
        flip_y=0.03,
        random_state=42,
    )

    # 2) Converte as features para a escala física (gramas e mm)
    X = np.column_stack([
        MinMaxScaler(PESO_RANGE).fit_transform(X[:, [0]]).ravel(),
        MinMaxScaler(ESPESSURA_RANGE).fit_transform(X[:, [1]]).ravel(),
    ])

    # 3) Atribui nomes às classes de forma coerente: o grupo mais próximo da
    #    peça "nominal" (centro da faixa) é Aprovada; o mais distante é Refugo.
    nominal = np.array([np.mean(PESO_RANGE), np.mean(ESPESSURA_RANGE)])
    escala = np.array([PESO_RANGE[1] - PESO_RANGE[0], ESPESSURA_RANGE[1] - ESPESSURA_RANGE[0]])
    dist = [np.linalg.norm((X[y == k].mean(axis=0) - nominal) / escala) for k in range(3)]
    ordem = np.argsort(dist)                      # [mais perto, meio, mais longe]
    mapa = {int(ordem[0]): 0, int(ordem[1]): 1, int(ordem[2]): 2}
    y = np.array([mapa[int(v)] for v in y])

    df = pd.DataFrame(X, columns=["Peso_gramas", "Espessura_mm"])
    df["Classe"] = [CLASSES[i] for i in y]

    # 4) Treino/teste e SVM
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42),
    )
    modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_te)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    acc = accuracy_score(y_te, y_pred)
    return modelo, df, cm, acc


modelo, df, cm, acc = treinar_modelo()

# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("🔩 Controle de Qualidade de Peças Usinadas")
st.caption("Classificação automática com SVM (scikit-learn) a partir de peso e espessura.")

col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    st.subheader("Dados da peça inspecionada")
    peso = st.slider(
        "Peso (g)",
        min_value=float(round(df["Peso_gramas"].min(), 1)),
        max_value=float(round(df["Peso_gramas"].max(), 1)),
        value=float(np.mean(PESO_RANGE)),
        step=0.5,
    )
    espessura = st.slider(
        "Espessura (mm)",
        min_value=float(round(df["Espessura_mm"].min(), 2)),
        max_value=float(round(df["Espessura_mm"].max(), 2)),
        value=float(np.mean(ESPESSURA_RANGE)),
        step=0.05,
    )

    if st.button("Avaliar Peça", type="primary", use_container_width=True):
        entrada = np.array([[peso, espessura]])
        idx = int(modelo.predict(entrada)[0])
        probs = modelo.predict_proba(entrada)[0]
        st.session_state["resultado"] = {
            "classe": CLASSES[idx],
            "probs": probs,
            "peso": peso,
            "espessura": espessura,
        }

    res = st.session_state.get("resultado")
    if res:
        msg = f"Peça **{res['classe']}** (peso {res['peso']:.1f} g · espessura {res['espessura']:.2f} mm)"
        if res["classe"] == "Aprovada":
            st.success(f"✅ {msg} — liberada para expedição.")
        elif res["classe"] == "Retrabalho":
            st.warning(f"⚠️ {msg} — encaminhar para retrabalho.")
        else:
            st.error(f"❌ {msg} — descartar como refugo.")

        st.write("Confiança do modelo:")
        st.dataframe(
            pd.DataFrame({"Classe": CLASSES, "Probabilidade": [f"{p:.1%}" for p in res["probs"]]}),
            hide_index=True,
            use_container_width=True,
        )

with col_out:
    st.subheader("Matriz de confusão (conjunto de teste)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    st.pyplot(fig)
    st.metric("Acurácia no teste", f"{acc:.1%}")

# ----------------------------------------------------------------------
# Região de decisão do modelo + peça avaliada
# ----------------------------------------------------------------------
st.divider()
st.subheader("Mapa de decisão do SVM")

cores = {"Aprovada": "#2e9e4f", "Retrabalho": "#e6b400", "Refugo": "#d33a3a"}
x_min, x_max = df["Peso_gramas"].min() - 5, df["Peso_gramas"].max() + 5
y_min, y_max = df["Espessura_mm"].min() - 0.2, df["Espessura_mm"].max() + 0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
zz = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
from matplotlib.colors import ListedColormap
ax2.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5],
             colors=[cores[c] for c in CLASSES], alpha=0.18)
for c in CLASSES:
    sub = df[df["Classe"] == c]
    ax2.scatter(sub["Peso_gramas"], sub["Espessura_mm"], s=10, color=cores[c], label=c, alpha=0.8)
res = st.session_state.get("resultado")
if res:
    ax2.scatter(res["peso"], res["espessura"], s=220, marker="*", color="black",
                edgecolor="white", linewidth=1.2, label="Peça avaliada", zorder=5)
ax2.set_xlabel("Peso (g)")
ax2.set_ylabel("Espessura (mm)")
ax2.legend(loc="best", fontsize=8)
st.pyplot(fig2)

with st.expander("Ver dados sintéticos"):
    st.dataframe(df, use_container_width=True)
