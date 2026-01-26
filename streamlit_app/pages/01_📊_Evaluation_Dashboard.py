#!/usr/bin/env python3
"""
RAGAS Evaluation Dashboard Page.
Visualizes evaluation results from CSV files.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from streamlit_app.shared.ui import inject_custom_css, render_footer

# Inject custom styles
inject_custom_css()


def load_metrics(file_path: str = "my_eval_metrics.csv") -> pd.DataFrame:
    """Load RAGAS metrics from CSV file and calculate composite score."""
    try:
        df = pd.read_csv(file_path)
        if not df.empty:
            df = calculate_composite_score(df)
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {file_path}")
        return pd.DataFrame()


def calculate_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a composite score based on weighted RAGAS metrics.
    Weights: Faithfulness (35%), Recall (30%), Correctness (20%), Precision (15%)
    """
    weights = {
        "faithfulness": 0.35,
        "context_recall": 0.30,
        "answer_correctness": 0.20,
        "context_precision": 0.15,
    }

    # Ensure all required columns exist
    for col in weights.keys():
        if col not in df.columns:
            df[col] = 0.0

    df["composite_score"] = (
        weights["faithfulness"] * df["faithfulness"]
        + weights["context_recall"] * df["context_recall"]
        + weights["answer_correctness"] * df["answer_correctness"]
        + weights["context_precision"] * df["context_precision"]
    )
    return df


def create_metrics_radar(df: pd.DataFrame) -> go.Figure:
    """Create radar chart for average metrics."""
    metrics_cols = [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]
    available_cols = [col for col in metrics_cols if col in df.columns]

    if not available_cols:
        return None

    avg_values = [df[col].mean() for col in available_cols]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=available_cols + [available_cols[0]],
            fill="toself",
            name="Média",
            line_color="rgb(99, 110, 250)",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Métricas RAGAS - Média Geral",
    )

    return fig


def create_metrics_bar(df: pd.DataFrame) -> go.Figure:
    """Create bar chart comparing metrics per question."""
    metrics_cols = [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]
    available_cols = [col for col in metrics_cols if col in df.columns]

    if not available_cols or "user_input" not in df.columns:
        return None

    df_plot = df.copy()
    df_plot["question_short"] = df_plot["user_input"].str[:50] + "..."

    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, col in enumerate(available_cols):
        fig.add_trace(
            go.Bar(
                name=col.replace("_", " ").title(),
                x=df_plot["question_short"],
                y=df_plot[col],
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        barmode="group",
        title="Métricas por Pergunta",
        xaxis_title="Pergunta",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_metrics_heatmap(
    df: pd.DataFrame, include_composite: bool = False
) -> go.Figure:
    """Create heatmap of all metrics, optionally including the composite score."""
    metrics_cols = [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]

    if include_composite and "composite_score" in df.columns:
        metrics_cols = ["composite_score"] + metrics_cols

    available_cols = [col for col in metrics_cols if col in df.columns]

    if not available_cols:
        return None

    # Create question labels
    q_labels = [f"Q{i+1}" for i in range(len(df))]

    # Format labels for the Y axis with clear separation
    labels = []
    for col in available_cols:
        if col == "composite_score":
            labels.append("⭐ COMPOSITE SCORE")
        else:
            labels.append(f"📊 {col.replace('_', ' ').title()}")

    # Add a visual separator in the data if composite is included
    # (Optional: we can just use the labels to differentiate)

    fig = go.Figure(
        data=go.Heatmap(
            z=df[available_cols].values.T,
            x=q_labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=df[available_cols].values.T.round(2),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title="Heatmap de Performance"
        + (" (com Composite Score)" if include_composite else ""),
        xaxis_title="Pergunta",
        yaxis_title="Métrica",
    )

    return fig


# Page Title
st.title("📊 RAGAS Evaluation Dashboard")
st.markdown("Visualização das métricas de avaliação do sistema RAG.")

# Sidebar
with st.sidebar:
    st.header("📁 Fonte de Dados")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = calculate_composite_score(df)
        st.success(f"Carregado: {uploaded_file.name}")
    else:
        # Try multiple paths (root and streamlit_app folder)
        possible_paths = [
            PROJECT_ROOT / "my_eval_metrics.csv",
            Path("my_eval_metrics.csv"),
        ]
        df = pd.DataFrame()
        for path in possible_paths:
            if path.exists():
                df = load_metrics(str(path))
                st.info(f"Usando: {path.name}")
                break

        if df.empty:
            st.warning("Nenhum arquivo de métricas encontrado.")

    st.divider()

    if not df.empty:
        st.metric("Total de Avaliações", len(df))

        # New Toggle for Heatmap in Sidebar or main
        st.header("🛠️ Opções de Visualização")
        view_composite_heatmap = st.toggle("Mostrar Composite no Heatmap", value=True)

if df.empty:
    st.warning("⚠️ Carregue um arquivo CSV com métricas RAGAS para visualizar.")
    st.markdown(
        """
    **Formato esperado do CSV:**
    - `user_input`: Pergunta do usuário
    - `retrieved_contexts`: Contextos recuperados
    - `response`: Resposta gerada
    - `reference`: Resposta de referência
    - `faithfulness`: Score de fidelidade (0-1)
    - `answer_correctness`: Score de correção (0-1)
    - `context_precision`: Precisão do contexto (0-1)
    - `context_recall`: Recall do contexto (0-1)
    """
    )
else:
    # Metrics summary
    st.header("📈 Resumo das Métricas")

    metrics_cols = [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]
    available_cols = [col for col in metrics_cols if col in df.columns]

    # Display Composite Score in a distinct section
    if "composite_score" in df.columns:
        composite_avg = df["composite_score"].mean()
        with st.container(border=True):
            st.markdown("### 🏆 Avaliação Executiva")
            col_cs, col_txt = st.columns([1, 2])
            with col_cs:
                st.metric("COMPOSITE SCORE", f"{composite_avg:.2%}")
            with col_txt:
                st.info(
                    """
                    **Métrica Customizada**: Média ponderada (Fidelidade 35%, Recall 30%, 
                    Correção 20%, Precisão 15%) para reduzir ruído de falsos negativos.
                    """
                )
        st.divider()

    st.subheader("📊 Métricas RAGAS (Originais)")
    # Display average metrics as cards
    available_cols = [col for col in metrics_cols if col in df.columns]
    cols = st.columns(len(available_cols))
    for i, col in enumerate(available_cols):
        with cols[i]:
            avg_val = df[col].mean()
            delta = df[col].std()
            st.metric(
                label=col.replace("_", " ").title(),
                value=f"{avg_val:.2%}",
                delta=f"±{delta:.2%}",
                delta_color="off",
            )

    st.divider()

    # Charts section
    col1, col2 = st.columns(2)

    with col1:
        radar_fig = create_metrics_radar(df)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)

    with col2:
        heatmap_fig = create_metrics_heatmap(
            df, include_composite=view_composite_heatmap
        )
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)

    bar_fig = create_metrics_bar(df)
    if bar_fig:
        st.plotly_chart(bar_fig, use_container_width=True)

    st.divider()

    # Detailed data
    st.header("📋 Dados Detalhados")

    with st.expander("Ver tabela completa", expanded=False):
        st.dataframe(df, use_container_width=True)

    # Individual question analysis
    st.header("🔍 Análise por Pergunta")

    if "user_input" in df.columns:
        questions = df["user_input"].tolist()
        selected_idx = st.selectbox(
            "Selecione uma pergunta:",
            range(len(questions)),
            format_func=lambda x: f"Q{x+1}: {questions[x][:80]}...",
        )

        if selected_idx is not None:
            row = df.iloc[selected_idx]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("❓ Pergunta")
                st.write(row.get("user_input", "N/A"))

                st.subheader("💬 Resposta Gerada")
                st.write(row.get("response", "N/A"))

            with col2:
                st.subheader("✅ Resposta de Referência")
                st.write(row.get("reference", "N/A"))

                st.subheader("📊 Métricas")
                if "composite_score" in row:
                    st.write(f"🏆 **Composite Score**: {row['composite_score']:.2%}")

                for col in available_cols:
                    val = row.get(col, 0)
                    color = "🟢" if val >= 0.7 else "🟡" if val >= 0.4 else "🔴"
                    st.write(f"{color} **{col.replace('_', ' ').title()}**: {val:.2%}")

            if "retrieved_contexts" in df.columns:
                with st.expander("📚 Contextos Recuperados"):
                    contexts = row.get("retrieved_contexts", "[]")
                    st.code(contexts, language="json")

render_footer()
