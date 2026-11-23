
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Sri Lanka Food Insecurity | Ensemble ML",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation-only styling; model and data logic remain unchanged.
st.markdown(
    """
    <style>
    .stApp { background: #F6F8FC; color: #182B43; }
    [data-testid="stHeader"] { background: #F6F8FC !important; }
    .block-container { padding-top: 2.2rem; max-width: 1420px; }
    h1, h2, h3 { color: #172B45 !important; letter-spacing: -0.025em; }
    h1 { font-weight: 750 !important; }
    [data-testid="stSidebar"] { background: #EDF2F8 !important; border-right: 1px solid #DDE5EE; }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.7rem; }
    [data-testid="stSidebar"] h1 { font-size: 1.8rem !important; font-weight: 850 !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #708299 !important; }
    /* Navigation uses real Streamlit buttons, not fragile hidden radio inputs. */
    [data-testid="stSidebar"] div.stButton > button {
        width: 100%; justify-content: flex-start; text-align: left;
        min-height: 52px; border-radius: 10px; padding: 0.65rem 1rem;
        font-size: 1.25rem !important; font-weight: 800 !important; box-shadow: none; transition: background .15s ease;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background: transparent !important; color: #34465F !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background: #DEE7F2 !important; color: #173B63 !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #173B63 !important; color: #FFFFFF !important;
        border: 1px solid #173B63 !important;
    }
    [data-testid="stSidebar"] div.stButton > button > div,
    [data-testid="stSidebar"] div.stButton > button [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        flex: 1 1 auto !important;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] div.stButton > button p {
        color: inherit !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        text-align: left !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    [data-testid="stSidebar"] .ensemble-nav-heading {
        color: #173B63;
        font-size: 1.5rem;
        font-weight: 850;
        letter-spacing: 0.02em;
        text-align: left;
        margin: 0.5rem 0 1rem;
    }
    /* White metric cards with subtle borders. */
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #DFE7F0;
        border-radius: 13px; padding: 1.15rem 1.2rem;
        box-shadow: 0 2px 8px rgba(22, 43, 69, 0.035);
    }
    [data-testid="stMetricLabel"] { color: #65768B !important; }
    [data-testid="stMetricValue"] { color: #172B45 !important; font-weight: 750; }
    /* Scope to native Streamlit tabs only; preserve every other component. */
    .stTabs [data-baseweb="tab-list"],
    [data-testid="stTabs"] [role="tablist"] {
        background: transparent !important;
        border-bottom: 1px solid #CBD5E1 !important;
        box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    .stTabs [data-baseweb="tab"],
    [data-testid="stTabs"] [role="tab"] {
        background: transparent !important;
        color: #34465F !important;
        -webkit-text-fill-color: #34465F !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    [data-testid="stTabs"] [role="tab"] p,
    [data-testid="stTabs"] [role="tab"] span {
        color: #34465F !important;
        -webkit-text-fill-color: #34465F !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #173B63 !important;
        -webkit-text-fill-color: #173B63 !important;
        border-bottom: 3px solid #2C9A91 !important;
        font-weight: 750 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] span {
        color: #173B63 !important;
        -webkit-text-fill-color: #173B63 !important;
        opacity: 1 !important;
    }
    /* Light table rendered directly in HTML so no external theme file is needed. */
    .ensemble-table-wrap { overflow-x: auto; border: 1px solid #DFE7F0;
        border-radius: 10px; background: white; max-height: 500px; overflow-y: auto; }
    .ensemble-table { width: 100%; border-collapse: collapse;
        background: #FFFFFF; color: #172B45; font-size: 0.94rem; }
    .ensemble-table th { background: #EDF2F8; color: #173B63;
        font-weight: 700; text-align: left; position: sticky; top: 0; }
    .ensemble-table th, .ensemble-table td { padding: 0.65rem 0.8rem;
        border-bottom: 1px solid #E5EBF2; }
    .ensemble-table tbody tr:nth-child(even) { background: #F8FAFD; }
    .ensemble-table tbody tr:hover { background: #EFF7F7; }
    /* Select controls remain light even if a previous dark Streamlit theme was set. */
    [data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label {
        color: #34465F !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] input,
    [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #172B45 !important;
        -webkit-text-fill-color: #172B45 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill: #34465F !important;
    }
    /* Help icons only: white circular background with a blue question mark. */
    .stApp [data-testid="stTooltipIcon"] {
        background: #FFFFFF !important;
        border-radius: 50% !important;
        color: #173B63 !important;
    }
    .stApp [data-testid="stTooltipIcon"] svg {
        color: #173B63 !important;
        stroke: #173B63 !important;
        fill: #FFFFFF !important;
    }
    .stApp [data-testid="stTooltipIcon"] svg circle {
        fill: #FFFFFF !important;
        stroke: #173B63 !important;
    }
    .stApp [data-testid="stTooltipIcon"] svg path,
    .stApp [data-testid="stTooltipIcon"] svg line {
        stroke: #173B63 !important;
    }
    /* Prediction submit button only: match the dashboard navy theme. */
    .stApp [data-testid="stFormSubmitButton"] button,
    .stApp [data-testid="stFormSubmitButton"] button[kind],
    .stApp [data-testid="stFormSubmitButton"] button[data-testid^="stBaseButton"] {
        background: #173B63 !important;
        background-color: #173B63 !important;
        border-color: #173B63 !important;
        color: #FFFFFF !important;
    }
    .stApp [data-testid="stFormSubmitButton"] button:hover,
    .stApp [data-testid="stFormSubmitButton"] button[kind]:hover {
        background: #285C87 !important;
        background-color: #285C87 !important;
        border-color: #285C87 !important;
    }
    .stApp [data-testid="stFormSubmitButton"] button *,
    .stApp [data-testid="stFormSubmitButton"] button p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    /* Primary prediction button: navy rather than the old red theme. */
    [data-testid="stFormSubmitButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] {
        background: #173B63 !important;
        border: 1px solid #173B63 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:hover {
        background: #285C87 !important;
        border-color: #285C87 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stFormSubmitButton"] button[kind="primary"]:focus,
    [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:focus {
        box-shadow: 0 0 0 2px #7DD3C7 !important;
    }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }
    /* Keep interactive form widgets readable on light surfaces. */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important; color: #172B45 !important;
        border-color: #D4DEEA !important;
    }
    /* Force ONLY input widgets back to the original light appearance. */
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] input,
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] input {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #172B45 !important;
        -webkit-text-fill-color: #172B45 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stNumberInput"] [data-baseweb="input"] * {
        color: #172B45 !important;
        -webkit-text-fill-color: #172B45 !important;
    }
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button[kind],
    [data-testid="stNumberInput"] [data-baseweb="input"] button {
        background: #F1F5FA !important;
        background-color: #F1F5FA !important;
        border-color: #D4DEEA !important;
        color: #173B63 !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background: #E3EBF4 !important;
    }
    [data-testid="stNumberInput"] button svg,
    [data-testid="stSelectbox"] svg {
        fill: #173B63 !important;
        color: #173B63 !important;
    }

    /* Widget-only light surfaces: match Streamlit's outer BaseWeb container
       as well as the inner control (previous rules missed the outer layer). */
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"],
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div,
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"],
    .stApp [data-testid="stNumberInput"] div[data-baseweb="input"],
    .stApp [data-testid="stNumberInput"] div[data-baseweb="input"] > div,
    .stApp [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #172B45 !important;
        -webkit-text-fill-color: #172B45 !important;
    }
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"],
    .stApp [data-testid="stNumberInput"] div[data-baseweb="input"] {
        border: 1px solid #D4DEEA !important;
        border-radius: 9px !important;
    }
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    .stApp [data-testid="stNumberInput"] div[data-baseweb="input"] * {
        color: #172B45 !important;
        -webkit-text-fill-color: #172B45 !important;
    }
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] svg,
    .stApp [data-testid="stNumberInput"] svg {
        fill: #34465F !important;
        color: #34465F !important;
    }
    .stApp [data-testid="stNumberInput"] button {
        background: #EDF2F8 !important;
        background-color: #EDF2F8 !important;
        border-color: #D4DEEA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def light_table(df, max_rows=100):
    """Display a readable light table without depending on Streamlit's global theme."""
    view = df.head(max_rows)
    html_table = view.to_html(index=False, escape=True, classes="ensemble-table",
                              border=0, float_format=lambda x: f"{x:.4f}")
    st.markdown('<div class="ensemble-table-wrap">' + html_table + '</div>',
                unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing the first {max_rows:,} of {len(df):,} rows.")


MODEL_KEYS = {
    "Decision Tree": "decision_tree",
    "Bagging": "bagging",
    "Random Forest": "random_forest",
    "AdaBoost": "adaboost",
    "XGBoost": "xgboost",
    "Stacking": "stacking",
    "RF + ANN Hybrid": "rf_ann_hybrid",
}


@st.cache_data
def load_data():
    dataset = pd.read_csv(
        RESULTS_DIR / "cleaned_household_data.csv",
        low_memory=False,
    )
    comparison = pd.read_csv(
        RESULTS_DIR / "model_comparison.csv"
    )
    predictions = pd.read_csv(
        RESULTS_DIR / "test_predictions.csv"
    )

    return dataset, comparison, predictions


def display_chart(filename, caption):
    path = RESULTS_DIR / filename

    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart not found: {filename}")



def display_metric_comparison(comparison):
    """Five evaluation metrics in the dashboard's original navy/blue/teal palette."""
    metric_colors = {
        "Accuracy": "#173B63",
        "Precision": "#32749A",
        "Recall": "#27968E",
        "F1 Score": "#5B83B1",
        "ROC AUC": "#89AABD",
    }
    metrics = list(metric_colors)
    long_data = comparison.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Evaluation Metric",
        value_name="Score",
    )
    model_order = comparison.sort_values("F1 Score", ascending=False)["Model"].tolist()
    fig = px.bar(
        long_data,
        x="Model",
        y="Score",
        color="Evaluation Metric",
        barmode="group",
        category_orders={"Model": model_order, "Evaluation Metric": metrics},
        color_discrete_map=metric_colors,
        title="Comparison of Ensemble Learning Models",
    )
    fig.update_layout(
        height=510,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#34465F", size=13),
        title=dict(font=dict(color="#173B63", size=19), x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=25, t=110, b=65),
        yaxis=dict(range=[0, 1], title="Score", gridcolor="#E4EBF3", zeroline=False),
        xaxis=dict(title=None, showgrid=False),
        bargap=0.23,
    )
    fig.update_traces(
        hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#34465F"))
    st.plotly_chart(fig, use_container_width=True, theme=None)


def overview_page(dataset, comparison):
    st.title("Household Food Insecurity Prediction")
    st.subheader("Ensemble Learning Dashboard — Sri Lanka")

    st.markdown(
        "Comparative analysis of **Bagging, Boosting, and "
        "Stacking** ensemble learning techniques using "
        "household survey data."
    )

    total = len(dataset)
    insecure = int(dataset["food_insecure"].sum())
    secure = total - insecure

    best_f1 = comparison.loc[
        comparison["F1 Score"].idxmax()
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Households Analyzed", f"{total:,}")
    col2.metric("Food Insecurity Experience", f"{insecure:,}")
    col3.metric("No Reported Experience", f"{secure:,}")
    col4.metric("Best F1 Model", best_f1["Model"])

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Target Class Distribution")

        counts = (
            dataset["food_insecure"]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Households")
        )

        counts["Class"] = counts["Class"].map({
            0: "No Reported Experience",
            1: "Food Insecurity Experience",
        })

        fig = px.pie(
            counts,
            names="Class",
            values="Households",
            hole=0.45,
            color_discrete_sequence=["#173B63", "#2C9A91"],
        )

        fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#34465F"))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with right:
        st.subheader("Model F1 Scores")

        fig = px.bar(
            comparison.sort_values("F1 Score"),
            x="F1 Score",
            y="Model",
            orientation="h",
            color="F1 Score",
            color_continuous_scale=["#B9D6E3", "#4C819D", "#173B63"],
            text="F1 Score",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )
        fig.update_layout(
            xaxis_range=[0, 1],
            showlegend=False,
        )

        fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#34465F"))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.info(
        "The target indicates whether a household reported "
        "at least one of six food insecurity experiences "
        "during the previous 12 months. It is not an "
        "official food insecurity severity score."
    )


def data_overview_page(dataset):
    st.title("Data Overview")
    st.write(
        "How six survey modules became a household-level dataset for ensemble learning."
    )

    original_count = 6847
    final_count = len(dataset)
    excluded_count = original_count - final_count
    insecure_count = int((dataset["food_insecure"] == 1).sum())
    no_experience_count = int((dataset["food_insecure"] == 0).sum())

    a, b, c, d = st.columns(4)
    a.metric("Original households", f"{original_count:,}")
    b.metric("Households modeled", f"{final_count:,}")
    c.metric("Records excluded", f"{excluded_count:,}")
    d.metric("Predictor columns", f"{len(dataset.columns) - 1:,}")

    st.subheader("Data preparation")
    flow = [
        ("1", "Survey modules", "Identification, housing, assets, food security, debt, and savings"),
        ("2", "Household-level merge", "Matched survey records using the household identifier"),
        ("3", "Target creation", "At least one Yes response to six food insecurity questions"),
        ("4", "Missing-response handling", "Excluded households without a usable binary target"),
        ("5", "Modeling dataset", f"{final_count:,} households available for training and evaluation"),
    ]
    for step, heading, description in flow:
        with st.container(border=True):
            st.markdown(f"**{step}. {heading}**")
            st.caption(description)

    st.subheader("Target class distribution")
    class_data = pd.DataFrame({
        "Outcome": ["Food insecurity experience", "No reported experience"],
        "Households": [insecure_count, no_experience_count],
    })
    fig = px.bar(
        class_data,
        x="Households",
        y="Outcome",
        orientation="h",
        text="Households",
        color="Outcome",
        color_discrete_map={
            "Food insecurity experience": "#247F92",
            "No reported experience": "#91B6C1",
        },
    )
    fig.update_layout(
        showlegend=False, height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=25, t=10, b=10),
        xaxis_title="Number of households", yaxis_title=None,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#34465F"))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.caption(
        f"Among the {final_count:,} retained households, "
        f"{insecure_count:,} ({insecure_count / final_count:.1%}) reported at least one "
        f"food insecurity experience, while {no_experience_count:,} "
        f"({no_experience_count / final_count:.1%}) did not."
    )

    st.subheader("Training and test data")
    train_n = round(final_count * 0.8)
    test_n = final_count - train_n
    t1, t2 = st.columns(2)
    t1.metric("Training set (80%)", f"{train_n:,}")
    t2.metric("Test set (20%)", f"{test_n:,}")
    st.caption(
        "The project uses an 80/20 stratified train-test split. "
        "Model evaluation is performed on held-out records."
    )

    st.subheader("Target definition")
    st.markdown(
        "A household is labeled **1 — Food insecurity experience** if it reported "
        "at least one of the following during the past 12 months: worrying about food, "
        "being unable to eat nutritious food, limiting food variety, skipping meals, "
        "eating less than needed, or running out of food. "
        "A household is labeled **0 — No reported experience** when all six responses "
        "are No. Missing, refused, or unknown responses require careful handling."
    )
    st.info(
        "This is a project-specific binary outcome, not an official food insecurity "
        "severity classification. Excluded records and target coding should be "
        "documented and checked against the survey codebook."
    )

    st.subheader("Source")
    st.markdown(
        "BRIGHT Integrated Household Survey of Sri Lanka (2024–2025), IFPRI. "
        "[Dataset and documentation](https://doi.org/10.7910/DVN/FH0DSN)"
    )


def comparison_page(comparison):
    st.title("Model Performance Comparison")

    st.markdown(
        "Compare seven machine learning classifiers evaluated "
        "on the same held-out test dataset."
    )

    formatted = comparison.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda x: f"{x:.2%}")
    if "ROC AUC" in formatted.columns:
        formatted["ROC AUC"] = formatted["ROC AUC"].map(lambda x: f"{x:.3f}")
    light_table(formatted)

    metric = st.selectbox(
        "Select evaluation metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
        ],
    )

    fig = px.bar(
        comparison.sort_values(metric),
        x="Model",
        y=metric,
        color="Model",
        color_discrete_sequence=["#173B63", "#276084", "#2C9A91", "#5C87A5", "#769DB5", "#A4BBCB"],
        text=metric,
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    fig.update_layout(
        yaxis_range=[0, 1],
        showlegend=False,
    )

    fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#34465F"))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    best = comparison.loc[comparison[metric].idxmax()]

    # Display the selected metric in an easy-to-read format.
    metric_value = (
        f"{best[metric]:.3f}" if metric == "ROC AUC"
        else f"{best[metric]:.2%}"
    )
    st.success(f"Best {metric}: {best['Model']} — {metric_value}")

    display_metric_comparison(comparison)

    # Derive the interpretation from the saved evaluation results,
    # so it remains correct if models are retrained.
    best_accuracy = comparison.loc[comparison["Accuracy"].idxmax()]
    best_f1 = comparison.loc[comparison["F1 Score"].idxmax()]
    best_auc = comparison.loc[comparison["ROC AUC"].idxmax()]

    interpretation = (
        f"{best_accuracy['Model']} achieved the highest accuracy "
        f"({best_accuracy['Accuracy']:.2%}), while "
        f"{best_f1['Model']} achieved the highest F1 score "
        f"({best_f1['F1 Score']:.2%}). "
        f"{best_auc['Model']} recorded the highest ROC-AUC "
        f"({best_auc['ROC AUC']:.3f}). "
        "These results indicate that the strongest model depends on "
        "the evaluation metric selected."
    )
    st.markdown(
        f'''<div style="background-color:#EAF3F6;"
                 "border-left:4px solid #299C95;"
                 "border-radius:8px;padding:16px 20px;"
                 "margin-top:18px;color:#193B63;"
                 "font-size:15px;line-height:1.7;">
            <strong>Model Performance Interpretation</strong><br>
            {interpretation}
        </div>''',
        unsafe_allow_html=True,
    )


def visualization_page(comparison):
    st.title("Model Evaluation Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "ROC Curves",
        "Confusion Matrices",
        "Metric Comparison",
        "Feature Importance",
    ])

    with tab1:
        st.subheader("Receiver Operating Characteristic Curves")
        display_chart(
            "roc_curves.png",
            "ROC curves of all seven classifiers",
        )
        st.caption(
            "A larger ROC-AUC indicates better discrimination "
            "between the two target classes."
        )

    with tab2:
        st.subheader("Confusion Matrices")
        display_chart(
            "confusion_matrices.png",
            "Confusion matrices for all seven models",
        )

    with tab3:
        st.subheader("Evaluation Metric Comparison")
        display_metric_comparison(comparison)

    with tab4:
        st.subheader("Feature Importance Analysis")
        st.write(
            "Explore the features used by the trained models. "
            "Importance does not establish causation."
        )

        selected_model = st.selectbox(
            "Select model for feature importance",
            ["XGBoost", "Random Forest", "Decision Tree"],
        )
        feature_files = {
            "XGBoost": "xgboost",
            "Random Forest": "random_forest",
            "Decision Tree": "decision_tree",
        }
        key = feature_files[selected_model]
        display_chart(
            f"{key}_feature_importance.png",
            f"Top 15 features — {selected_model}",
        )

        csv_path = RESULTS_DIR / f"{key}_feature_importance.csv"
        if csv_path.exists():
            importance_df = pd.read_csv(csv_path)
            st.subheader("Feature Importance Values")
            light_table(importance_df.head(20))
        else:
            st.warning(f"Feature importance file not found: {csv_path.name}")

        st.info(
            "These scores describe model behavior, not causal relationships. "
            "Interpret survey fields using their original question labels."
        )


def prediction_page(predictions, dataset):
    st.title("Prediction Explorer")
    existing_tab, new_tab = st.tabs(["Existing Predictions", "New Household Prediction"])
    with existing_tab:
        saved_prediction_tab(predictions)
    with new_tab:
        new_household_prediction_tab(dataset)


@st.cache_resource(show_spinner=False)
def load_saved_model(model_path):
    return joblib.load(model_path)


# Verified against the value labels in the BRIGHT household identification .dta file.
DISTRICT_CODES = {
    "Colombo": 1, "Gampaha": 2, "Kalutara": 3,
    "Galle": 4, "Hambantota": 5, "Matara": 6,
    "Jaffna": 7, "Kilinochchi": 8, "Mannar": 9,
    "Mullaitivu": 10, "Vavuniya": 11,
    "Ampara": 12, "Batticaloa": 13, "Trincomalee": 14,
    "Anuradhapura": 15, "Polonnaruwa": 16,
    "Kandy": 17, "Matale": 18, "Nuwara Eliya": 19,
    "Kurunegala": 20, "Puttalam": 21,
    "Badulla": 22, "Monaragala": 23,
    "Kegalle": 24, "Ratnapura": 25,
}
PROVINCE_CODES = {
    "Western": 1, "Southern": 2, "Northern": 3, "Eastern": 4,
    "North Central": 5, "Central": 6, "North Western": 7,
    "Uva": 8, "Sabaragamuwa": 9,
}
DISTRICT_PROVINCES = {
    "Colombo": "Western", "Gampaha": "Western", "Kalutara": "Western",
    "Galle": "Southern", "Hambantota": "Southern", "Matara": "Southern",
    "Jaffna": "Northern", "Kilinochchi": "Northern", "Mannar": "Northern",
    "Mullaitivu": "Northern", "Vavuniya": "Northern",
    "Ampara": "Eastern", "Batticaloa": "Eastern", "Trincomalee": "Eastern",
    "Anuradhapura": "North Central", "Polonnaruwa": "North Central",
    "Kandy": "Central", "Matale": "Central", "Nuwara Eliya": "Central",
    "Kurunegala": "North Western", "Puttalam": "North Western",
    "Badulla": "Uva", "Monaragala": "Uva",
    "Kegalle": "Sabaragamuwa", "Ratnapura": "Sabaragamuwa",
}


def friendly_feature_label(column):
    """UI labels; the actual model still receives its original feature names."""
    known = {
        "a_01_province": "Province",
        "a_02_district": "District",
        "a_05": "Residential area (urban / rural / estate)",
        "hh_type": "Interview type",
        "c_respondent_housing": "Person answering housing questions",
        "c_present_housing": "Person present during housing interview",
        "d_respondent_hhassets": "Person answering household asset questions",
        "c1_01": "Type of main dwelling",
        "c1_01_other": "Other dwelling type (please specify)",
        "c1_02": "Number of bedrooms",
        "c1_03": "Total dwelling floor area (survey category)",
        "c1_04": "Main material used for house walls",
    }
    if column in known:
        return known[column]
    import re
    match = re.fullmatch(r"([a-z]+)(\d*)_(\d+)(?:_(.*))?", column.lower())
    module_names = {"a": "Household identification", "c": "Housing", "d": "Household assets",
                    "n": "Financial information", "n1": "Debt", "n2": "Savings"}
    if match:
        prefix, section, number, suffix = match.groups()
        module = module_names.get(prefix + section, module_names.get(prefix, "Survey"))
        question = f"{prefix.upper()}{section}.{number}"
        if suffix:
            return f"{module} — {question}: {suffix.replace('_', ' ').title()}"
        return f"{module} — Question {question}"
    return column.replace("_", " ").strip().title()




@st.cache_data(show_spinner=False)
def load_survey_labels():
    """Read real question and response labels from the original Stata metadata."""
    import pyreadstat

    files = {
        "identification": "mod_a_household_identification.dta",
        "housing": "mod_c_housing.dta",
        "assets": "mod_d_assets.dta",
        "food_security": "mod_l1_food_security_w.dta",
        "debt": "mod_n1_debt.dta",
        "savings": "mod_n2_savings.dta",
    }
    labels, questions = {}, {}
    for module, filename in files.items():
        path = BASE_DIR / "data" / filename
        if not path.exists():
            continue
        try:
            _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
            for col, choices in meta.variable_value_labels.items():
                labels.setdefault(col, choices)
                labels.setdefault(f"{module}_{col}", choices)
            for col, question in meta.column_names_to_labels.items():
                if question:
                    questions.setdefault(col, question)
                    questions.setdefault(f"{module}_{col}", question)
        except Exception:
            # Keep the rest of the dashboard usable if one metadata file fails.
            continue
    return labels, questions


def display_survey_choice(value, mapping):
    """Show the real response meaning while retaining the original stored code."""
    if pd.isna(value):
        return "Not recorded"
    try:
        key = float(value)
    except (ValueError, TypeError):
        key = value
    meaning = mapping.get(key)
    if meaning is None:
        meaning = mapping.get(value)
    if meaning is None:
        return f"Survey code {value} (meaning not available)"
    meaning = str(meaning).strip()
    # Make roster references understandable without changing the stored survey code.
    if meaning.lower().startswith("hh roster name"):
        suffix = meaning[len("HH Roster Name"):].strip()
        return f"Household member {suffix}" if suffix else "Household member"
    return meaning if meaning else f"Survey code {value}"


def survey_field_widget(col, series, default, widget_key, labels, questions, other_dwelling_enabled=True):
    """Choose a suitable control; return the raw code expected by the model."""
    name = friendly_feature_label(col)
    question = questions.get(col)
    help_text = f"Original survey field: {col}"
    if question:
        help_text += f". Survey question: {question}"

    # Bedrooms are a count, not a coded category.
    if col == "c1_02":
        observed = pd.to_numeric(series, errors="coerce")
        observed = observed[observed.ge(0) & observed.le(100)]
        typical = int(round(observed.median())) if not observed.empty else 0
        return st.number_input("Number of bedrooms", min_value=0, max_value=100,
                               value=typical, step=1, key=widget_key, help=help_text)

    if col == "c1_01_other":
        return st.text_input(
            "Other dwelling type (if applicable)",
            value=str(default) if other_dwelling_enabled and pd.notna(default) else "",
            key=widget_key,
            help=help_text + ". Complete only if the main dwelling type is Other.",
            disabled=not other_dwelling_enabled,
            placeholder="Specify dwelling type" if other_dwelling_enabled else "Not applicable",
        )

    mapping = labels.get(col, {})
    if mapping:
        # Offer only codes that are present in the model's reference data.
        options = list(series.unique())
        options = sorted(options, key=lambda v: str(display_survey_choice(v, mapping)).lower())
        if options:
            selected_index = next((i for i, v in enumerate(options) if v == default), 0)
            return st.selectbox(name, options, index=selected_index, key=widget_key,
                                format_func=lambda v: display_survey_choice(v, mapping),
                                help=help_text)

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 15:
            options = sorted(series.unique(), key=lambda v: str(v))
            if options:
                idx = next((i for i, v in enumerate(options) if v == default), 0)
                return st.selectbox(name, options, index=idx, key=widget_key,
                                    format_func=lambda v: f"Survey code {v} (label unavailable)",
                                    help=help_text + ". Verify these codes in the questionnaire.")
        numeric_default = float(default) if pd.notna(default) else 0.0
        return st.number_input(name, value=numeric_default, key=widget_key,
                               help=help_text)

    options = list(series.unique()) if series.nunique() <= 15 else []
    if options:
        options = sorted(options, key=str)
        idx = next((i for i, v in enumerate(options) if v == default), 0)
        return st.selectbox(name, options, index=idx, key=widget_key, help=help_text)
    return st.text_input(name, value=str(default) if pd.notna(default) else "",
                         key=widget_key, help=help_text)


def new_household_prediction_tab(dataset):
    st.markdown(
        "Enter household characteristics to generate a new prediction using a saved "
        "trained model. Other required features are filled using reference values "
        "from the available household dataset."
    )
    # Support both the existing .pkl model filenames and newer .joblib exports.
    # A saved model must still contain compatible preprocessing and input metadata.
    available = {}
    for name, key in MODEL_KEYS.items():
        candidates = (
            MODELS_DIR / f"{key}.pkl",
            MODELS_DIR / f"model_{key}.pkl",
            MODELS_DIR / f"model_{key}.joblib",
            MODELS_DIR / f"{key}.joblib",
        )
        found = next((path for path in candidates if path.is_file()), None)
        if found is not None:
            available[name] = found

    if not available:
        st.warning(
            "No saved trained models were found in the models/ folder. "
            "Expected files include bagging.pkl, xgboost.pkl, or "
            "model_bagging.joblib. Existing Predictions remains available."
        )
        return

    selected_name = st.selectbox("Select trained model", list(available), key="new_household_model")
    try:
        model = load_saved_model(str(available[selected_name]))
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
        return

    # The saved pipeline must expose the exact raw training feature names.
    if not hasattr(model, "feature_names_in_"):
        st.warning(
            "This model does not contain its original input column names. "
            "Save the complete fitted preprocessing-and-classifier pipeline "
            "to enable safe new-household predictions."
        )
        return
    required = list(model.feature_names_in_)
    missing = [c for c in required if c not in dataset.columns]
    if missing:
        st.error(
            f"The reference dataset is missing {len(missing)} model inputs. "
            "A compatible trained model and dataset are required."
        )
        return

    reference = dataset[required].copy()
    # Do not expose food-insecurity outcome answers or target-derived variables as inputs.
    forbidden = [
        c for c in required
        if c.lower() in {"food_insecure", "food_insecurity_risk", "food_insecurity", "target", "hhcode"}
        or c.lower().startswith("l1_")
    ]
    if forbidden:
        st.error("The saved model includes target/identifier fields as predictors. Please review training features first.")
        return

    # Default every input to a typical observed value, retaining its training dtype.
    defaults = {}
    for col in required:
        series = reference[col].dropna()
        if series.empty:
            defaults[col] = np.nan
        elif pd.api.types.is_numeric_dtype(reference[col]):
            defaults[col] = float(series.median())
        else:
            defaults[col] = series.mode().iloc[0]

    st.info(
        f"This model expects {len(required)} input features. The form displays a "
        "manageable selection; all remaining inputs use median or most-frequent "
        "values from the reference dataset. Results are illustrative and are not "
        "a validated food-security assessment."
    )

    # Show district by name, and automatically derive the matching province code.
    # Both code lists come from the source Stata value labels, not an assumed ordering.
    hints = ("household", "hhsize", "member", "size", "housing", "roof", "wall", "floor",
             "water", "toilet", "electric", "asset", "debt", "loan", "saving", "income")
    ranked = sorted(required, key=lambda c: (not any(h in c.lower() for h in hints), required.index(c)))
    editable = ranked[:min(12, len(ranked))]
    for priority in ("a_02_district", "a_01_province"):
        if priority in required and priority not in editable:
            editable.insert(0, priority)
    # Province is automatically populated when the user selects a district.
    if "a_02_district" in editable and "a_01_province" in editable:
        editable.remove("a_01_province")
    survey_labels, survey_questions = load_survey_labels()
    entries = {}
    other_dwelling_applicable = True
    with st.form("new_household_form"):
        st.subheader("Household Information")
        st.caption("Choose readable answers. The original survey values are used internally for predictions.")
        left, right = st.columns(2)
        for i, col in enumerate(editable):
            with (left if i % 2 == 0 else right):
                if col == "a_02_district":
                    district_names = sorted(DISTRICT_CODES)
                    entries[col] = st.selectbox(
                        "District", district_names, index=None,
                        placeholder="Select your district",
                        key="new_household_district",
                        help="All 25 Sri Lankan districts; survey codes are handled automatically.",
                    )
                    if entries[col]:
                        st.caption(f"Province: {DISTRICT_PROVINCES[entries[col]]}")
                elif col == "a_01_province":
                    entries[col] = st.selectbox(
                        "Province", sorted(PROVINCE_CODES), index=None,
                        placeholder="Select your province", key="new_household_province"
                    )
                else:
                    other_dwelling_enabled = True
                    if col == "c1_01_other" and "c1_01" in entries:
                        dwelling_code = entries["c1_01"]
                        dwelling_label = display_survey_choice(
                            dwelling_code, survey_labels.get("c1_01", {})
                        ).strip().lower()
                        other_dwelling_enabled = (
                            dwelling_label == "other"
                            or dwelling_label.startswith("other ")
                            or dwelling_label.startswith("other(")
                            or dwelling_label.startswith("others")
                        )
                    if col == "c1_01_other":
                        other_dwelling_applicable = other_dwelling_enabled
                        # Hide the conditional input entirely unless "Other" was selected.
                        if not other_dwelling_enabled:
                            continue
                    entries[col] = survey_field_widget(
                        col, reference[col].dropna(), defaults[col],
                        f"new_input_{i}", survey_labels, survey_questions,
                        other_dwelling_enabled=other_dwelling_enabled,
                    )
        submitted = st.form_submit_button(
            "Predict Food Insecurity Experience", type="primary", use_container_width=True
        )

    if not submitted:
        return
    if "a_02_district" in entries and entries["a_02_district"] is None:
        st.warning("Please select your district before predicting.")
        return
    input_row = pd.DataFrame([defaults], columns=required)
    for col, value in entries.items():
        if col == "a_02_district":
            input_row.at[0, col] = float(DISTRICT_CODES[value])
            if "a_01_province" in required:
                input_row.at[0, "a_01_province"] = PROVINCE_CODES[DISTRICT_PROVINCES[value]]
        elif col == "a_01_province":
            if value is not None:
                input_row.at[0, col] = PROVINCE_CODES[value]
        elif col == "c1_01_other" and not other_dwelling_applicable:
            # Not applicable: preserve the trained pipeline's reference default.
            continue
        else:
            input_row.at[0, col] = value
    for col in required:
        if pd.api.types.is_numeric_dtype(reference[col]):
            input_row[col] = pd.to_numeric(input_row[col], errors="coerce")
    try:
        predicted = int(model.predict(input_row)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(model.predict_proba(input_row)[0, classes.index(1)])
    except Exception as exc:
        st.error(f"Prediction could not be generated with these inputs: {exc}")
        return

    st.subheader("Prediction Result")
    a, b = st.columns(2)
    a.metric("Predicted Class", "Food Insecurity Experience" if predicted == 1 else "No Reported Experience")
    if probability is not None:
        b.metric("Model Score for Food Insecurity", f"{probability:.1%}")
        st.progress(max(0.0, min(1.0, probability)))
    st.caption(
        "The score is a model output, not a clinically or nationally validated risk estimate. "
        "Most inputs were set to typical dataset values; the result may change if "
        "those inputs differ for the actual household."
    )


def saved_prediction_tab(predictions):

    st.markdown(
        "Explore saved predictions from the held-out "
        "test dataset. These are existing model outputs, "
        "not predictions for new households."
    )

    model_name = st.selectbox(
        "Select model",
        list(MODEL_KEYS.keys()),
    )

    key = MODEL_KEYS[model_name]

    pred_col = f"{key}_pred"
    prob_col = f"{key}_prob"

    if pred_col not in predictions or prob_col not in predictions:
        st.error("Prediction columns not found.")
        return

    row_index = st.number_input(
        "Select test household index",
        min_value=0,
        max_value=len(predictions) - 1,
        value=0,
        step=1,
    )

    record = predictions.iloc[int(row_index)]

    actual = int(record["actual"])
    predicted = int(record[pred_col])
    probability = float(record[prob_col])

    left, middle, right = st.columns(3)

    left.metric(
        "Actual Class",
        "Food Insecurity" if actual == 1 else "No Experience",
    )

    middle.metric(
        "Predicted Class",
        "Food Insecurity" if predicted == 1 else "No Experience",
    )

    right.metric(
        "Predicted Probability",
        f"{probability:.1%}",
    )

    if actual == predicted:
        st.success("Correct classification")
    else:
        st.warning("Incorrect classification")

    st.progress(min(max(probability, 0.0), 1.0))

    st.subheader("Browse Test Predictions")

    table = pd.DataFrame({
        "Actual Class": predictions["actual"],
        "Predicted Class": predictions[pred_col],
        "Probability": predictions[prob_col],
    })

    light_table(table)

    st.download_button(
        "Download Predictions",
        data=table.to_csv(index=False),
        file_name=f"{key}_predictions.csv",
        mime="text/csv",
    )


def about_page():
    st.title("About the Research Project")

    st.subheader("Research Title")
    st.write(
        "Comparative Analysis of Bagging, Boosting, and "
        "Stacking Ensemble Learning Techniques for Household "
        "Food Insecurity Risk Prediction in Sri Lanka"
    )

    st.subheader("Research Objective")
    st.write(
        "To compare ensemble machine learning approaches "
        "for identifying households reporting food "
        "insecurity experiences."
    )

    st.subheader("Dataset")
    st.write(
        "BRIGHT Integrated Household Survey of Sri Lanka "
        "(2024–2025), IFPRI."
    )

    st.markdown(
        "[View dataset source]"
        "(https://doi.org/10.7910/DVN/FH0DSN)"
    )

    st.subheader("Machine Learning Techniques")
    st.markdown(
        """
        - **Baseline:** Decision Tree
        - **Bagging:** Bagging Classifier, Random Forest
        - **Boosting:** AdaBoost, XGBoost
        - **Stacking:** Stacking Classifier
        - **Hybridization:** Random Forest + Artificial Neural Network (RF + ANN)
        """
    )

    st.subheader("Evaluation Metrics")
    st.write(
        "Accuracy, Precision, Recall, F1 Score, "
        "ROC-AUC, ROC Curves, and Confusion Matrices."
    )

    st.subheader("Methodology and Limitations")
    st.write(
        "The models use an 80/20 stratified train-test "
        "split with random state 42. Missing values are "
        "imputed within the model pipeline."
    )

    st.warning(
        "These results are preliminary. Survey feature "
        "coding, special missing-value values, identifiers, "
        "and possible target leakage require further review. "
        "The dashboard is an academic demonstration, not "
        "a validated household screening tool."
    )


def main():
    # Sidebar branding: preserve the original network logo and navy palette.
    st.sidebar.markdown(
        """
        <div style="display:flex;flex-direction:column;align-items:flex-start;gap:7px;margin:0 0 12px 0;">
          <svg xmlns="http://www.w3.org/2000/svg" width="62" height="62"
               viewBox="0 0 64 64" role="img" aria-label="Ensemble ML network logo"
               style="flex-shrink:0;">
            <rect width="64" height="64" rx="14" fill="#173B63"/>
            <g fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round">
              <path d="M17 17 L32 32 L47 17 M17 47 L32 32 L47 47" opacity="0.85"/>
            </g>
            <g fill="#FFFFFF">
              <circle cx="17" cy="17" r="5"/><circle cx="47" cy="17" r="5"/>
              <circle cx="17" cy="47" r="5"/><circle cx="47" cy="47" r="5"/>
            </g>
            <circle cx="32" cy="32" r="7" fill="#7DD3C7" stroke="#FFFFFF" stroke-width="2"/>
          </svg>
          <div style="font-size:1.8rem;font-weight:850;letter-spacing:-0.03em;
                      color:#173B63;line-height:1.1;">ENSEMBLE ML</div>
          <div style="font-size:0.98rem;font-weight:650;color:#526A84;line-height:1.35;">
            Food Insecurity Prediction
          </div>
          <div style="font-size:0.82rem;font-weight:500;color:#8798AC;line-height:1.3;">
            Sri Lankan Household Survey
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown('<div class="ensemble-nav-heading">NAVIGATION</div>', unsafe_allow_html=True)

    pages = [
        "Overview",
        "Data Overview",
        "Model Comparison",
        "Visualizations",
        "Prediction Explorer",
        "About Project",
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Overview"

    for name in pages:
        is_active = st.session_state.active_page == name
        if st.sidebar.button(
            name,
            key=f"nav_{name.lower().replace(' ', '_')}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = name
            st.rerun()

    page = st.session_state.active_page

    try:
        dataset, comparison, predictions = load_data()
    except FileNotFoundError as exc:
        st.error(f"Required file missing: {exc}")
        st.stop()

    if page == "Overview":
        overview_page(dataset, comparison)

    elif page == "Data Overview":
        data_overview_page(dataset)

    elif page == "Model Comparison":
        comparison_page(comparison)

    elif page == "Visualizations":
        visualization_page(comparison)

    elif page == "Prediction Explorer":
        prediction_page(predictions, dataset)

    elif page == "About Project":
        about_page()


if __name__ == "__main__":
    main()


