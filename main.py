from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression

# -------------------- CONFIGURAÇÃO GERAL --------------------

EXPECTED_COLS = ["DATA", "TIPO", "CATEGORIA", "DESCRIÇÃO", "VALOR", "OBSERVAÇÃO"]

# Cores simples mantidas para os gráficos
COLORS = {
    "receita": "#2ca02c", # Verde
    "despesa": "#d62728", # Vermelho
    "saldo": "#1f77b4",   # Azul
    "trend": "#ff9896",   # Laranja claro
    "neutral": "#6c757d", # Cinza
}

DEFAULT_CHART_HEIGHT = 360

st.set_page_config(page_title="Dashboard Financeiro DEMO Simples", layout="wide", initial_sidebar_state="expanded",
                   menu_items={"About": "Dashboard Financeiro DEMO Simples © 2025"})

# -------------------- UTILITÁRIOS E PRÉ-PROCESSAMENTO --------------------

def parse_val_str_to_float(val) -> float:
    if pd.isna(val) or val == "": return 0.0
    s = str(val).strip()
    neg = False
    if (s.startswith("(") and s.endswith(")")) or s.startswith("-"):
        neg = True
        s = s.strip("()-")
    s = s.replace("R$", "").replace(" ", "").replace(".", "").replace(",", ".")
    try: v = float(s)
    except Exception: return 0.0
    return -abs(v) if neg else abs(v)

def money_fmt_br(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_category_color_map(df: pd.DataFrame) -> Dict[str, str]:
    if df is None or df.empty: return {}
    cats = sorted(df["CATEGORIA"].dropna().unique())
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    colors = [base[i % len(base)] for i in range(len(cats))]
    return {cat: colors[i] for i, cat in enumerate(cats)}

def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATA"]).reset_index(drop=True)
    df["VALOR_NUM"] = df["VALOR"].apply(parse_val_str_to_float)
    df["TIPO"] = df["TIPO"].fillna("").astype(str).str.strip()
    mask_empty_tipo = df["TIPO"] == ""
    df.loc[mask_empty_tipo, "TIPO"] = df.loc[mask_empty_tipo, "VALOR_NUM"].apply(lambda v: "Despesa" if v < 0 else "Receita")
    mask_receita = df["TIPO"].str.contains("Receita", case=False, na=False)
    mask_despesa = df["TIPO"].str.contains("Despesa", case=False, na=False)
    df.loc[mask_receita, "VALOR_NUM"] = abs(df.loc[mask_receita, "VALOR_NUM"])
    df.loc[mask_despesa, "VALOR_NUM"] = -abs(df.loc[mask_despesa, "VALOR_NUM"])
    df["CATEGORIA"] = df["CATEGORIA"].fillna("").astype(str).str.strip()
    
    def is_mostly_numeric_or_empty_category(s):
        s = str(s)
        if s == "": return True
        if s.isdigit() and len(s) < 5: return True
        return False
        
    mask_invalid_cat = df["CATEGORIA"].apply(is_mostly_numeric_or_empty_category)
    df.loc[mask_invalid_cat, "CATEGORIA"] = "NÃO CATEGORIZADO"
    df.loc[df["DESCRIÇÃO"] == "", "DESCRIÇÃO"] = "N/D"
    df.loc[df["OBSERVAÇÃO"] == "", "OBSERVAÇÃO"] = "N/D"
    df = df.sort_values("DATA").reset_index(drop=True)
    df["Saldo Acumulado"] = df["VALOR_NUM"].cumsum()
    df["year_month"] = df["DATA"].dt.to_period("M").astype(str)
    return df

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    f = df.copy()
    if filters.get("mode") == "range":
        f = f[(f["DATA"] >= filters["date_from"]) & (f["DATA"] <= filters["date_to"])]
    else:
        month = filters.get("month", "Todos")
        if month and month != "Todos":
            f = f[f["year_month"] == month]
    cats = filters.get("categories", [])
    if cats and "Todos" not in cats:
        f = f[f["CATEGORIA"].isin(cats)]
    return f.reset_index(drop=True)

# -------------------- DADOS MOCK (SUBSTITUINDO CONEXÃO) --------------------

def generate_mock_data() -> pd.DataFrame:
    """Gera dados imaginários (mock) com 1 ano de transações."""
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=d) for d in range(365)]
    
    data_points = []
    
    np.random.seed(42) 
    
    for date in dates:
        # Receita no dia
        if np.random.rand() > 0.3:
            value = np.random.randint(1000, 5000) * (0.8 + np.random.rand() * 0.4) 
            data_points.append({
                "DATA": date.strftime("%d/%m/%Y"), 
                "TIPO": "Receita", 
                "CATEGORIA": np.random.choice(["Mensalidade", "Serviços", "Doações"]),
                "DESCRIÇÃO": "Pagamento de Cliente/Membro",
                "VALOR": str(value),
                "OBSERVAÇÃO": "N/D"
            })
            
        # Despesa no dia
        if np.random.rand() > 0.2:
            value = -np.random.randint(500, 3000) * (0.8 + np.random.rand() * 0.4) 
            data_points.append({
                "DATA": date.strftime("%d/%m/%Y"), 
                "TIPO": "Despesa", 
                "CATEGORIA": np.random.choice(["Marketing", "Aluguel", "Salários", "Material de Escritório", "Utilities"]),
                "DESCRIÇÃO": np.random.choice(["Fatura", "Contas", "Compra de Suprimentos"]),
                "VALOR": str(value),
                "OBSERVAÇÃO": "N/D"
            })
            
    df_raw = pd.DataFrame(data_points, columns=EXPECTED_COLS)
    return preprocess_df(df_raw)

@st.cache_data(ttl=600)
def load_and_preprocess_data() -> pd.DataFrame:
    """Carrega APENAS dados mock. Não há tentativa de conexão com Sheets."""
    st.info("⚠️ Modo de Demonstração Simples Ativo. Carregando dados financeiros simulados.")
    return generate_mock_data()


# -------------------- PLOTS --------------------

def _get_empty_fig(text: str = "Sem dados") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=DEFAULT_CHART_HEIGHT)
    return fig

def plot_saldo_acumulado(df: pd.DataFrame) -> go.Figure:
    if df.empty: return _get_empty_fig()
    daily = df.groupby(df["DATA"].dt.date)["Saldo Acumulado"].last().reset_index()
    daily["DATA"] = pd.to_datetime(daily["DATA"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["DATA"], y=daily["Saldo Acumulado"], mode="lines+markers",
                             name="Saldo", line=dict(color=COLORS["saldo"], width=2)))
    if len(daily) > 1:
        X = daily["DATA"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = daily["Saldo Acumulado"].values
        reg = LinearRegression().fit(X, y)
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
        y_pred = reg.predict(X_line)
        dates_line = [datetime.fromordinal(int(x)) for x in X_line.flatten()]
        fig.add_trace(go.Scatter(x=dates_line, y=y_pred, mode="lines", name="Tendência", line=dict(color=COLORS["trend"], dash="dash")))
    fig.update_layout(height=DEFAULT_CHART_HEIGHT, title="Saldo Acumulado e Tendência")
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Saldo (R$)")
    return fig

def plot_fluxo_diario(df: pd.DataFrame) -> go.Figure:
    if df.empty: return _get_empty_fig()
    fluxo = df.groupby(df["DATA"].dt.date)["VALOR_NUM"].sum().reset_index()
    fluxo["DATA"] = pd.to_datetime(fluxo["DATA"])
    cores = [COLORS["receita"] if v >= 0 else COLORS["despesa"] for v in fluxo["VALOR_NUM"]]
    fig = go.Figure(go.Bar(x=fluxo["DATA"], y=fluxo["VALOR_NUM"], marker_color=cores))
    fig.update_layout(height=DEFAULT_CHART_HEIGHT, title="Fluxo de Caixa Diário (Líquido)")
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Valor (R$)")
    return fig

def plot_categoria_barras_pct(df: pd.DataFrame, kind: str = "Receita", category_colors: Dict[str,str]=None) -> go.Figure:
    assert kind in ("Receita", "Despesa")
    base = df[df["VALOR_NUM"] > 0] if kind == "Receita" else df[df["VALOR_NUM"] < 0]
    if base.empty: return _get_empty_fig(f"Sem dados de {kind}")
    
    df_plot = base["VALOR_NUM"].abs().groupby(base["CATEGORIA"]).sum().reset_index()
    df_plot.columns = ["CATEGORIA", "VALOR"]
    total = df_plot["VALOR"].sum()
    df_plot["PERCENT"] = (df_plot["VALOR"] / total) * 100
    df_plot = df_plot.sort_values("PERCENT", ascending=False)
    
    marker_colors = [category_colors.get(c, COLORS["neutral"]) for c in df_plot["CATEGORIA"]]

    fig = go.Figure(go.Bar(
        x=df_plot["CATEGORIA"], 
        y=df_plot["PERCENT"], 
        marker_color=marker_colors,
        hovertemplate="<b>%{x}</b><br>Valor: %{y:.1f}%<extra></extra>",
    ))
    
    fig.update_layout(
        height=DEFAULT_CHART_HEIGHT, 
        title=f'Composição de {kind} (Porcentagem)',
        xaxis_tickangle=-45
    )
    fig.update_xaxes(title_text="Categoria")
    fig.update_yaxes(title_text="Porcentagem (%)", ticksuffix="%")
    return fig

def plot_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty: return _get_empty_fig()
    dfh = df.copy()
    dfh['day'] = dfh['DATA'].dt.day
    dfh['ym'] = dfh['DATA'].dt.to_period('M').astype(str)
    pivot = dfh.groupby(['ym','day'])['VALOR_NUM'].sum().reset_index()
    heat = pivot.pivot(index='ym', columns='day', values='VALOR_NUM').fillna(0)
    fig = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='RdBu', reversescale=True,
                                     hovertemplate="Mês: %{y}<br>Dia: %{x}<br>Saldo Diário: %{z:.2f} R$<extra></extra>"))
    fig.update_layout(title='Heatmap Mensal de Saldo Diário', height=DEFAULT_CHART_HEIGHT+40)
    fig.update_xaxes(title_text="Dia do Mês")
    fig.update_yaxes(title_text="Mês")
    return fig

# -------------------- SIDEBAR E FILTROS --------------------

def sidebar_filters_and_controls(df: pd.DataFrame) -> Tuple[str, Dict]:
    st.sidebar.title("Dashboard Financeiro DEMO Simples")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Altere a visualização", options=["Resumo Financeiro", "Dashboard Detalhado"], key="sb_page")
    toggle_multi = st.sidebar.checkbox("Ativar filtro avançado (múltipla seleção e período)", value=False, key="sb_toggle_multi")
    
    # Datas para o slider (baseadas nos dados mock)
    min_ts = df["DATA"].min() if not df.empty else pd.Timestamp(datetime.today() - timedelta(days=365))
    max_ts = df["DATA"].max() if not df.empty else pd.Timestamp(datetime.today())
    min_d = min_ts.date()
    max_d = max_ts.date()
    
    filters: Dict = {"mode": "month", "month": "Todos", "categories": []}
    
    if toggle_multi:
        with st.sidebar.expander("Filtros Avançados", expanded=True):
            categories = sorted(df["CATEGORIA"].unique()) if not df.empty else []
            categories = [c for c in categories if c != ""]
            selected_cats = st.multiselect("Categorias (múltiplas)", options=categories, default=categories if categories else [], key="sb_cat_multi")
            slider_val = st.slider("Período (arraste)", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM-DD", step=timedelta(days=1), key="sb_date_slider")
            
            date_from = pd.to_datetime(slider_val[0])
            date_to = pd.to_datetime(slider_val[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            filters["mode"] = "range"
            filters["date_from"] = date_from
            filters["date_to"] = date_to
            filters["categories"] = selected_cats
    else:
        st.sidebar.markdown("### Filtro Rápido")
        months = ["Todos"] + sorted(df["year_month"].unique(), reverse=True) if not df.empty else ["Todos"]
        selected_month = st.sidebar.selectbox("Mês (ano-mês)", months, key="sb_month")
        categories = ["Todos"] + sorted(df["CATEGORIA"].unique()) if not df.empty else ["Todos"]
        categories = [c for c in categories if c != ""]
        selected_category = st.sidebar.selectbox("Categoria", categories, key="sb_cat_single")
        
        filters["mode"] = "month"
        filters["month"] = selected_month
        filters["categories"] = [selected_category] if selected_category != "Todos" else []
        
    st.sidebar.markdown("---")
    if st.sidebar.button("Recarregar Dados (Mock)", key="sb_clear_cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.caption("Dashboard de Demonstração Simples")
    return page, filters

# -------------------- KPIS --------------------

def _sum_period(df: pd.DataFrame, start_dt: datetime, end_dt: datetime, tipo: str = "all") -> float:
    if df.empty: return 0.0
    mask = (df["DATA"] >= start_dt) & (df["DATA"] <= end_dt)
    s = df.loc[mask, "VALOR_NUM"]
    if tipo == "receita": return s[s > 0].sum()
    elif tipo == "despesa": return s[s < 0].sum()
    else: return s.sum()

def _kpi_delta_text(curr: float, prev: float) -> str:
    diff = curr - prev
    pct = (diff / abs(prev)) * 100 if abs(prev) > 0.0001 else (100.0 if abs(diff) > 0.0 else 0.0)
    sign = "+" if diff >= 0 else "-"
    absdiff = abs(diff)
    return f"{sign}{money_fmt_br(absdiff)} ({sign}{pct:.0f}%)"

def render_kpi_cards(df_full: pd.DataFrame, df_filtered: pd.DataFrame):
    if df_full.empty:
        st.info("Sem dados para KPIs")
        return

    # Valores filtrados
    receita_filtrada = df_filtered[df_filtered["VALOR_NUM"] > 0]["VALOR_NUM"].sum()
    despesa_filtrada = df_filtered[df_filtered["VALOR_NUM"] < 0]["VALOR_NUM"].sum()
    saldo_filtrado = receita_filtrada + despesa_filtrada

    # Delta (Comparação dos últimos 30 dias com os 30 dias anteriores)
    end = df_full["DATA"].max()
    last30_end = pd.to_datetime(end)
    last30_start = last30_end - pd.Timedelta(days=29)
    prev30_end = last30_start - pd.Timedelta(seconds=1)
    prev30_start = prev30_end - pd.Timedelta(days=29)

    receita_curr = _sum_period(df_full, last30_start, last30_end, tipo="receita")
    receita_prev = _sum_period(df_full, prev30_start, prev30_end, tipo="receita")
    despesa_curr = _sum_period(df_full, last30_start, last30_end, tipo="despesa")
    despesa_prev = _sum_period(df_full, prev30_start, prev30_end, tipo="despesa")
    saldo_curr = receita_curr + despesa_curr
    saldo_prev = receita_prev + despesa_prev

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Receita Total (Filtrada)",
            value=money_fmt_br(receita_filtrada),
            delta=_kpi_delta_text(receita_curr, receita_prev),
            delta_color="normal"
        )
    with c2:
        st.metric(
            label="Despesa Total (Filtrada)",
            value=money_fmt_br(abs(despesa_filtrada)),
            delta=_kpi_delta_text(-despesa_curr, -despesa_prev), # Compara valores absolutos (gasto)
            delta_color="inverse"
        )
    with c3:
        st.metric(
            label="Saldo Total (Filtrado)",
            value=money_fmt_br(saldo_filtrado),
            delta=_kpi_delta_text(saldo_curr, saldo_prev),
            delta_color="normal"
        )

# -------------------- TABELA / EXPORT --------------------

def render_table(df: pd.DataFrame, key: str):
    if df.empty:
        st.info("Sem lançamentos para mostrar com os filtros atuais.")
        return
    df_display = df.copy()
    df_display["Data"] = df_display["DATA"].dt.date
    df_display["Valor (R$)"] = df_display["VALOR_NUM"].apply(money_fmt_br)
    df_display = df_display.rename(columns={"TIPO":"Tipo","CATEGORIA":"Categoria","DESCRIÇÃO":"Descrição","OBSERVAÇÃO":"Observação"})
    st.dataframe(df_display[["Data","Tipo","Categoria","Descrição","Valor (R$)","Observação"]], use_container_width=True, key=key, hide_index=True)

# -------------------- ESTRUTURA DO DASHBOARD --------------------

def main():
    
    # 1. Carregar dados (APENAS MOCK)
    df_full = load_and_preprocess_data()
    
    # 2. Sidebar e Filtros
    page, filters = sidebar_filters_and_controls(df_full)
    
    # 3. Aplicar Filtros
    df_filtered = apply_filters(df_full, filters)
    
    # 4. Título
    st.title("Dashboard Financeiro DEMONSTRAÇÃO Simples")
    
    if df_full.empty and df_filtered.empty:
        st.error("Sem dados simulados para exibir.")
        return

    # 5. Renderizar KPIs
    st.header("Indicadores Chave de Performance (KPIs)")
    render_kpi_cards(df_full, df_filtered)
    st.markdown("---")

    # 6. Renderizar Conteúdo da Página
    category_colors = get_category_color_map(df_full)
    
    if page == "Resumo Financeiro":
        st.header("Fluxo e Saldo")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_saldo_acumulado(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_fluxo_diario(df_filtered), use_container_width=True)

        st.header("Composição de Receita e Despesa")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_categoria_barras_pct(df_filtered, kind="Receita", category_colors=category_colors), use_container_width=True)
        with col4:
            st.plotly_chart(plot_categoria_barras_pct(df_filtered, kind="Despesa", category_colors=category_colors), use_container_width=True)
            
        st.markdown("---")
        st.header("Tabela de Lançamentos")
        render_table(df_filtered, key="table_summary")

    elif page == "Dashboard Detalhado":
        st.header("Análise Temporal Avançada")
        st.plotly_chart(plot_monthly_heatmap(df_filtered), use_container_width=True)
        
        st.header("Tabela Completa de Lançamentos")
        render_table(df_filtered, key="table_detailed")
        
    st.markdown("---")
    st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":
    main()
