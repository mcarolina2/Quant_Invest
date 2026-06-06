import streamlit as st

st.set_page_config(
    page_title="Sistema de Suporte à Decisão de Investimentos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-low { border-left-color: #2ca02c; }
    .risk-medium { border-left-color: #ff7f0e; }
    .risk-high { border-left-color: #d62728; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Sistema de Suporte à Decisão de Investimentos</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fase I — Filtro de Risco e Econometria | CAPM · Fama-French · ARCH/GARCH</div>', unsafe_allow_html=True)

# Sidebar — configurações globais
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("Ativos")
    tickers_input = st.text_area(
        "Tickers (um por linha)",
        value="PETR4.SA\nVALE3.SA\nITUB4.SA\nBBDC4.SA\nABEV3.SA",
        height=120,
        help="Use sufixo .SA para ativos brasileiros (B3)"
    )
    tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]

    st.subheader("Período de Análise")
    import datetime
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Início", value=datetime.date(2020, 1, 1), key="date_start")
    with col2:
        end_date = st.date_input("Fim", value=datetime.date.today(), key="date_end")

    start_str = str(start_date)
    end_str = str(end_date)

    st.subheader("Benchmark & Mercado")
    benchmark = st.selectbox(
        "Índice de Mercado",
        ["^BVSP", "^GSPC", "^IXIC"],
        format_func=lambda x: {"^BVSP": "Ibovespa (B3)", "^GSPC": "S&P 500", "^IXIC": "NASDAQ"}[x]
    )
    risk_free_rate = st.number_input(
        "Taxa Livre de Risco (% a.a.)",
        min_value=0.0, max_value=30.0, value=10.75, step=0.25,
        help="Selic atual como proxy"
    ) / 100

    st.subheader("Parâmetros GARCH")
    garch_p = st.slider("Ordem p (ARCH)", 1, 3, 1)
    garch_q = st.slider("Ordem q (GARCH)", 1, 3, 1)

    run_btn = st.button("🚀 Executar Análise", type="primary", use_container_width=True)

# Store in session state (use distinct keys from widget keys)
st.session_state["_tickers"] = tickers
st.session_state["_start"] = start_str
st.session_state["_end"] = end_str
st.session_state["_benchmark"] = benchmark
st.session_state["_rf"] = risk_free_rate
st.session_state["_garch_p"] = garch_p
st.session_state["_garch_q"] = garch_q
st.session_state["run"] = run_btn or st.session_state.get("run", False)

# Pages via tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral & Dados",
    "📐 CAPM",
    "🔬 Fama-French",
    "🌊 ARCH/GARCH"
])

if not st.session_state.get("run"):
    for tab in [tab1, tab2, tab3, tab4]:
        with tab:
            st.info("👈 Configure os parâmetros na barra lateral e clique em **Executar Análise** para começar.")
    st.stop()

# Load data (shared across tabs)
from modules.data_loader import load_data
from modules.capm import run_capm
from modules.fama_french import run_fama_french
from modules.garch import run_garch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

with st.spinner("Baixando dados de mercado..."):
    data = load_data(tickers, benchmark, start_str, end_str)

if data is None or data["prices"].empty:
    st.error("❌ Não foi possível carregar os dados. Verifique os tickers e o período selecionado.")
    st.stop()

prices = data["prices"]
returns = data["returns"]
mkt_returns = data["mkt_returns"]
benchmark_prices = data["benchmark_prices"]

# ─── TAB 1: Visão Geral ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Preços Históricos Normalizados (Base 100)")
    norm = (prices / prices.iloc[0]) * 100
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
    bm_norm = (benchmark_prices / benchmark_prices.iloc[0]) * 100
    fig.add_trace(go.Scatter(x=bm_norm.index, y=bm_norm.values.flatten(),
                             mode="lines", name=f"Benchmark ({benchmark})",
                             line=dict(dash="dash", color="black")))
    fig.update_layout(height=400, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estatísticas Descritivas dos Retornos Diários")
    stats = returns.describe().T
    stats["skewness"] = returns.skew()
    stats["kurtosis"] = returns.kurtosis()
    stats["annualized_return"] = (1 + returns.mean()) ** 252 - 1
    stats["annualized_vol"] = returns.std() * np.sqrt(252)
    stats["sharpe"] = (stats["annualized_return"] - risk_free_rate) / stats["annualized_vol"]
    display_cols = ["mean", "std", "min", "max", "skewness", "kurtosis", "annualized_return", "annualized_vol", "sharpe"]
    rename = {
        "mean": "Retorno Médio Diário", "std": "Desvio Padrão Diário",
        "min": "Mínimo", "max": "Máximo",
        "skewness": "Assimetria", "kurtosis": "Curtose",
        "annualized_return": "Retorno Anualizado", "annualized_vol": "Volatilidade Anualizada",
        "sharpe": "Índice de Sharpe"
    }
    st.dataframe(
        stats[display_cols].rename(columns=rename).style
            .format("{:.4f}")
            .background_gradient(cmap="RdYlGn", subset=["Índice de Sharpe"])
            .background_gradient(cmap="RdYlGn_r", subset=["Volatilidade Anualizada"]),
        use_container_width=True
    )

    st.subheader("Matriz de Correlação")
    corr = returns.corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

# ─── TAB 2: CAPM ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Capital Asset Pricing Model (CAPM)")
    with st.spinner("Estimando CAPM..."):
        capm_results = run_capm(returns, mkt_returns, risk_free_rate)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Resultados por Ativo")
        df_capm = pd.DataFrame(capm_results).T
        df_capm.index.name = "Ticker"
        st.dataframe(
            df_capm.style
                .format("{:.4f}")
                .background_gradient(cmap="RdYlGn_r", subset=["Beta"])
                .background_gradient(cmap="RdYlGn", subset=["Alpha (Jensen)"]),
            use_container_width=True
        )

    with col2:
        st.markdown("#### Beta × Retorno Esperado (SML)")
        betas = [v["Beta"] for v in capm_results.values()]
        alphas = [v["Alpha (Jensen)"] for v in capm_results.values()]
        names = list(capm_results.keys())
        beta_range = np.linspace(0, max(betas) * 1.2, 100)
        er_sml = risk_free_rate + beta_range * (mkt_returns.mean() * 252 - risk_free_rate)
        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(x=beta_range, y=er_sml, mode="lines",
                                      name="SML", line=dict(color="gray", dash="dash")))
        er_actual = [v["Retorno Esperado (CAPM)"] for v in capm_results.values()]
        er_realized = [v["Retorno Realizado"] for v in capm_results.values()]
        fig_sml.add_trace(go.Scatter(x=betas, y=er_realized, mode="markers+text",
                                      text=names, textposition="top center",
                                      marker=dict(size=12, color="steelblue"),
                                      name="Retorno Realizado"))
        fig_sml.update_layout(xaxis_title="Beta", yaxis_title="Retorno Anualizado",
                               height=380, hovermode="closest")
        st.plotly_chart(fig_sml, use_container_width=True)

    st.markdown("#### Detalhamento das Regressões")
    for ticker, res in capm_results.items():
        with st.expander(f"📋 {ticker} — β = {res['Beta']:.4f} | α = {res['Alpha (Jensen)']:.4f} | R² = {res['R²']:.4f}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta (β)", f"{res['Beta']:.4f}", help="Sensibilidade ao mercado")
            c2.metric("Alpha de Jensen (α)", f"{res['Alpha (Jensen)']:.4f}")
            c3.metric("R²", f"{res['R²']:.4f}")
            c4.metric("p-valor (β)", f"{res['p-valor Beta']:.4f}",
                      delta="Sig." if res['p-valor Beta'] < 0.05 else "N. Sig.",
                      delta_color="normal" if res['p-valor Beta'] < 0.05 else "off")
            c1.metric("Prêmio de Risco", f"{res['Prêmio de Risco']:.2%}")
            c2.metric("Retorno Esperado (CAPM)", f"{res['Retorno Esperado (CAPM)']:.2%}")
            c3.metric("Retorno Realizado", f"{res['Retorno Realizado']:.2%}")
            risk_label = "🟢 Defensivo" if res['Beta'] < 0.8 else ("🟡 Neutro" if res['Beta'] < 1.2 else "🔴 Agressivo")
            st.info(f"**Classificação de Risco Sistemático:** {risk_label}")

# ─── TAB 3: FAMA-FRENCH ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Modelo de Três Fatores de Fama-French")
    st.info("""
    **Metodologia:** Os fatores SMB e HML são aproximados (proxy) a partir dos retornos dos ativos disponíveis na carteira.
    Para análises acadêmicas rigorosas, recomenda-se utilizar os fatores do Kenneth French Data Library ou do NEFIN (Brasil).
    """)

    with st.spinner("Estimando Fama-French 3F..."):
        ff_results = run_fama_french(returns, mkt_returns, risk_free_rate)

    if ff_results:
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            st.markdown("#### Coeficientes do Modelo 3F")
            df_ff = pd.DataFrame(ff_results).T
            df_ff.index.name = "Ticker"
            st.dataframe(
                df_ff[["Alpha", "Beta_Mkt", "Beta_SMB", "Beta_HML", "R²", "R² Ajustado"]].style
                    .format("{:.4f}")
                    .background_gradient(cmap="RdYlGn", subset=["R²"]),
                use_container_width=True
            )
        with col2:
            st.markdown("#### Exposição Média aos Fatores")
            avg_betas = {
                "Mercado (Mkt-RF)": df_ff["Beta_Mkt"].mean(),
                "Tamanho (SMB)": df_ff["Beta_SMB"].mean(),
                "Valor (HML)": df_ff["Beta_HML"].mean(),
            }
            fig_bar = go.Figure(go.Bar(
                x=list(avg_betas.keys()),
                y=list(avg_betas.values()),
                marker_color=["#1f77b4", "#2ca02c", "#ff7f0e"]
            ))
            fig_bar.update_layout(height=300, yaxis_title="Beta Médio",
                                   title="Exposição Média da Carteira")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Interpretação dos Fatores")
        for ticker, res in ff_results.items():
            with st.expander(f"📋 {ticker} — R² = {res['R²']:.4f} | R² Aj. = {res['R² Ajustado']:.4f}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Alpha (α)", f"{res['Alpha']:.4f}")
                c2.metric("β Mercado", f"{res['Beta_Mkt']:.4f}")
                c3.metric("β SMB (Tamanho)", f"{res['Beta_SMB']:.4f}",
                          delta="Small-cap bias" if res['Beta_SMB'] > 0 else "Large-cap bias",
                          delta_color="off")
                c4.metric("β HML (Valor)", f"{res['Beta_HML']:.4f}",
                          delta="Value bias" if res['Beta_HML'] > 0 else "Growth bias",
                          delta_color="off")
                smb_interp = "📦 Tendência para small-caps" if res['Beta_SMB'] > 0 else "🏦 Tendência para large-caps"
                hml_interp = "💎 Estilo Valor (Value)" if res['Beta_HML'] > 0 else "🚀 Estilo Crescimento (Growth)"
                st.markdown(f"- **SMB:** {smb_interp}  \n- **HML:** {hml_interp}")
    else:
        st.warning("São necessários pelo menos 3 ativos para construir os fatores SMB e HML proxy.")

# ─── TAB 4: ARCH/GARCH ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Modelos ARCH/GARCH — Volatilidade Condicional")

    ticker_sel = st.selectbox("Selecionar Ativo para Análise GARCH", tickers)

    with st.spinner(f"Ajustando GARCH({garch_p},{garch_q}) para {ticker_sel}..."):
        garch_res = run_garch(returns[ticker_sel].dropna(), p=garch_p, q=garch_q)

    if garch_res:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Log-Likelihood", f"{garch_res['log_likelihood']:.2f}")
        col2.metric("AIC", f"{garch_res['aic']:.2f}")
        col3.metric("BIC", f"{garch_res['bic']:.2f}")
        col4.metric("Persistência (α+β)", f"{garch_res['persistence']:.4f}",
                    delta="Alta" if garch_res['persistence'] > 0.95 else "Moderada",
                    delta_color="off")

        st.markdown("#### Volatilidade Condicional Estimada vs Retornos")
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=garch_res["conditional_vol"].index,
            y=garch_res["conditional_vol"].values * 100,
            mode="lines", name="Vol. Condicional (%)", line=dict(color="orange")))
        fig_g.add_trace(go.Scatter(
            x=returns[ticker_sel].index,
            y=returns[ticker_sel].values * 100,
            mode="lines", name="Retorno Diário (%)",
            line=dict(color="steelblue", width=0.8), opacity=0.6,
            yaxis="y2"))
        fig_g.update_layout(
            height=420,
            yaxis=dict(title="Volatilidade Condicional (%)"),
            yaxis2=dict(title="Retorno (%)", overlaying="y", side="right"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("#### Parâmetros Estimados do Modelo")
        df_params = garch_res["params"]
        st.dataframe(df_params.style.format("{:.6f}"), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Distribuição dos Resíduos Padronizados")
            std_resid = garch_res["std_resid"]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=std_resid, nbinsx=50, name="Resíduos",
                                             histnorm="probability density",
                                             marker_color="steelblue", opacity=0.7))
            import scipy.stats as stats
            x_range = np.linspace(std_resid.min(), std_resid.max(), 200)
            fig_hist.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range),
                                          mode="lines", name="Normal Teórica",
                                          line=dict(color="red", dash="dash")))
            fig_hist.update_layout(height=320, bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            st.markdown("#### Volatilidade Anualizada Recente (252 dias)")
            ann_vol_series = garch_res["conditional_vol"] * np.sqrt(252) * 100
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=ann_vol_series.index[-252:],
                y=ann_vol_series.values[-252:],
                mode="lines", fill="tozeroy", name="Vol. Anualizada",
                line=dict(color="orange")))
            fig_vol.update_layout(height=320, yaxis_title="Volatilidade Anualizada (%)")
            st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("#### Previsão de Volatilidade (Próximos 10 dias)")
        forecast_vol = garch_res.get("forecast_vol")
        if forecast_vol is not None:
            df_fc = pd.DataFrame({
                "Dia": range(1, len(forecast_vol) + 1),
                "Vol. Condicional (%)": forecast_vol * 100,
                "Vol. Anualizada (%)": forecast_vol * np.sqrt(252) * 100
            })
            st.dataframe(df_fc.set_index("Dia").style.format("{:.4f}"), use_container_width=True)
    else:
        st.error("Não foi possível ajustar o modelo GARCH. Tente aumentar o período de análise.")

st.divider()
st.caption("📌 Fase I — Filtro de Risco e Econometria | Projeto de Suporte à Decisão de Investimentos")