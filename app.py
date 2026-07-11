import streamlit as st
from modules.data_loader import load_data
from modules.capm import run_capm
from modules.fama_french import run_fama_french
from modules.garch import run_garch
from modules.risk_metrics import compute_risk_metrics
from modules.features import build_features, train_test_split_ts
from modules.ml_models import run_random_forest, run_xgboost, run_lightgbm, HAS_XGB, HAS_LGB
from modules.dl_models import run_lstm, run_gru, HAS_TORCH
from modules.backtesting import run_backtest
from modules.b3_tickers import (get_all_tickers, get_sectors, get_tickers_by_sector, B3_TICKERS_BY_SECTOR)
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as stats


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

    st.subheader("📋 Seleção de Ativos")
         #Modo de seleção
    modo = st.radio(
        "Modo de entrada",
        ["🔍 Busca por ativo", "📂 Por setor", "✏️ Digitar manualmente"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    if modo == "🔍 Busca por ativo":
        tickers = st.multiselect(
            "Busque e selecione os ativos:",
            options=get_all_tickers(),
            default=["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"],
            placeholder="Digite o ticker (ex: PETR4.SA)...",
            help="Todos os principais ativos da B3. Digite para filtrar.",
        )
    
    elif modo == "📂 Por setor":
        setor_sel = st.selectbox(
            "Setor:",
            options=get_sectors(),
            index=0,
        )
        opcoes_setor = get_tickers_by_sector(setor_sel)
        tickers = st.multiselect(
            f"Ativos de {setor_sel}:",
            options=opcoes_setor,
            default=opcoes_setor[:3] if len(opcoes_setor) >= 3 else opcoes_setor,
            placeholder="Selecione um ou mais ativos...",
        )
    
    else:  # ✏️ Digitar manualmente
        tickers_input = st.text_area(
            "Tickers (um por linha)",
            value="PETR4.SA\nVALE3.SA\nITUB4.SA\nBBDC4.SA\nABEV3.SA",
            height=130,
            help="Qualquer ticker do Yahoo Finance. Sufixo .SA para B3.",
        )
        tickers = [t.strip().upper() for t in tickers_input.splitlines() if t.strip()]
    
    # ── Validação e feedback visual ───────────────────────────────────────────────
    if not tickers:
        st.warning("⚠️ Selecione pelo menos um ativo.")
    else:
        st.caption(f"✅ {len(tickers)} ativo(s) selecionado(s)")
        # Preview compacto dos selecionados
        with st.expander("Ver ativos selecionados", expanded=False):
            for t in tickers:
                setor_do_ticker = next(
                    (s for s, lst in B3_TICKERS_BY_SECTOR.items() if t in lst),
                    "Externo / Manual"
                )
                st.write(f"`{t}` — {setor_do_ticker}")
#── Período ───
    st.subheader("Período de Análise")
    import datetime
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Início", value=datetime.date(2020, 1, 1), key="date_start")
    with col2:
        end_date = st.date_input("Fim", value=datetime.date.today(), key="date_end")
    start_str = str(start_date)
    end_str = str(end_date)

 # ── Benchmark ─

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
# ── Filtros CAPM ────
    st.subheader("🔍 Filtros de Risco (CAPM)")
    beta_min, beta_max = st.slider("Intervalo de Beta", 0.0, 3.0, (0.0, 3.0), step=0.05)
    alpha_min = st.number_input("Alpha mínimo (a.a.)", value=-1.0, step=0.01, format="%.2f")

# ── GARCH ────
    st.subheader("Parâmetros GARCH")
    garch_p = st.slider("Ordem p (ARCH)", 1, 3, 1)
    garch_q = st.slider("Ordem q (GARCH)", 1, 3, 1)

    run_btn = st.button("🚀 Executar Análise", type="primary", use_container_width=True)

 # ── ML ────────────────────────────────────────────────────────────────────
    st.subheader("🤖 Machine Learning")
    ml_ticker = st.selectbox("Ativo para ML/DL", tickers if tickers else ["PETR4.SA"])
    ml_horizon = st.slider("Horizonte de Previsão (dias)", 1, 10, 1)
    ml_test_ratio = st.slider("Proporção de Teste (%)", 10, 40, 20) / 100

    with st.expander("Parâmetros RF / Boosting"):
        rf_n_est  = st.slider("RF — N° Estimadores", 50, 500, 200, 50)
        rf_depth  = st.slider("RF — Profundidade Máx.", 2, 10, 5)
        xgb_n_est = st.slider("XGB/LGB — N° Estimadores", 50, 500, 200, 50)
        xgb_lr    = st.number_input("XGB/LGB — Learning Rate", 0.001, 0.3, 0.05, 0.005)
        n_cv      = st.slider("Folds CV Temporal", 3, 10, 5)

    with st.expander("Parâmetros LSTM / GRU"):
        dl_seq_len   = st.slider("Seq. Length", 5, 60, 20)
        dl_hidden    = st.slider("Hidden Size", 16, 256, 64, 16)
        dl_layers    = st.slider("Num Layers", 1, 4, 2)
        dl_epochs    = st.slider("Épocas", 10, 200, 50, 10)
        dl_batch     = st.slider("Batch Size", 8, 128, 32, 8)
        dl_lr        = st.number_input("Learning Rate DL", 0.0001, 0.01, 0.001, format="%.4f")

# ── Backtesting ───────────────────────────────────────────────────────────
st.subheader("📊 Backtesting")
bt_threshold = st.number_input("Threshold de Sinal", -0.05, 0.05, 0.0, 0.001, format="%.3f")
bt_cost      = st.number_input("Custo de Transação (1-way, %)", 0.0, 1.0, 0.1, 0.01) / 100

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral & Dados",
    "📐 CAPM",
    "🔬 Fama-French",
    "🌊 ARCH/GARCH",
    "🤖 ML & Backtesting"
])

if not st.session_state.get("run"):
    #tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visão Geral","CAPM","Fama-French","ARCH/GARCH","ML & Backtesting"])
    for t in [tab1,tab2,tab3,tab4,tab5]:
        with t:
            st.info("👈 Configure os parâmetros na barra lateral e clique em **Executar Análise** para começar.")
    st.stop()


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
#######Adicionando a parte de risco
    risk_metrics = {}

    for ticker in returns.columns:

        risk_metrics[ticker] = (
            compute_risk_metrics(
                returns[ticker].dropna(),
                risk_free_rate
            )
        )

    risk_df = pd.DataFrame(
        risk_metrics
    ).T

    stats["skewness"] = returns.skew()
    stats["kurtosis"] = returns.kurtosis()
    stats["annualized_return"] = (1 + returns.mean()) ** 252 - 1
    stats["annualized_vol"] = returns.std() * np.sqrt(252)
    #stats["sharpe"] = (stats["annualized_return"] - risk_free_rate) / stats["annualized_vol"]
    stats["sharpe"] = (    stats["annualized_return"]    - risk_free_rate) / stats["annualized_vol"] #####
    stats = stats.join(risk_df)
    display_cols = ["mean", "std", "min", "max", "skewness", "kurtosis", "annualized_return", "annualized_vol", "sharpe", "CV", "Jarque-Bera", "JB p-value", "Ljung-Box p-value",
    "ARCH p-value", "VaR 95%", "CVaR 95%","Sortino"]
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

    st.subheader("Diagnóstico Estatístico")

    for ticker in returns.columns:

        m = risk_metrics[ticker]

        with st.expander(f"📊 {ticker}"):

            st.write(
                f"Normalidade (JB): "
                f"{'✅' if m['JB p-value'] > 0.05 else '❌'}"
            )

            st.write(
                f"Autocorrelação (Ljung-Box): "
                f"{'❌ Presente' if m['Ljung-Box p-value'] < 0.05 else '✅ Não detectada'}"
            )

            st.write(
                f"Efeito ARCH: "
                f"{'✅ Sim' if m['ARCH p-value'] < 0.05 else '❌ Não'}"
            )

            st.write(f"VaR(95%): {m['VaR 95%']:.2%}")
            st.write(f"CVaR(95%): {m['CVaR 95%']:.2%}")
            st.write(f"Sortino: {m['Sortino']:.2f}")


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
        # 
        col5, col6 = st.columns(2)
        col5.metric("VaR GARCH (95%)", f"{garch_res['var_garch_95']:.2%}" )
        col6.metric("Volatilidade Prevista", f"{garch_res['forecast_vol'][0]:.2%}")
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

    # ─── TAB 5: ML & BACKTESTING ──────────────────────────────────────────────────
with tab5:
    st.subheader("🤖 Motor Preditivo — Machine Learning & Deep Learning")

    # Validate ticker selection
    if ml_ticker not in returns.columns:
        st.error(f"Ticker '{ml_ticker}' não disponível nos dados carregados.")
        st.stop()

    with st.spinner("Construindo features..."):
        from modules.features import build_features, train_test_split_ts
        df_feat = build_features(returns[ml_ticker], horizon=ml_horizon)

    if len(df_feat) < 100:
        st.warning("Dados insuficientes para treinar os modelos. Aumente o período de análise.")
        st.stop()

    X_train, y_train, X_test, y_test, feat_cols = train_test_split_ts(df_feat, test_ratio=ml_test_ratio)
    st.caption(f"Treino: {len(X_train)} obs | Teste: {len(X_test)} obs | Features: {len(feat_cols)}")

    model_tabs = st.tabs(["🌲 Random Forest", "⚡ XGBoost", "💡 LightGBM", "🧠 LSTM", "🔄 GRU", "📊 Backtesting"])

    # ── helpers ──────────────────────────────────────────────────────────────
    def _plot_predictions(y_true, y_pred, dates, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines", name="Real", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="Previsto",
                                  line=dict(color="orange", dash="dash")))
        fig.update_layout(height=350, title=title, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        return fig

    def _plot_feature_importance(fi: pd.Series, top_n: int = 15):
        fi_top = fi.head(top_n)
        fig = go.Figure(go.Bar(x=fi_top.values[::-1], y=fi_top.index[::-1],
                                orientation="h", marker_color="steelblue"))
        fig.update_layout(height=max(300, top_n * 22), title=f"Top {top_n} Features",
                          xaxis_title="Importância")
        return fig

    def _show_cv_metrics(cv_df: pd.DataFrame):
        st.markdown("**Validação Cruzada Temporal**")
        mean_row = cv_df.mean().rename("Média").to_frame().T
        std_row  = cv_df.std().rename("Desvio Padrão").to_frame().T
        summary  = pd.concat([cv_df, mean_row, std_row])
        st.dataframe(summary.style.format("{:.4f}")
                     .background_gradient(cmap="RdYlGn", subset=["Acurácia Direcional"])
                     .background_gradient(cmap="RdYlGn_r", subset=["RMSE"]),
                     use_container_width=True)

    def _show_metrics(test_m, train_m):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("RMSE (Teste)",  f"{test_m['RMSE']:.6f}")
        c2.metric("MAE (Teste)",   f"{test_m['MAE']:.6f}")
        c3.metric("Dir. Acc.",     f"{test_m['Acurácia Direcional']:.2%}")
        c4.metric("Corr. IC",      f"{test_m['Correlação IC']:.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    with model_tabs[0]:
        st.markdown("#### Random Forest Regressor")
        if st.button("Treinar Random Forest", key="btn_rf"):
            with st.spinner("Treinando Random Forest..."):
                rf_res = run_random_forest(X_train, y_train, X_test, y_test,
                                           n_estimators=rf_n_est, max_depth=rf_depth,
                                           n_cv_splits=n_cv)
            st.session_state["rf_res"] = rf_res

        if "rf_res" in st.session_state:
            r = st.session_state["rf_res"]
            _show_metrics(r["test_metrics"], r["train_metrics"])
            _show_cv_metrics(r["cv_results"])
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.plotly_chart(_plot_predictions(
                    r["y_test"].values, r["preds_test"],
                    r["X_test"].index, "Random Forest — Teste"
                ), use_container_width=True)
            with col_r:
                st.plotly_chart(_plot_feature_importance(r["feature_importance"]), use_container_width=True)
            st.session_state["last_preds"] = {"preds": r["preds_test"],
                                               "y_true": r["y_test"].values,
                                               "dates": r["X_test"].index,
                                               "model": "Random Forest"}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    with model_tabs[1]:
        st.markdown("#### XGBoost Regressor")
        if not HAS_XGB:
            st.warning("XGBoost não instalado. Execute: `pip install xgboost`")
        else:
            if st.button("Treinar XGBoost", key="btn_xgb"):
                with st.spinner("Treinando XGBoost..."):
                    xgb_res = run_xgboost(X_train, y_train, X_test, y_test,
                                          n_estimators=xgb_n_est, learning_rate=xgb_lr,
                                          n_cv_splits=n_cv)
                st.session_state["xgb_res"] = xgb_res

            if "xgb_res" in st.session_state and st.session_state["xgb_res"]:
                r = st.session_state["xgb_res"]
                _show_metrics(r["test_metrics"], r["train_metrics"])
                _show_cv_metrics(r["cv_results"])
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.plotly_chart(_plot_predictions(
                        r["y_test"].values, r["preds_test"],
                        r["X_test"].index, "XGBoost — Teste"
                    ), use_container_width=True)
                with col_r:
                    st.plotly_chart(_plot_feature_importance(r["feature_importance"]), use_container_width=True)
                st.session_state["last_preds"] = {"preds": r["preds_test"],
                                                   "y_true": r["y_test"].values,
                                                   "dates": r["X_test"].index,
                                                   "model": "XGBoost"}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    with model_tabs[2]:
        st.markdown("#### LightGBM Regressor")
        if not HAS_LGB:
            st.warning("LightGBM não instalado. Execute: `pip install lightgbm`")
        else:
            if st.button("Treinar LightGBM", key="btn_lgb"):
                with st.spinner("Treinando LightGBM..."):
                    lgb_res = run_lightgbm(X_train, y_train, X_test, y_test,
                                           n_estimators=xgb_n_est, learning_rate=xgb_lr,
                                           n_cv_splits=n_cv)
                st.session_state["lgb_res"] = lgb_res

            if "lgb_res" in st.session_state and st.session_state["lgb_res"]:
                r = st.session_state["lgb_res"]
                _show_metrics(r["test_metrics"], r["train_metrics"])
                _show_cv_metrics(r["cv_results"])
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.plotly_chart(_plot_predictions(
                        r["y_test"].values, r["preds_test"],
                        r["X_test"].index, "LightGBM — Teste"
                    ), use_container_width=True)
                with col_r:
                    st.plotly_chart(_plot_feature_importance(r["feature_importance"]), use_container_width=True)
                st.session_state["last_preds"] = {"preds": r["preds_test"],
                                                   "y_true": r["y_test"].values,
                                                   "dates": r["X_test"].index,
                                                   "model": "LightGBM"}

    # ── LSTM ──────────────────────────────────────────────────────────────────
    with model_tabs[3]:
        st.markdown("#### LSTM — Long Short-Term Memory")
        if not HAS_TORCH:
            st.warning("PyTorch não instalado. Execute: `pip install torch`")
        else:
            if st.button("Treinar LSTM", key="btn_lstm"):
                progress_bar = st.progress(0)
                status_txt   = st.empty()
                def _cb(epoch, total, loss):
                    progress_bar.progress(epoch / total)
                    status_txt.text(f"Época {epoch}/{total} — Loss: {loss:.6f}")
                with st.spinner("Treinando LSTM..."):
                    lstm_res = run_lstm(X_train, y_train, X_test, y_test,
                                        seq_len=dl_seq_len, hidden_size=dl_hidden,
                                        num_layers=dl_layers, epochs=dl_epochs,
                                        batch_size=dl_batch, lr=dl_lr,
                                        progress_cb=_cb)
                progress_bar.empty(); status_txt.empty()
                st.session_state["lstm_res"] = lstm_res

            if "lstm_res" in st.session_state and st.session_state["lstm_res"]:
                r = st.session_state["lstm_res"]
                _show_metrics(r["test_metrics"], r["train_metrics"])

                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.plotly_chart(_plot_predictions(
                        r["y_test"], r["preds_test"],
                        X_test.index[-len(r["y_test"]):], "LSTM — Teste"
                    ), use_container_width=True)
                with col_r:
                    fig_loss = go.Figure(go.Scatter(y=r["train_losses"], mode="lines",
                                                     line=dict(color="orange")))
                    fig_loss.update_layout(height=320, title="Curva de Loss (Treino)",
                                           xaxis_title="Época", yaxis_title="MSE Loss")
                    st.plotly_chart(fig_loss, use_container_width=True)

                st.session_state["last_preds"] = {"preds": r["preds_test"],
                                                   "y_true": r["y_test"],
                                                   "dates": X_test.index[-len(r["y_test"]):],
                                                   "model": "LSTM"}

    # ── GRU ───────────────────────────────────────────────────────────────────
    with model_tabs[4]:
        st.markdown("#### GRU — Gated Recurrent Unit")
        if not HAS_TORCH:
            st.warning("PyTorch não instalado. Execute: `pip install torch`")
        else:
            if st.button("Treinar GRU", key="btn_gru"):
                progress_bar2 = st.progress(0)
                status_txt2   = st.empty()
                def _cb2(epoch, total, loss):
                    progress_bar2.progress(epoch / total)
                    status_txt2.text(f"Época {epoch}/{total} — Loss: {loss:.6f}")
                with st.spinner("Treinando GRU..."):
                    gru_res = run_gru(X_train, y_train, X_test, y_test,
                                      seq_len=dl_seq_len, hidden_size=dl_hidden,
                                      num_layers=dl_layers, epochs=dl_epochs,
                                      batch_size=dl_batch, lr=dl_lr,
                                      progress_cb=_cb2)
                progress_bar2.empty(); status_txt2.empty()
                st.session_state["gru_res"] = gru_res

            if "gru_res" in st.session_state and st.session_state["gru_res"]:
                r = st.session_state["gru_res"]
                _show_metrics(r["test_metrics"], r["train_metrics"])
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.plotly_chart(_plot_predictions(
                        r["y_test"], r["preds_test"],
                        X_test.index[-len(r["y_test"]):], "GRU — Teste"
                    ), use_container_width=True)
                with col_r:
                    fig_loss2 = go.Figure(go.Scatter(y=r["train_losses"], mode="lines",
                                                      line=dict(color="orange")))
                    fig_loss2.update_layout(height=320, title="Curva de Loss (Treino)",
                                            xaxis_title="Época", yaxis_title="MSE Loss")
                    st.plotly_chart(fig_loss2, use_container_width=True)
                st.session_state["last_preds"] = {"preds": r["preds_test"],
                                                   "y_true": r["y_test"],
                                                   "dates": X_test.index[-len(r["y_test"]):],
                                                   "model": "GRU"}

    # ── BACKTESTING ───────────────────────────────────────────────────────────
    with model_tabs[5]:
        st.markdown("#### Backtesting — Estratégia Long/Flat")
        if "last_preds" not in st.session_state:
            st.info("Treine pelo menos um modelo acima para habilitar o backtesting.")
        else:
            lp = st.session_state["last_preds"]
            st.caption(f"Usando previsões do modelo: **{lp['model']}** | {len(lp['preds'])} observações")

            bt = run_backtest(
                y_true=lp["y_true"],
                y_pred=lp["preds"],
                dates=lp["dates"],
                threshold=bt_threshold,
                transaction_cost=bt_cost,
                rf_annual=risk_free_rate,
            )

            # ── Equity Curves ──
            st.markdown("**Curvas de Equity**")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=bt["dates"], y=bt["strat_equity"],
                                         mode="lines", name=f"Estratégia ({lp['model']})",
                                         line=dict(color="steelblue")))
            fig_eq.add_trace(go.Scatter(x=bt["dates"], y=bt["bh_equity"],
                                         mode="lines", name="Buy & Hold",
                                         line=dict(color="gray", dash="dash")))
            fig_eq.update_layout(height=360, hovermode="x unified",
                                  yaxis_title="Valor da Carteira (R$1 inicial)",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── Drawdown ──
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=bt["dates"], y=bt["strat_dd"]*100,
                                         fill="tozeroy", mode="lines",
                                         name="Drawdown Estratégia", line=dict(color="red")))
            fig_dd.add_trace(go.Scatter(x=bt["dates"], y=bt["bh_dd"]*100,
                                         fill="tozeroy", mode="lines",
                                         name="Drawdown B&H", line=dict(color="orange", dash="dash")))
            fig_dd.update_layout(height=250, yaxis_title="Drawdown (%)",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── Statistics ──
            st.markdown("**Estatísticas de Performance**")
            df_stats = pd.DataFrame({
                f"Estratégia ({lp['model']})": bt["strat_stats"],
                "Buy & Hold": bt["bh_stats"],
            })
            fmt = {c: "{:.2%}" if "Ret" in c or "Vol" in c or "Draw" in c or "Taxa" in c else "{:.4f}"
                   for c in df_stats.index}
            st.dataframe(df_stats.style.format("{:.4f}"), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Nº de Operações", bt["n_trades"])
            c2.metric("Custo Total Estimado", f"{bt['n_trades'] * bt_cost:.2%}")

            # ── Monthly returns ──
            st.markdown("**Retornos Mensais da Estratégia**")
            mr = bt["monthly_returns"]
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in mr.values]
            fig_mr = go.Figure(go.Bar(x=mr.index.strftime("%Y-%m"), y=mr.values*100,
                                       marker_color=colors))
            fig_mr.update_layout(height=280, yaxis_title="Retorno (%)",
                                  xaxis_tickangle=-45)
            st.plotly_chart(fig_mr, use_container_width=True)
