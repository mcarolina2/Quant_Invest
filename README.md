# Quant_Invest 📈🤖

## 📌 Visão Geral

O **Quant_Invest** é um Sistema de Suporte à Decisão de Investimentos desenvolvido como um Web App interativo utilizando **Streamlit**. O projeto combina conceitos de **Finanças Quantitativas, Econometria Financeira, Estatística, Ciência de Dados, Machine Learning e Deep Learning** para auxiliar investidores no processo de análise, seleção e monitoramento de ativos financeiros.

A proposta do projeto é evoluir gradualmente de um sistema de análise quantitativa para uma plataforma completa de apoio à decisão em investimentos, integrando modelos econométricos clássicos, algoritmos de inteligência artificial e técnicas modernas de otimização de portfólio.

Atualmente, a plataforma permite:

- Avaliar risco e retorno de ativos;
- Estimar modelos econométricos clássicos;
- Modelar volatilidade condicional;
- Aplicar algoritmos de Machine Learning e Deep Learning para previsão de retornos;
- Realizar validação temporal dos modelos;
- Executar backtesting de estratégias quantitativas;
- Servir como base para futuras técnicas de otimização de portfólio.

---

# 🏗️ Arquitetura do Projeto

O desenvolvimento está estruturado em três níveis de maturidade analítica:

| Fase | Descrição | Status |
|--------|------------|---------|
| Fase I | Econometria e Gestão de Risco | ✅ Implementada |
| Fase II | Inteligência Artificial e Modelos Preditivos | ✅ Implementada |
| Fase III | Otimização Quantitativa de Portfólios | 🚧 Em Desenvolvimento |

---

# 🏛️ Fase I — Econometria e Gestão de Risco

Nesta etapa o sistema atua como um filtro quantitativo para avaliação de ativos da B3 e do mercado internacional.

---

## 📐 Capital Asset Pricing Model (CAPM)

O CAPM é utilizado para estimar a relação entre o retorno esperado de um ativo e o risco sistemático do mercado.

### Métricas Calculadas

- Beta (β)
- Alpha de Jensen
- R²
- p-valor dos coeficientes
- Intervalos de confiança de 95%
- Prêmio de risco
- Retorno esperado pelo CAPM
- Retorno realizado

### Características

- Regressão OLS
- Erros robustos HC3
- Inferência estatística robusta
- Comparação entre retorno esperado e realizado

---

## 🔬 Modelo de Três Fatores de Fama-French

O modelo expande o CAPM ao incorporar fatores adicionais de risco.

### Fatores

#### Mercado (MKT-RF)

Prêmio de risco associado ao mercado.

#### SMB (Small Minus Big)

Captura o prêmio associado ao tamanho das empresas.

#### HML (High Minus Low)

Captura o prêmio associado ao fator valor versus crescimento.

### Métricas

- Alpha
- Beta de Mercado
- Beta SMB
- Beta HML
- R²
- R² Ajustado
- p-valores dos coeficientes

---

## 🌊 Modelos ARCH/GARCH

Utilizados para modelar a volatilidade condicional dos retornos financeiros.

### Funcionalidades

- Ajuste de modelos GARCH(p,q)
- Volatilidade condicional
- Persistência da volatilidade
- Forecast de volatilidade
- Diagnóstico dos resíduos

### Métricas

- Log-Likelihood
- AIC
- BIC
- Persistência (α + β)
- Volatilidade prevista
- VaR baseado em GARCH

### Visualizações

- Volatilidade condicional
- Distribuição dos resíduos
- Volatilidade anualizada
- Forecast de volatilidade

---

## 📊 Diagnóstico Estatístico

O sistema executa automaticamente testes estatísticos para avaliar propriedades dos retornos financeiros.

### Testes de Normalidade

#### Jarque-Bera

Avalia se a distribuição dos retornos segue uma distribuição normal.

Métricas:

- Estatística JB
- p-valor

---

### Testes de Dependência Temporal

#### Ljung-Box

Detecta autocorrelação serial nos retornos.

Métricas:

- Estatística Ljung-Box
- p-valor

---

### Testes de Heterocedasticidade

#### ARCH-LM

Identifica a presença de agrupamento de volatilidade.

Métricas:

- Estatística ARCH
- p-valor

---

## ⚠️ Métricas de Risco

### Retorno e Volatilidade

- Retorno Médio Diário
- Retorno Anualizado
- Volatilidade Diária
- Volatilidade Anualizada

### Risco-Retorno

- Índice de Sharpe
- Índice de Sortino
- Coeficiente de Variação (CV)

### Medidas de Cauda

- VaR (95%)
- CVaR (95%)

### Estatísticas Descritivas

- Assimetria
- Curtose
- Máximo
- Mínimo

---

# 🤖 Fase II — Inteligência Artificial e Modelos Preditivos

A segunda fase adiciona um motor preditivo capaz de estimar retornos futuros utilizando técnicas modernas de Machine Learning e Deep Learning.

---

## ⚙️ Engenharia de Features

O sistema gera automaticamente variáveis preditoras para alimentar os modelos.

### Features Utilizadas

- Retornos defasados
- Médias móveis
- Momentum
- Volatilidade histórica
- Tendências de preço
- Indicadores técnicos derivados

---

## 🌲 Random Forest

Modelo baseado em ensemble de árvores de decisão.

### Características

- Captura relações não lineares
- Robusto a ruídos
- Baixo risco de overfitting

### Métricas

- RMSE
- MAE
- Correlação IC
- Acurácia Direcional

### Recursos

- Importância das variáveis
- Validação cruzada temporal
- Predição fora da amostra

---

## ⚡ XGBoost

Implementação avançada de Gradient Boosting.

### Características

- Alta capacidade preditiva
- Tratamento eficiente de não linearidades
- Regularização automática

### Métricas

- RMSE
- MAE
- Correlação IC
- Acurácia Direcional

---

## 💡 LightGBM

Implementação otimizada de Gradient Boosting baseada em histogramas.

### Vantagens

- Treinamento mais rápido
- Menor consumo de memória
- Excelente escalabilidade

### Métricas

- RMSE
- MAE
- Correlação IC
- Acurácia Direcional

---

## 🧠 Deep Learning — LSTM

Rede neural recorrente especializada em séries temporais.

### Capacidades

- Captura dependências temporais de longo prazo
- Modela relações não lineares complexas
- Identifica padrões dinâmicos do mercado

### Recursos

- Curva de Loss
- Monitoramento do treinamento
- Predições fora da amostra

---

## 🔄 Deep Learning — GRU

Versão simplificada das redes LSTM.

### Vantagens

- Menor custo computacional
- Treinamento mais rápido
- Desempenho competitivo para séries financeiras

---

# 📈 Validação Cruzada Temporal

Todos os modelos de Machine Learning são avaliados utilizando técnicas apropriadas para séries temporais.

## Time Series Cross Validation

Características:

- Divisão temporal treino/teste
- Folds temporais
- Ausência de vazamento de dados
- Avaliação robusta

### Métricas

- RMSE
- MAE
- Correlação IC
- Acurácia Direcional

---

# 💰 Backtesting

O sistema inclui um módulo de backtesting para avaliar o desempenho histórico das estratégias geradas pelos modelos preditivos.

---

## Estratégia Long/Flat

### Regras

- Compra quando o retorno previsto excede um limite configurável.
- Permanece em caixa quando o sinal previsto não é suficientemente forte.

### Parâmetros

- Threshold configurável
- Custos de transação
- Taxa livre de risco

---

## Métricas de Performance

### Retorno

- Retorno Acumulado
- Retorno Anualizado

### Risco

- Volatilidade
- Sharpe Ratio
- Drawdown Máximo

### Operacional

- Número de Operações
- Custos Totais de Transação

---

## Visualizações

- Curva de Equity
- Curva de Drawdown
- Retornos Mensais
- Comparação com Buy & Hold

---

# 🚧 Fase III — Otimização Quantitativa de Portfólio

A próxima etapa do projeto consiste na construção de um módulo de alocação ótima de ativos.

Possíveis implementações:

### Teoria Moderna do Portfólio

- Carteira de Mínima Variância
- Carteira de Máximo Sharpe
- Fronteira Eficiente

### Risk Parity

- Alocação baseada em contribuição ao risco

### Black-Litterman

- Integração entre opiniões do investidor e equilíbrio de mercado

### Otimização com Inteligência Artificial

- Construção dinâmica de portfólios utilizando previsões dos modelos de Machine Learning e Deep Learning

---

# 🛠️ Tecnologias e Bibliotecas Utilizadas

## Interface

- Streamlit

## Manipulação de Dados

- Pandas
- NumPy

## Econometria

- Statsmodels
- ARCH

## Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM

## Deep Learning

- PyTorch

## Visualização

- Plotly
- Matplotlib

## Dados Financeiros

- yFinance
- SGS (Banco Central do Brasil)

---

# 📂 Estrutura do Projeto

```text
Quant_Invest/
│
├── data/
│   └── Arquivos históricos em formato Parquet
│
├── modules/
│   ├── data_loader.py
│   ├── update_data.py
│   ├── capm.py
│   ├── fama_french.py
│   ├── garch.py
│   ├── risk_metrics.py
│   ├── features.py
│   ├── ml_models.py
│   ├── dl_models.py
│   └── backtesting.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🎯 Objetivo Acadêmico

O Quant_Invest foi concebido como um projeto de integração entre:

- Finanças Quantitativas
- Econometria Financeira
- Estatística Aplicada
- Ciência de Dados
- Machine Learning
- Deep Learning
- Engenharia de Software

O objetivo é construir uma plataforma de apoio à decisão que una metodologias acadêmicas consolidadas com técnicas modernas de inteligência artificial, permitindo a análise, previsão e avaliação sistemática de ativos financeiros.