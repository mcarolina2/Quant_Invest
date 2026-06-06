# Quant_Invest
O **Quant_Invest** é um Sistema de Suporte à Decisão de Investimentos desenvolvido na forma de um Web App interativo utilizando Streamlit. O objetivo principal do projeto é evoluir ao longo do semestre, deixando de ser um mero visualizador de dados para se transformar em uma plataforma analítica robusta que integra filtros econométricos de risco, inteligência artificial preditiva e algoritmos de otimização quantitativa de portfólios# Quant_Invest 📈🤖

---

## 📌 Visão Geral do Projeto

O desenvolvimento está estruturado de forma incremental, sendo dividido em fases de maturidade analítica:

* **Fase I (Atual):** Filtro de Risco e Econometria (Seleção Fundamentalista e Estatística de Ativos).
* **Fase II (Futura):** Modelos Preditivos com Inteligência Artificial.
* **Fase III (Futura):** Algoritmos de Otimização de Portfólio.

---

## 🏛️ Fase I: Filtro de Risco e Econometria

Nesta primeira etapa, o aplicativo atua como o primeiro "filtro" quantitativo na seleção de ativos da B3 e do mercado internacional. Ele se apoia em três pilares matemáticos e estatísticos:

### 1. Modelo CAPM (Capital Asset Pricing Model)
Estimação do coeficiente **Beta ($\beta$)** e do prêmio de risco para cada ativo, tanto de forma estática quanto dinâmica por meio de **Janelas Deslizantes (*Rolling OLS*)**. Permite mensurar a sensibilidade do ativo frente ao comportamento do benchmark de mercado (ex: Ibovespa ou S&P 500).

### 2. Modelo de Três Fatores de Fama-French
Decomposição e explicação dos retornos dos ativos além do prêmio de mercado, capturando anomalias associadas a:
* **Tamanho (SMB - *Small Minus Big*):** Prêmio por investir em empresas de menor capitalização de mercado.
* **Valor (HML - *High Minus Low*):** Prêmio por investir em ações de valor (múltiplos baixos) em detrimento de ações de crescimento.

### 3. Modelos ARCH/GARCH
Modelagem da heterocedasticidade condicional autorregressiva para analisar a dinâmica da volatilidade no tempo. Essencial para identificar o fenômeno de *volatility clustering* (agrupamento de volatilidade) e calcular riscos condicionados.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O projeto é desenvolvido puramente em **Python**, utilizando o seguinte ecossistema de bibliotecas quantitativas e de dados:

* **Streamlit:** Framework para a criação da interface web e interatividade.
* **yFinance:** Extração de dados históricos de preços diretamente do Yahoo Finance.
* **Statsmodels:** Estimação das regressões lineares ordinárias (OLS) e estruturas de rolling windows.
* **Arch:** Implementação e especificação dos modelos ARCH/GARCH para volatilidade.
* **BCB (Python-SGS):** Integração direta com o Sistema Gerenciador de Séries Temporais do Banco Central para captura da taxa Selic (Risk-Free nacional).
* **Pandas & NumPy:** Manipulação algorítmica e tratamento matricial dos dados financeiros.
* **Plotly & Matplotlib:** Geração de gráficos dinâmicos e interativos para o usuário final.

---

## 📁 Estrutura Básica do Repositório

```text
Quant_Invest/
│
├── app.py                  # Arquivo principal que executa a interface Streamlit
├── requirements.txt        # Dependências do projeto para replicação do ambiente
├── README.md               # Documentação do projeto
│
└── data/                   # Diretório opcional para armazenamento local ou cache de fatores