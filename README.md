# Indicadores Macroeconômicos para Análise

Pipeline para **baixar, limpar e consolidar** indicadores macroeconômicos (Brasil e exterior) e rodar modelos de séries temporais (**VAR, VECM, PCA–ARX, Markov-Switching**).  
A saída principal é um Excel mensal `indicadores_macro.xlsx` e um relatório LaTeX com metodologia e resultados.

---

## Visão geral

- **Janela temporal:** últimos 10 anos, até o **último mês fechado**.  
- **Agregação mensal:** séries diárias/intra-mensais → fim do mês (ME) usando `last` (ou `mean`, quando aplicável).  
- **Robustez:** chamadas em **blocos** (BCB/SGS), **fallbacks** (PTAX → Yahoo), e **alinhamento** em índice temporal único.

---

## Fontes de dados

### Brasil
- **BCB/SGS** via `python-bcb`  
  (Selic meta, crédito, inadimplência, endividamento/comprometimento).
- **PTAX – Olinda/OData** (R$/US$ venda)  
  *Fallback* via Yahoo (`USDBRL=X`) se OData falhar.
- **SIDRA/IBGE** *(estrutura pronta no código)*  
  PIB trimestral, PNAD, PIM-PF, PMC, PMS, IPCA (e livres x administrados).
- **Ipeadata** *(estrutura pronta)*  
  Confiança FGV, IGP-M, fiscal (resultado primário, etc.).

### Exterior
- **FRED**: Treasury 10Y, **Fed Funds**, **PIB EUA (% a/a)**.  
  *Fallback* do Treasury via Yahoo `^TNX`/10.

### Unidades (padronização)
- Juros: **% a.a.** (o script corrige se vier em fração).  
- Câmbio/índices: **níveis** (transformações feitas nos modelos).  
- Crédito: **% do PIB** (12m, quando aplicável).  
- Inadimplência: **% da carteira (≥90d)**.  
- Endividamento/Comprometimento: **% da renda** (conceitos BCB).

---

## Estrutura

IndicadoresMacro/
├─ indimacrov2.py # coleta + consolidação + modelos
├─ indicadores_macro.xlsx # saída (gerado)
├─ relatorio.tex # LaTeX com metodologia/resultados
├─ Figure 2025-09-24 181629.png # IRFs do VAR (gerado)
├─ README.md
└─ requirements.txt # opcional

### Indicadores já mapeados
- IBC-Br dessaz. (índice 2002=100) — BCB/SGS
- Selic meta (% a.a.) — BCB/SGS
- BRL/USD (PTAX – venda) — BCB/OData (fallback Yahoo)
- USD/BRL, EUR/USD, Ibovespa, S&P 500, VIX, Brent, DBC — Yahoo
- Treasury 10 anos (% a.a.) — FRED (fallback ^TNX/10)
- Fed Funds (média da banda, % a.a.) — FRED
- PIB EUA (% a/a) — FRED (A191RL1Q225SBEA)
- Crédito: total, PF, PJ, PJ direcionado, outros — % do PIB
- Inadimplência: total, PF/PJ (livres e total) — % da carteira (≥90d)
- Famílias: Endividamento (% renda, 12m) e Comprometimento (% renda, MM3)
- Mapas SIDRA, Ipeadata e SGS extra estão prontos no código para você apenas preencher os IDs

### Modelos implementados
VAR (previsão multivariada + IRFs)
Painel (mensal):
g_ibc = Δlog(IBC-Br), d_usdbrl = Δlog(USD/BRL), d_brent = Δlog(Brent),
ret_spx = Δlog(S&P500), selic = Δ Selic, ust10 = Δ UST10.
Defasagens escolhidas por AIC.
Saídas: previsão 12m (impresso no console) e IRFs (PNG).
Observações desta rodada: coeficientes individuais pouco significativos (amostra curta), mas relações qualitativas coerentes nas IRFs entre fatores globais e condições financeiras.

### VECM (cointegração)
Par em nível: usdbrl e spread = selic − fed.
Teste de Johansen (trace): sem cointegração a 95% na amostra atual
(Fed Funds via FRED ou proxy UST10).
Sugerido: VAR em diferenças ou ampliar amostra/variáveis.

### PCA–ARX (fatores dinâmicos → Ibovespa)
PCA em variáveis padronizadas (atividade, câmbio, commodities, juros, risco).
Regressão: ret_ibov ~ const + ret_ibov(-1) + F1(-1) + F2(-1) com erros HAC.
R² in-sample baixo (≈ 0,7% — típico).
Sinal 1-passo desta rodada: retorno previsto positivo (~0,9%/mês).
Próximos passos: walk-forward, regularização, mais fatores.

### Markov-Switching (2 regimes)
MS em retornos do Ibovespa com média/variância por regime.
Pode emitir avisos de convergência; use a probabilidade suavizada do regime ruim como gatilho de risco.
Regra exemplo: reduzir exposição quando P(regime ruim) > 0.7.
