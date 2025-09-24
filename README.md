IndicadoresMacro

Pipeline completo para baixar, limpar e consolidar indicadores macroeconômicos (Brasil e exterior) e rodar modelos de séries temporais (VAR, VECM, PCA–ARX e Markov-Switching).
A saída principal é um Excel mensal indicadores_macro.xlsx e um relatório .tex com metodologia e resultados.

Visão geral

Janela temporal: últimos 10 anos, até o último mês fechado.

Agregação mensal: séries diárias/intra-mensais são reamostradas para fim do mês (ME) via último valor (last) ou média (mean).

Robustez: chamadas em blocos (BCB/SGS), fallbacks (PTAX → Yahoo) e alinhamento por índice temporal unificado.

Fontes de dados

Brasil

BCB/SGS via python-bcb (Selic meta, crédito, inadimplência, endividamento/comprometimento).

PTAX – Olinda/OData (R$/US$ venda), com fallback via Yahoo.

SIDRA/IBGE (estrutura pronta para PIB trimestral, PNAD, PIM-PF, PMC, PMS, IPCA) — opcional.

Ipeadata (estrutura pronta para confiança FGV, IGP-M, fiscal) — opcional.

Exterior

FRED (Treasury 10Y, Fed Funds, PIB EUA % a/a). Treasury tem fallback via Yahoo ^TNX/10.

Unidades principais

Juros: % a.a. (o script corrige automaticamente se vier em fração).

Câmbio e índices: níveis (as transformações para modelos são feitas no código).

Crédito: % do PIB (12m, quando aplicável).

Inadimplência: % da carteira (≥90 dias).

Endividamento/Comprometimento: % da renda (conceitos BCB).

Estrutura do projeto
IndicadoresMacro/
├─ indimacrov2.py                 # script principal: coleta + modelos
├─ indicadores_macro.xlsx         # saída consolidada (gerada)
├─ relatorio.tex                  # LaTeX com metodologia e análises
├─ Figure 2025-09-24 181629.png   # IRFs do VAR (gerada)
├─ README.md
└─ requirements.txt               # opcional

Requisitos

Python ≥ 3.10

Pacotes:

pandas
numpy
requests
yfinance
python-dateutil
statsmodels
scikit-learn
xlsxwriter
bcb


Instalação:

pip install -r requirements.txt
# ou
pip install pandas numpy requests yfinance python-dateutil statsmodels scikit-learn xlsxwriter bcb


Variáveis de ambiente (opcional):

FRED_API_KEY para acesso ao FRED (Fed Funds e PIB EUA).

# Windows (PowerShell)
setx FRED_API_KEY "SUA_CHAVE_AQUI"

Como executar

Ajuste o diretório de saída no topo do script:

OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro"


Execute:

python indimacrov2.py


Saídas principais:

indicadores_macro.xlsx (abas: “Indicadores (mensal)”, “Metadados”, “Medias anuais”).

Figure 2025-09-24 181629.png (IRFs do VAR).

Compilação do LaTeX (local):

pdflatex relatorio.tex

Indicadores incluídos (exemplos já mapeados)

IBC-Br dessazonalizado (índice 2002=100) — BCB/SGS

Taxa Selic (% a.a.) [meta] — BCB/SGS

BRL/USD (PTAX – venda) — BCB/OData, fallback Yahoo

USD/BRL, EUR/USD, Ibovespa, S&P 500, VIX, Brent, DBC — Yahoo

Treasury 10 anos (% a.a.) — FRED (fallback ^TNX/10)

Fed Funds (média da banda, % a.a.) — FRED (se FRED_API_KEY)

PIB EUA (% a/a) — FRED (A191RL1Q225SBEA)

Crédito (total, PF, PJ, PJ direcionado, outros) — % do PIB

Inadimplência total, PF e PJ (livres e total) — % da carteira (≥90d)

Endividamento famílias (% renda, 12m) e Comprometimento (% renda, média móvel trimestral)

Mapeamentos SIDRA, Ipeadata e SGS extra estão preparados no código para você preencher os IDs adicionais.

Modelos implementados
1) VAR (previsão multivariada + IRFs)

Painel: g_ibc (Δlog IBC-Br), d_usdbrl (Δlog USD/BRL), d_brent (Δlog Brent), ret_spx (Δlog S&P 500), selic (Δ Selic), ust10 (Δ UST10).

Defasagens escolhidas por AIC.

Saídas: previsão 12m (impresso), IRFs salvas em Figure 2025-09-24 181629.png.

Resumo dos resultados observados:

Baixa significância individual dos coeficientes (amostra curta e multicolinearidade), mas sinais plausíveis em alguns vínculos (ex.: UST10 afetando atividade).

IRFs com respostas qualitativas coerentes entre fatores globais (Brent, S&P) e condições financeiras.

2) VECM (cointegração)

Par: usdbrl (nível) e spread (selic - fed).

Teste de Johansen (trace): sem cointegração a 95% na amostra corrente (Fed Funds via FRED ou proxy UST10).

Ação: usar VAR em diferenças ou ampliar amostra/variáveis (ex.: termos de troca).

3) PCA–ARX (fatores dinâmicos para Ibovespa)

PCA em variáveis padronizadas (atividade, câmbio, commodities, juros, risco).

Regressão: ret_ibov ~ const + ret_ibov(-1) + F1(-1) + F2(-1) (HAC).

R² in-sample baixo (≈0,7%), comum em retorno mensal.

Sinal de 1 passo: previsão positiva (~0,9% no próximo mês na rodada documentada).

Recomendações: avaliação out-of-sample (walk-forward), regularização e mais fatores.

4) Markov-Switching (2 regimes)

MS em retornos do Ibovespa com média e variância por regime.

Avisos de convergência podem ocorrer; a probabilidade suavizada do regime “ruim” é o insumo para regra de risco.

Exemplo: reduzir exposição quando P(regime ruim) > 0.7.

Boas práticas e limitações

Conferir “Metadados” no Excel para ver a fonte efetiva (inclui fallbacks).

VAR requer estacionariedade; VECM usa níveis apenas com cointegração.

Janela de 10 anos pode ser curta para alguns testes; valide fora da amostra.

Sempre considerar custos de transação e robustez dos sinais.

Para reprodutibilidade, congele versões em requirements.txt.

Roadmap

Mapear SIDRA (PIB Brasil, PNAD, PIM-PF, PMC, PMS, IPCA e livres x administrados).

Adicionar fiscal (primário R$ e %PIB), DBGG, balança comercial (Secex).

Novos modelos: DFM/Kalman, TVP-VAR, regime-switching com covariáveis.

Backtests para sinais (PCA–ARX condicionado ao regime MS).
