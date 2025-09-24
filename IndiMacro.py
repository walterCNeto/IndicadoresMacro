import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import yfinance as yf

# -------------------- Paths --------------------
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro"
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "indicadores_macro.xlsx")

# -------------------- Datas base --------------------
today = datetime.today()

def month_start(dt: datetime) -> datetime:
    return dt.replace(day=1)

def month_end(dt: datetime) -> datetime:
    return (pd.Timestamp(dt) + pd.offsets.MonthEnd(0)).to_pydatetime()

# Último mês fechado
LAST_CLOSED_MONTH_START = (month_start(today) - relativedelta(days=1)).replace(day=1)
LAST_CLOSED_MONTH_END = month_end(LAST_CLOSED_MONTH_START)

# Início da janela de 10 anos: (último dia do mês fechado) - 10 anos + 1 dia
START_10Y = (LAST_CLOSED_MONTH_END - relativedelta(years=10)) + relativedelta(days=1)

def ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def mmddyyyy_dash(dt: datetime) -> str:
    return dt.strftime("%m-%d-%Y")

# -------------------- Helpers --------------------
def ensure_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] else pd.Series(dtype=float)
    elif isinstance(x, (list, tuple, np.ndarray)):
        x = pd.Series(x)
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    try:
        x.index = pd.to_datetime(x.index)
    except Exception:
        pass
    return x.dropna()

def to_monthly_eom(s: pd.Series, how="last") -> pd.Series:
    s = ensure_series(s)
    if s.empty:
        return s
    out = s.resample("ME").mean() if how == "mean" else s.resample("ME").last()
    return ensure_series(out)

def clip_last_10y_and_last_month(s: pd.Series) -> pd.Series:
    s = ensure_series(s)
    if s.empty:
        return s
    return s[(s.index >= pd.to_datetime(ymd(START_10Y))) & (s.index <= LAST_CLOSED_MONTH_END)]

# -------------------- HTTP Session --------------------
def make_session():
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.bcb.gov.br/",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    })
    retries = Retry(total=3, backoff_factor=0.6,
                    status_forcelist=[406,408,429,500,502,503,504],
                    allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

HTTP = make_session()

# -------------------- Yahoo Finance --------------------
def fetch_yahoo(symbol: str, start_iso: str, tries: int = 2) -> pd.Series:
    for k in range(tries):
        try:
            data = yf.download(symbol, start=start_iso, progress=False, auto_adjust=False)
            if data is None or data.empty:
                raise RuntimeError("Retorno vazio do Yahoo.")
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            s = data[col].dropna()
            s.index = pd.to_datetime(s.index)
            s.name = None
            return ensure_series(s)
        except Exception as e:
            if k == tries - 1:
                print(f"[Yahoo] Falha {symbol}: {e}")
                return pd.Series(dtype=float)

# -------------------- IPEADATA (OData v4) --------------------
def fetch_ipea_series(sercodigo: str) -> pd.Series:
    """
    Busca qualquer série do Ipeadata por SERCODIGO.
    Ex.: 'IGPDI12' (apenas exemplo). Retorna Series indexada por data.
    Docs (resumo): https://ipeadata.gov.br/api/
    """
    base = "https://ipeadata.gov.br/api/odata4/ValoresSerie"
    params = {"$filter": f"SERCODIGO eq '{sercodigo}'", "$orderby": "VALDATA"}
    try:
        r = HTTP.get(base, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        vals = js.get("value", [])
        if not vals:
            return pd.Series(dtype=float)
        dt = pd.to_datetime([v["VALDATA"] for v in vals])
        x  = pd.to_numeric([v["VALVALOR"] for v in vals], errors="coerce")
        s = pd.Series(x, index=dt).sort_index()
        return ensure_series(s)
    except Exception as e:
        print(f"[Ipeadata] Falha {sercodigo}: {e}")
        return pd.Series(dtype=float)

# -------------------- SIDRA/IBGE (genérico) --------------------
def fetch_sidra_table(table: int, params: dict) -> pd.DataFrame:
    """
    Wrapper simples para a API do SIDRA (IBGE).
    Retorna DataFrame com o JSON 'values' já explodido.
    Você passa a 'tabela' e os 'params' da query.
    Docs: https://apisidra.ibge.gov.br/
    """
    url = f"https://apisidra.ibge.gov.br/values/t/{table}"
    try:
        r = HTTP.get(url, params=params, timeout=30)
        r.raise_for_status()
        arr = r.json()
        if not isinstance(arr, list) or len(arr) < 2:
            return pd.DataFrame()
        cols = list(arr[0].values())
        rows = [list(x.values()) for x in arr[1:]]
        df = pd.DataFrame(rows, columns=cols)
        return df
    except Exception as e:
        print(f"[SIDRA] Falha t={table}: {e}")
        return pd.DataFrame()

def sidra_to_series(df: pd.DataFrame, date_col: str, value_col: str, freq="M") -> pd.Series:
    """
    Converte um DF do SIDRA em Series (index datetime), escolhendo a coluna de data e valor.
    - freq="M" para mensal, "Q" para trimestral (PIB), etc.
    """
    if df.empty or date_col not in df or value_col not in df:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[value_col], errors="coerce")
    # normaliza datas (SIDRA costuma ter 'Mês (Código)' ou 'Trimestre (Código)')
    dt = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    out = pd.Series(s.values, index=dt).sort_index().dropna()
    if freq == "M":
        out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    elif freq == "Q":
        out.index = pd.to_datetime(out.index).to_period("Q").to_timestamp("Q")
    return ensure_series(out)

# -------------------- PTAX via OData (Olinda/BCB) - FATIAS ANUAIS --------------------
def fetch_ptax_usd_sell_odata_chunked(start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Puxa PTAX venda (R$/US$) via Olinda usando o dataset mais robusto:
      CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)
    Faz em FATIAS ANUAIS e concatena.
    """
    base = (
        "https://olinda.bcb.gov.br/olinda/service/PTAX/version/v1/odata/"
        "CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    )
    cur = datetime(start_date.year, start_date.month, start_date.day)
    parts = []
    while cur <= end_date:
        year_end = datetime(cur.year, 12, 31)
        chunk_end = min(year_end, end_date)
        params = {
            "@moeda": "'USD'",
            "@dataInicial": f"'{mmddyyyy_dash(cur)}'",
            "@dataFinalCotacao": f"'{mmddyyyy_dash(chunk_end)}'",
            "$top": "100000",
            "$format": "json",
        }
        try:
            r = HTTP.get(base, params=params, timeout=30)
            r.raise_for_status()
            obj = r.json()
            vals = obj.get("value", [])
            if vals:
                df = pd.DataFrame(vals)
                # campos: dataHoraCotacao, cotacaoCompra, cotacaoVenda
                df["data"] = pd.to_datetime(df["dataHoraCotacao"], errors="coerce").dt.date
                s = df.groupby("data")["cotacaoVenda"].last()
                s.index = pd.to_datetime(s.index)
                parts.append(ensure_series(s.sort_index()))
        except Exception as e:
            print(f"[PTAX-OData/MoedaPeriodo] Falha no ano {cur.year}: {e}")
        cur = chunk_end + relativedelta(days=1)

    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return ensure_series(out)

# -------------------- SGS via python-bcb (com JANELAS) --------------------
def fetch_sgs_python_bcb_chunked(series_id: int, start_dt: datetime, end_dt: datetime,
                                 years_per_chunk: int = 5) -> pd.Series:
    """Divide em janelas de até 'years_per_chunk' e concatena."""
    try:
        from bcb import sgs
    except Exception as e:
        print(f"[python-bcb] Indisponível: {e}")
        return pd.Series(dtype=float)

    pieces = []
    cur_start = start_dt
    while cur_start <= end_dt:
        cur_end = min(cur_start + relativedelta(years=years_per_chunk) - relativedelta(days=1), end_dt)
        try:
            df = sgs.get({str(series_id): series_id}, start=ymd(cur_start), end=ymd(cur_end))
            if df is not None and not df.empty:
                s = df.iloc[:, 0]
                s.index = pd.to_datetime(s.index)
                pieces.append(ensure_series(s))
        except Exception as e:
            print(f"[python-bcb] Falha série {series_id} no bloco {ymd(cur_start)} a {ymd(cur_end)}: {e}")
        cur_start = cur_end + relativedelta(days=1)

    if not pieces:
        return pd.Series(dtype=float)
    s_all = pd.concat(pieces).sort_index()
    s_all = s_all[~s_all.index.duplicated(keep="last")]
    return ensure_series(s_all)

# -------------------- FRED (API JSON) --------------------
def fetch_fred_json(series_id: str, start_iso: str) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        return pd.Series(dtype=float)
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_iso,
        "observation_end": "9999-12-31"
    }
    try:
        r = HTTP.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        dates = [o.get("date") for o in obs]
        vals = pd.to_numeric([o.get("value", "nan") for o in obs], errors="coerce")
        s = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
        return ensure_series(s)
    except Exception as e:
        print(f"[FRED-JSON] Falha {series_id}: {e}")
        return pd.Series(dtype=float)

# -------------------- CRÉDITO (SGS) — mapeamento de séries --------------------
CREDIT_SGS = {
    # Estoques em % do PIB (mensal)
    "Saldo crédito total (% PIB)": 20622,     # dadosabertos.bcb.gov.br dataset/20622
    "Saldo crédito PF (% PIB)":    20624,     # dataset/20624
    "Saldo crédito outros setores (% PIB)": 22070,  # dataset/22070
    "Saldo crédito PJ (% PIB)":    20623,      # << NOVO (PJ total)  ──> dadosabertos 20623
    "Saldo crédito PJ dir. (% PIB)": 20629,    # << NOVO (PJ direcionado) ──> dadosabertos 20629
    "Inadimplência PJ total (>=90d, %)": 21083,   # << NOVO (PJ total)
    "Inadimplência PJ livres (>=90d, %)": 21086,  # << NOVO (PJ recursos livres)
    "Inadimplência PF total (>=90d, %)": 21084,  # PF total (livre + direcionado)
    "Inadimplência PF livres (>=90d, %)": 21112,  # PF - recursos livres (agregado)

    # Inadimplência (% com atraso >= 90 dias) — total
    "Inadimplência total (>=90d, %)": 21082,  # dataset/21082

    # Exemplos de livres / faixas (adicione os que desejar):
    # "Inadimplência livres PJ (15-90d, %)": 21015,  # dataset/21015 (exemplo)
    # "Inadimplência livres PF (15-90d, %)": 21007,  # dataset/21007 (exemplo)

    # Famílias: endividamento/comprometimento da renda
    "Endividamento famílias (% renda, 12m)": 29038,  # dataset/29038
    "Comprometimento renda famílias (% renda, média mov. trimestral)": 29265,  # dataset/29265
}

# -------------------- Main --------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_iso = ymd(START_10Y)
    series = {}

    # --------- SGS (Brasil) - últimos 10 anos, em blocos ---------
    selic = fetch_sgs_python_bcb_chunked(432, START_10Y, LAST_CLOSED_MONTH_END, years_per_chunk=5)
    series["Taxa Selic (% a.a.) [meta]"] = clip_last_10y_and_last_month(to_monthly_eom(selic, how="last"))

    ibc = fetch_sgs_python_bcb_chunked(24364, START_10Y, LAST_CLOSED_MONTH_END, years_per_chunk=5)
    series["IBC-Br dessaz. (índice 2002=100)"] = clip_last_10y_and_last_month(to_monthly_eom(ibc, how="last"))

    # --------- Indicadores de CRÉDITO (SGS) ---------
    for nice_name, sid in CREDIT_SGS.items():
        s = fetch_sgs_python_bcb_chunked(sid, START_10Y, LAST_CLOSED_MONTH_END, years_per_chunk=5)
        # A maioria é mensal; agrego por último dia do mês (last)
        series[nice_name] = clip_last_10y_and_last_month(to_monthly_eom(s, how="last"))

    # --------- Câmbio ---------
    ptax = fetch_ptax_usd_sell_odata_chunked(START_10Y, LAST_CLOSED_MONTH_END)
    if ptax.empty:
        print("[PTAX] OData indisponível — usando proxy Yahoo (USDBRL=X) para não ficar vazio.")
        ptax = fetch_yahoo("USDBRL=X", start_iso)  # proxy
    series["BRL/USD (PTAX - venda)"] = clip_last_10y_and_last_month(to_monthly_eom(ptax, how="last"))

    series["USD/BRL (mercado, Yahoo)"] = clip_last_10y_and_last_month(
        to_monthly_eom(fetch_yahoo("USDBRL=X", start_iso), how="last")
    )
    series["EUR/USD (nível)"] = clip_last_10y_and_last_month(
        to_monthly_eom(fetch_yahoo("EURUSD=X", start_iso), how="last")
    )

    # --------- Bolsas / Commodities ---------
    series["Ibovespa (nível)"] = clip_last_10y_and_last_month(to_monthly_eom(fetch_yahoo("^BVSP", start_iso), how="last"))
    series["S&P 500 (nível)"]  = clip_last_10y_and_last_month(to_monthly_eom(fetch_yahoo("^GSPC", start_iso), how="last"))
    series["VIX (nível)"]      = clip_last_10y_and_last_month(to_monthly_eom(fetch_yahoo("^VIX",  start_iso), how="last"))
    series["Brent (US$/bbl)"]  = clip_last_10y_and_last_month(to_monthly_eom(fetch_yahoo("BZ=F",  start_iso), how="last"))
    series["Commodities (proxy DBC)"] = clip_last_10y_and_last_month(to_monthly_eom(fetch_yahoo("DBC", start_iso), how="last"))

    # --------- EUA (FRED JSON; fallback Yahoo ^TNX/10) ---------
    ust10 = fetch_fred_json("DGS10", start_iso)
    if ust10.empty:
        tnx = fetch_yahoo("^TNX", start_iso)  # ^TNX = 10y * 10
        ust10 = tnx / 10.0
    series["Treasury 10 anos (% a.a.)"] = clip_last_10y_and_last_month(to_monthly_eom(ust10, how="last"))

    ff_u = fetch_fred_json("DFEDTARU", start_iso)
    ff_l = fetch_fred_json("DFEDTARL", start_iso)
    if not ff_u.empty and not ff_l.empty:
        fed_mid = ensure_series((ff_u + ff_l) / 2.0)
        series["Fed Funds (média da banda, % a.a.)"] = clip_last_10y_and_last_month(to_monthly_eom(fed_mid, how="last"))
        
    # --------- Consolidação ---------
    series_clean = {k: ensure_series(v) for k, v in series.items()}
    all_idx = None
    for s in series_clean.values():
        if s.empty: 
            continue
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    if all_idx is None:
        all_idx = pd.DatetimeIndex([])

    for k, s in series_clean.items():
        series_clean[k] = s.reindex(all_idx)

    df = pd.DataFrame(series_clean)

    desired = [
        "IBC-Br dessaz. (índice 2002=100)",
        "Taxa Selic (% a.a.) [meta]",
        "BRL/USD (PTAX - venda)",
        "USD/BRL (mercado, Yahoo)",
        "EUR/USD (nível)",
        "Ibovespa (nível)",
        "S&P 500 (nível)",
        "VIX (nível)",
        "Brent (US$/bbl)",
        "Commodities (proxy DBC)",
        "Fed Funds (média da banda, % a.a.)",
        "Treasury 10 anos (% a.a.)",

        # --- Crédito (novo) ---
        # --- Crédito (PF/PJ) ---
        "Saldo crédito total (% PIB)",
        "Saldo crédito PF (% PIB)",
        "Saldo crédito PJ (% PIB)",
        "Saldo crédito PJ dir. (% PIB)",
        "Saldo crédito outros setores (% PIB)",
        "Inadimplência total (>=90d, %)",
        "Inadimplência PF total (>=90d, %)",       # << NOVO
        "Inadimplência PF livres (>=90d, %)",      # << NOVO
        "Inadimplência PJ total (>=90d, %)",
        "Inadimplência PJ livres (>=90d, %)",
        "Endividamento famílias (% renda, 12m)",
        "Comprometimento renda famílias (% renda, média mov. trimestral)",
        # "Inadimplência livres PJ (15-90d, %)",
        # "Inadimplência livres PF (15-90d, %)",
    ]

    cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    df = df[cols]

    # --------- Excel ---------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        df.to_excel(writer, sheet_name="Indicadores (mensal)")

        meta = pd.DataFrame({
            "Indicador": cols,
            "Fonte": [
                "BCB/SGS via python-bcb (24364)" if c.startswith("IBC-Br") else
                "BCB/SGS via python-bcb (432)"  if c.startswith("Taxa Selic") else
                "BCB PTAX OData (ou proxy Yahoo se OData indisponível)" if c.startswith("BRL/USD (PTAX") else
                "Yahoo Finance (USDBRL=X)" if c.startswith("USD/BRL") else
                "Yahoo Finance (EURUSD=X)" if c.startswith("EUR/USD") else
                "Yahoo Finance (^BVSP)"    if c.startswith("Ibovespa") else
                "Yahoo Finance (^GSPC)"    if c.startswith("S&P 500") else
                "BCB/SGS (20622/20624/20623/20629/22070)"  if c.startswith("Saldo crédito") else                
                "BCB/SGS (21082/21083/21086/21004/21112)" if c.startswith("Inadimplência total") else
                "BCB/SGS (29038/29265)"    if c.startswith("Endividamento") or c.startswith("Comprometimento") else
                "Yahoo Finance (^VIX)"     if c.startswith("VIX") else
                "Yahoo Finance (BZ=F)"     if c.startswith("Brent") else
                "Yahoo Finance (DBC ETF)"  if c.startswith("Commodities") else
                "FRED API (DFEDTARU/DFEDTARL -> média)" if c.startswith("Fed Funds") else
                "FRED API (DGS10) ou Yahoo (^TNX/10)"   if c.startswith("Treasury 10 anos") else
                "—"
                for c in cols
            ],
            "Frequência": ["Mensal (fim do mês)"] * len(cols),
            "Observação": [
                "Índice dessazonalizado"              if c.startswith("IBC-Br") else
                "Meta do Copom (% a.a.)"              if c.startswith("Taxa Selic") else
                "PTAX venda (R$/US$). Pode ser proxy Yahoo se OData falhar." if c.startswith("BRL/USD (PTAX") else                
                "Série mensal; % do PIB (12m)"  if c.startswith("Saldo crédito") else
                "Parcela >=90 dias em atraso"   if ">=90d" in c else
                "Parcela com atraso entre 15 e 90 dias" if "15-90d" in c else

                "% da renda (conceitos BCB)"    if c.startswith("Endividamento") or c.startswith("Comprometimento") else
                "Fechamento mensal (último dia útil)" if any(c.startswith(x) for x in [
                    "Ibovespa", "S&P 500", "VIX", "Brent", "Commodities", "USD/BRL", "EUR/USD"
                ]) else
                "Média da banda (upper/lower)"        if c.startswith("Fed Funds") else
                "Yield nominal EUA 10y"               if c.startswith("Treasury 10 anos") else
                ""
                for c in cols
            ]
        })
        meta.to_excel(writer, sheet_name="Metadados", index=False)

    print(f"OK! Excel salvo em: {OUTPUT_XLSX}")

# -------------------- Run --------------------
if __name__ == "__main__":
    main()

##########Analises###########

#1) VAR (previsão multi-variada + IRFs)

import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, sheet_name="Indicadores (mensal)", index_col=0, parse_dates=True)

# --- Seleção das séries e transformações para estacionar ---
# Log-níveis para índices/preços; difs para torná-las estacionárias
def ld(x): return np.log(x).diff()
def d(x):  return x.diff()

panel = pd.DataFrame({
    "g_ibc":      ld(df.get("IBC-Br dessaz. (índice 2002=100)")),
    "d_usdbrl":   ld(df.get("USD/BRL (mercado, Yahoo)")),
    "d_brent":    ld(df.get("Brent (US$/bbl)")),
    "ret_spx":    ld(df.get("S&P 500 (nível)")),
    "selic":      df.get("Taxa Selic (% a.a.) [meta]").pct_change(1),  # variação p/ aproximar estacion.
    "ust10":      df.get("Treasury 10 anos (% a.a.)").pct_change(1),
})

panel = panel.dropna()
# mantém só últimos 10 anos (o Excel já vem ~10y, mas garantimos)
panel = panel.loc[panel.index >= (panel.index.max() - pd.DateOffset(years=10))]

# --- Escolha automática de defasagens ---
model = VAR(panel)
sel = model.select_order(maxlags=12)   # critérios AIC/HQ/SC/FPE
p = sel.aic or 6                        # fallback
res = model.fit(p)

print(res.summary())

# --- Previsão (12 meses à frente) ---
fcst = res.forecast(panel.values[-p:], steps=12)
fcst_idx = pd.date_range(panel.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
fcst_df = pd.DataFrame(fcst, index=fcst_idx, columns=panel.columns)
print("\nForecast 12m:\n", fcst_df.head())

# --- IRFs (respostas a choque estrutural simples) ---
irf = res.irf(12)
irf.plot(orth=False)
plt.show()

#2) VECM (cointegração: níveis com relação de longo prazo)

import pandas as pd, numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, sheet_name="Indicadores (mensal)", index_col=0, parse_dates=True)

# ---- nomes esperados ----
col_usd = "USD/BRL (mercado, Yahoo)"
col_selic = "Taxa Selic (% a.a.) [meta]"
col_fed = "Fed Funds (média da banda, % a.a.)"
col_ust10 = "Treasury 10 anos (% a.a.)"

# ---- checagem das colunas disponíveis ----
have = set(df.columns)
missing = [c for c in [col_usd, col_selic, col_fed] if c not in have]
if missing:
    print("[Aviso] Faltam no Excel:", missing)

# Fed Funds: se não existir, usa proxy (UST10)
if col_fed not in have:
    if col_ust10 in have:
        print("[Info] Usando UST 10y como proxy do Fed Funds para construir o spread.")
        df[col_fed] = df[col_ust10]
    else:
        raise RuntimeError("Nem 'Fed Funds' nem 'Treasury 10 anos' estão no Excel. "
                           "Ative a FRED_API_KEY na v4.1 do pipeline ou inclua UST10.")

# ---- montar painel com o que existe ----
want = [col_usd, col_selic, col_fed]
avail = [c for c in want if c in df.columns]
panel = df[avail].copy()

# renomear dinamicamente
rename_map = {
    col_usd: "usdbrl",
    col_selic: "selic",
    col_fed: "fed"
}
panel = panel.rename(columns=rename_map)

# garantir que temos as 3 (ou pelo menos usdbrl & selic/fed)
need = {"usdbrl", "selic", "fed"}
if not need.issubset(set(panel.columns)):
    raise RuntimeError(f"Colunas insuficientes para VECM. Tenho {list(panel.columns)}, preciso de {list(need)}.")

# construir o spread e recortar últimos 10 anos
panel["spread"] = panel["selic"] - panel["fed"]
Z = panel[["usdbrl", "spread"]].dropna()

# recorte de janela (só por garantia)
if len(Z) == 0:
    raise RuntimeError("Sem dados após o dropna. Confira se as colunas não estão inteiramente NaN.")
Z = Z.loc[Z.index >= (Z.index.max() - pd.DateOffset(years=10))]

# checar amostra mínima (VECM precisa de observações o suficiente)
if len(Z) < 36:
    raise RuntimeError(f"Amostra muito curta para VECM/Johansen: {len(Z)} obs (<36).")

# ---- Johansen (cointegração) ----
# k_ar_diff = número de defasagens em diferenças (padrão 2 é ok para mensal como exemplo)
jres = coint_johansen(Z, det_order=0, k_ar_diff=2)  # sem determinísticos
print("Eigenvalues:", jres.eig)
print("Trace stats:", jres.lr1)
print("Trace crit (90/95/99%):\n", jres.cvt)

# regra prática: se o primeiro trace > crit de 95%, há pelo menos 1 relação
has_coint = jres.lr1[0] > jres.cvt[0,1]
print(f"Cointegração detectada (95%)? {has_coint}")

# ---- Estima VECM (se r=1) ----
if has_coint:
    vecm = VECM(Z, k_ar_diff=2, coint_rank=1, deterministic="co")  # constante só no termo de cointegração
    res = vecm.fit()
    print(res.summary())

    # previsão (em níveis) para 12 meses
    fc = res.predict(steps=12)  # DataFrame com 'usdbrl' e 'spread'
    print("\nPrevisão 12m (primeiras linhas):\n", fc.head())
else:
    print("Sem cointegração significativa a 95%. Sugestão: usar VAR em diferenças.")
    
# 3) Fator Dinâmico (PCA) + modelo de previsão de Ibovespa

import pandas as pd, numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, "Indicadores (mensal)", index_col=0, parse_dates=True)

def ld(x): return np.log(x).diff()

panel = pd.DataFrame({
    "ret_ibov": ld(df.get("Ibovespa (nível)")),
    "g_ibc":    ld(df.get("IBC-Br dessaz. (índice 2002=100)")),
    "d_usd":    ld(df.get("USD/BRL (mercado, Yahoo)")),
    "d_brent":  ld(df.get("Brent (US$/bbl)")),
    "d_spx":    ld(df.get("S&P 500 (nível)")),
    "selic":    df.get("Taxa Selic (% a.a.) [meta]").pct_change(),
    "ust10":    df.get("Treasury 10 anos (% a.a.)").pct_change(),
    "vix":      ld(df.get("VIX (nível)")),
}).dropna()

# janela de 10 anos (só por garantia)
panel = panel.loc[panel.index >= (panel.index.max() - pd.DateOffset(years=10))]

# X para PCA (sem a variável-alvo)
X = panel.drop(columns=["ret_ibov"])
Xz = (X - X.mean()) / X.std(ddof=0)

# 2 fatores principais
pca = PCA(n_components=2).fit(Xz)
factors = pd.DataFrame(pca.transform(Xz), index=X.index, columns=["F1","F2"])

Y = panel["ret_ibov"].copy()

# ----- Regressão ARX com defasagens -----
# Treino com as MESMAS variáveis defasadas que usaremos no predict
lags = pd.concat(
    [
        Y.shift(1).rename("ret_ibov_lag1"),
        factors["F1"].shift(1).rename("F1_lag1"),
        factors["F2"].shift(1).rename("F2_lag1"),
    ],
    axis=1
).dropna()

Y_ = Y.loc[lags.index]
X_ = sm.add_constant(lags, has_constant="add")

res = sm.OLS(Y_, X_).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(res.summary())

# ----- Previsão (1 passo à frente) -----
# Use a ÚLTIMA linha disponível das variáveis defasadas
x_last = lags.iloc[[-1]]
X_last = sm.add_constant(x_last, has_constant="add")
signal = float(res.predict(X_last))
print("Previsão de retorno (próximo mês):", signal)

#4) Regressão com Troca de Regime (Markov-Switching) no Ibovespa

import pandas as pd, numpy as np
import statsmodels.api as sm

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, "Indicadores (mensal)", index_col=0, parse_dates=True)
ret_ibov = np.log(df.get("Ibovespa (nível)")).diff().dropna()
ret_ibov = ret_ibov.loc[ret_ibov.index >= (ret_ibov.index.max() - pd.DateOffset(years=10))]

mod = sm.tsa.MarkovRegression(ret_ibov, k_regimes=2, trend="c", switching_variance=True)
res = mod.fit()
print(res.summary())
# Probabilidade filtrada do "regime ruim" (alta vol)
probs = res.smoothed_marginal_probabilities[0]  # regime 0, por ex.
print(probs.tail())

# Sinal: reduzir exposição quando prob_regime_ruim > 0.7