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
    if df.empty or date_col not in df or value_col not in df:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[value_col], errors="coerce")
    dt = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    out = pd.Series(s.values, index=dt).sort_index().dropna()
    if freq == "M":
        out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    elif freq == "Q":
        out.index = pd.to_datetime(out.index).to_period("Q").to_timestamp("Q")
    return ensure_series(out)

# -------------------- PTAX via OData (Olinda/BCB) - FATIAS ANUAIS --------------------
def fetch_ptax_usd_sell_odata_chunked(start_date: datetime, end_date: datetime) -> pd.Series:
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
    "Saldo crédito total (% PIB)": 20622,
    "Saldo crédito PF (% PIB)":    20624,
    "Saldo crédito outros setores (% PIB)": 22070,
    "Saldo crédito PJ (% PIB)":    20623,
    "Saldo crédito PJ dir. (% PIB)": 20629,

    # Inadimplência (% da carteira)
    "Inadimplência total (>=90d, %)": 21082,
    "Inadimplência PJ total (>=90d, %)": 21083,
    "Inadimplência PJ livres (>=90d, %)": 21086,
    "Inadimplência PF total (>=90d, %)": 21084,
    "Inadimplência PF livres (>=90d, %)": 21112,

    # Famílias: endividamento/comprometimento da renda
    "Endividamento famílias (% renda, 12m)": 29038,
    "Comprometimento renda famílias (% renda, média mov. trimestral)": 29265,
}

# ----- MAPAS DE SÉRIES (preencha quando tiver os códigos) -----
IPEA_MAP = {
    # "IGP-M (% acumulado no ano)": "SEU_SERCODIGO_AQUI",
    # "Confiança consumidor (FGV)": "SEU_SERCODIGO_AQUI",
    # "Confiança empresarial (FGV)": "SEU_SERCODIGO_AQUI",
    # "Resultado primário (% PIB)": "SEU_SERCODIGO_AQUI",
}

SIDRA_MAP = {
    # "PIB Brasil (% a/a)": {
    #     "table": 1846,
    #     "params": {"n1":"all","p":"all","v":"XXXX","d":"2"},
    #     "date_col": "Trimestre (Código)", "value_col": "Valor",
    #     "freq": "Q", "transform": "YoY"
    # },
    # "Desemprego 12m (PNAD)": {
    #     "table": 4099,
    #     "params": {"n1":"all","p":"all","v":"XXXX","d":"2"},
    #     "date_col": "Mês (Código)", "value_col": "Valor",
    #     "freq": "M", "transform": "MA12"
    # },
    # "PIM-PF (%12m)": {...},
    # "PMC ampliado (%12m)": {...},
    # "PMS (%12m)": {...},
    # "IPCA (% acumulado no ano)": {...},
    # "IPCA livre (% acumulado no ano)": {...},
}

SGS_EXTRA = {
    # "Resultado primário (R$ Bilhões)": 99999,
    # "Resultado primário (% PIB)":      99998,
    # "DBGG (% PIB)":                    99997,
    # "Saldo balança comercial (US$)":   99996,
    # "Crédito doméstico (R$ Bilhões)":  99995,
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
        series[nice_name] = clip_last_10y_and_last_month(to_monthly_eom(s, how="last"))

    # --------- Câmbio ---------
    ptax = fetch_ptax_usd_sell_odata_chunked(START_10Y, LAST_CLOSED_MONTH_END)
    if ptax.empty:
        print("[PTAX] OData indisponível — usando proxy Yahoo (USDBRL=X) para não ficar vazio.")
        ptax = fetch_yahoo("USDBRL=X", start_iso)
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
    used_source = "FRED"
    if ust10.empty:
        tnx = fetch_yahoo("^TNX", start_iso)  # ^TNX = 10y * 10
        ust10 = tnx / 10.0
        used_source = "Yahoo(^TNX)/10"
    
    # --- FIX DE UNIDADE ---
    # Se veio do FRED, os valores típicos são ~2–5; se por algum motivo estiverem <1, reescala ×10.
    # Se algum dia vier como fração (0.02 = 2%), reescala ×100.
    if not ust10.empty:
        med = ust10.dropna().median()
        if med < 0.2:          # e.g., 0.04 = 4% -> estava em fração; vira %.
            ust10 = ust10 * 100.0
        elif med < 1.0:        # e.g., 0.45 -> deveria ser 4.5%
            ust10 = ust10 * 10.0
    
    series["Treasury 10 anos (% a.a.)"] = clip_last_10y_and_last_month(to_monthly_eom(ust10, how="last"))

    ff_u = fetch_fred_json("DFEDTARU", start_iso)
    ff_l = fetch_fred_json("DFEDTARL", start_iso)
    if not ff_u.empty and not ff_l.empty:
        fed_mid = ensure_series((ff_u + ff_l) / 2.0)
        series["Fed Funds (média da banda, % a.a.)"] = clip_last_10y_and_last_month(to_monthly_eom(fed_mid, how="last"))

    # --- PIB EUA (% a/a) — trimestral no FRED, replicado para mensal no fim do trimestre
    us_gdp_yoy = fetch_fred_json("A191RL1Q225SBEA", start_iso)  # Percent Change from Year Ago
    if not us_gdp_yoy.empty:
        us_gdp_yoy_q = us_gdp_yoy.copy()
        # garante marcação no fim do trimestre
        us_gdp_yoy_q.index = us_gdp_yoy_q.index.to_period("Q").to_timestamp("Q")
        series["PIB EUA (% a/a)"] = clip_last_10y_and_last_month(to_monthly_eom(us_gdp_yoy_q, how="last"))

    # --------- IPEADATA (se houver códigos mapeados) ---------
    for nice, code in IPEA_MAP.items():
        s = fetch_ipea_series(code)
        series[nice] = clip_last_10y_and_last_month(to_monthly_eom(s, how="last"))

    # --------- SIDRA (se houver tabelas mapeadas) ---------
    for nice, cfg in SIDRA_MAP.items():
        t = cfg["table"]; params = cfg["params"]
        df_sidra = fetch_sidra_table(t, params)
        s = sidra_to_series(df_sidra, cfg["date_col"], cfg["value_col"], freq=cfg.get("freq","M"))
        tf = cfg.get("transform")
        if tf == "YoY":
            s = (s / s.shift(4) - 1.0) * 100.0   # trimestral
        elif tf == "YoY_M":
            s = (s / s.shift(12) - 1.0) * 100.0  # mensal
        elif tf == "MA12":
            s = s.rolling(12).mean()
        series[nice] = clip_last_10y_and_last_month(to_monthly_eom(s, how="last"))

    # --------- SGS EXTRA (se IDs mapeados) ---------
    for nice, sid in SGS_EXTRA.items():
        s = fetch_sgs_python_bcb_chunked(sid, START_10Y, LAST_CLOSED_MONTH_END, years_per_chunk=5)
        series[nice] = clip_last_10y_and_last_month(to_monthly_eom(s, how="last"))

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
        "PIB EUA (% a/a)",   # novo

        # --- Crédito (PF/PJ) ---
        "Saldo crédito total (% PIB)",
        "Saldo crédito PF (% PIB)",
        "Saldo crédito PJ (% PIB)",
        "Saldo crédito PJ dir. (% PIB)",
        "Saldo crédito outros setores (% PIB)",
        "Inadimplência total (>=90d, %)",
        "Inadimplência PF total (>=90d, %)",
        "Inadimplência PF livres (>=90d, %)",
        "Inadimplência PJ total (>=90d, %)",
        "Inadimplência PJ livres (>=90d, %)",
        "Endividamento famílias (% renda, 12m)",
        "Comprometimento renda famílias (% renda, média mov. trimestral)",

        # (quando mapear, acrescente aqui)
        # "PIB Brasil (% a/a)", "Desemprego 12m (PNAD)",
        # "PIM-PF (%12m)", "PMC ampliado (%12m)", "PMS (%12m)",
        # "Confiança empresarial (FGV) - média do ano",
        # "Confiança consumidor (FGV) - média do ano",
        # "Resultado primário (R$ Bilhões)", "Resultado primário (% PIB)",
        # "DBGG (% PIB)", "Saldo balança comercial (US$)",
        # "IPCA (% acumulado no ano)", "IPCA livre (% acumulado no ano)",
        # "IGP-M (% acumulado no ano)", "Crédito doméstico (R$ Bilhões)",
    ]

    cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    df = df[cols]

    # --------- Excel ---------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        # Aba mensal
        df.to_excel(writer, sheet_name="Indicadores (mensal)")

        # Metadados
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
                "Yahoo Finance (^VIX)"     if c.startswith("VIX") else
                "Yahoo Finance (BZ=F)"     if c.startswith("Brent") else
                "Yahoo Finance (DBC ETF)"  if c.startswith("Commodities") else
                "FRED API (DFEDTARU/DFEDTARL -> média)" if c.startswith("Fed Funds") else
                "FRED API (DGS10) ou Yahoo (^TNX/10)"   if c.startswith("Treasury 10 anos") else
                "FRED (A191RL1Q225SBEA)"                if c.startswith("PIB EUA") else
                "BCB/SGS (20622/20624/20623/20629/22070)"  if c.startswith("Saldo crédito") else
                "BCB/SGS (21082/21083/21086/21084/21112)"  if c.startswith("Inadimplência") else
                "BCB/SGS (29038/29265)" if c.startswith("Endividamento") or c.startswith("Comprometimento") else
                "Ipeadata (OData v4)"    if c in IPEA_MAP.keys() else
                "SIDRA/IBGE"             if c in SIDRA_MAP.keys() else
                "BCB/SGS (ids extras)"   if c in SGS_EXTRA.keys() else
                "—"
                for c in cols
            ],
            "Frequência": ["Mensal (fim do mês)"] * len(cols),
            "Observação": [
                "Índice dessazonalizado"              if c.startswith("IBC-Br") else
                "Meta do Copom (% a.a.)"              if c.startswith("Taxa Selic") else
                "PTAX venda (R$/US$). Pode ser proxy Yahoo se OData falhar." if c.startswith("BRL/USD (PTAX") else
                "Série mensal; % do PIB (12m)"        if c.startswith("Saldo crédito") else
                "Parcela >=90 dias em atraso"         if c.startswith("Inadimplência") else
                "% da renda (conceitos BCB)"          if c.startswith("Endividamento") or c.startswith("Comprometimento") else
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

        # ----- Médias anuais -----
        df_year = df.copy()
        if not df_year.empty:
            df_year["ano"] = df_year.index.year
            annual = df_year.groupby("ano").mean(numeric_only=True)
            annual.to_excel(writer, sheet_name="Medias anuais")

    print(f"OK! Excel salvo em: {OUTPUT_XLSX}")

# -------------------- Run --------------------
if __name__ == "__main__":
    main()

########## Analises ###########

# 1) VAR (previsão multi-variada + IRFs)
import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, sheet_name="Indicadores (mensal)", index_col=0, parse_dates=True)

def ld(x): return np.log(x).diff()
def d(x):  return x.diff()

panel = pd.DataFrame({
    "g_ibc":      ld(df.get("IBC-Br dessaz. (índice 2002=100)")),
    "d_usdbrl":   ld(df.get("USD/BRL (mercado, Yahoo)")),
    "d_brent":    ld(df.get("Brent (US$/bbl)")),
    "ret_spx":    ld(df.get("S&P 500 (nível)")),
    "selic":      df.get("Taxa Selic (% a.a.) [meta]").pct_change(1),
    "ust10":      df.get("Treasury 10 anos (% a.a.)").pct_change(1),
}).dropna()

panel = panel.loc[panel.index >= (panel.index.max() - pd.DateOffset(years=10))]

model = VAR(panel)
sel = model.select_order(maxlags=12)
p = sel.aic or 6
res = model.fit(p)
print(res.summary())

fcst = res.forecast(panel.values[-p:], steps=12)
fcst_idx = pd.date_range(panel.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
fcst_df = pd.DataFrame(fcst, index=fcst_idx, columns=panel.columns)
print("\nForecast 12m:\n", fcst_df.head())

irf = res.irf(12)
irf.plot(orth=False)
plt.show()

# 2) VECM (cointegração)
import pandas as pd, numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, sheet_name="Indicadores (mensal)", index_col=0, parse_dates=True)

col_usd = "USD/BRL (mercado, Yahoo)"
col_selic = "Taxa Selic (% a.a.) [meta]"
col_fed = "Fed Funds (média da banda, % a.a.)"
col_ust10 = "Treasury 10 anos (% a.a.)"

have = set(df.columns)
missing = [c for c in [col_usd, col_selic, col_fed] if c not in have]
if missing:
    print("[Aviso] Faltam no Excel:", missing)

if col_fed not in have:
    if col_ust10 in have:
        print("[Info] Usando UST 10y como proxy do Fed Funds para construir o spread.")
        df[col_fed] = df[col_ust10]
    else:
        raise RuntimeError("Nem 'Fed Funds' nem 'Treasury 10 anos' estão no Excel.")

want = [col_usd, col_selic, col_fed]
avail = [c for c in want if c in df.columns]
panel = df[avail].copy().rename(columns={
    col_usd: "usdbrl", col_selic: "selic", col_fed: "fed"
})

need = {"usdbrl", "selic", "fed"}
if not need.issubset(set(panel.columns)):
    raise RuntimeError(f"Colunas insuficientes para VECM. Tenho {list(panel.columns)}, preciso de {list(need)}.")

panel["spread"] = panel["selic"] - panel["fed"]
Z = panel[["usdbrl", "spread"]].dropna()
if len(Z) == 0:
    raise RuntimeError("Sem dados após o dropna.")
Z = Z.loc[Z.index >= (Z.index.max() - pd.DateOffset(years=10))]
if len(Z) < 36:
    raise RuntimeError(f"Amostra muito curta para VECM/Johansen: {len(Z)} obs (<36).")

jres = coint_johansen(Z, det_order=0, k_ar_diff=2)
print("Eigenvalues:", jres.eig)
print("Trace stats:", jres.lr1)
print("Trace crit (90/95/99%):\n", jres.cvt)

has_coint = jres.lr1[0] > jres.cvt[0,1]
print(f"Cointegração detectada (95%)? {has_coint}")

if has_coint:
    vecm = VECM(Z, k_ar_diff=2, coint_rank=1, deterministic="co")
    res = vecm.fit()
    print(res.summary())
    fc = res.predict(steps=12)
    print("\nPrevisão 12m (primeiras linhas):\n", fc.head())
else:
    print("Sem cointegração significativa a 95%. Sugestão: usar VAR em diferenças.")

# 3) Fator Dinâmico (PCA) + ARX p/ Ibovespa
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

panel = panel.loc[panel.index >= (panel.index.max() - pd.DateOffset(years=10))]
X = panel.drop(columns=["ret_ibov"])
Xz = (X - X.mean()) / X.std(ddof=0)

pca = PCA(n_components=2).fit(Xz)
factors = pd.DataFrame(pca.transform(Xz), index=X.index, columns=["F1","F2"])

Y = panel["ret_ibov"].copy()
lags = pd.concat(
    [Y.shift(1).rename("ret_ibov_lag1"),
     factors["F1"].shift(1).rename("F1_lag1"),
     factors["F2"].shift(1).rename("F2_lag1")],
    axis=1
).dropna()

Y_ = Y.loc[lags.index]
X_ = sm.add_constant(lags, has_constant="add")
res = sm.OLS(Y_, X_).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(res.summary())

x_last = lags.iloc[[-1]]
X_last = sm.add_constant(x_last, has_constant="add")
signal = float(res.predict(X_last))
print("Previsão de retorno (próximo mês):", signal)

# 4) Markov-Switching no Ibovespa
import pandas as pd, numpy as np
import statsmodels.api as sm

XLS = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\IndicadoresMacro\indicadores_macro.xlsx"
df = pd.read_excel(XLS, "Indicadores (mensal)", index_col=0, parse_dates=True)
ret_ibov = np.log(df.get("Ibovespa (nível)")).diff().dropna()
ret_ibov = ret_ibov.loc[ret_ibov.index >= (ret_ibov.index.max() - pd.DateOffset(years=10))]

mod = sm.tsa.MarkovRegression(ret_ibov, k_regimes=2, trend="c", switching_variance=True)
res = mod.fit()
print(res.summary())
probs = res.smoothed_marginal_probabilities[0]
print(probs.tail())
# Exemplo de regra: reduzir exposição quando prob_regime_ruim > 0.7
