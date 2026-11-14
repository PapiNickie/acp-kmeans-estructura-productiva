"""Descarga los indicadores del Banco Mundial (WDI) que usa el proyecto.

Uso:
    python scripts/descargar_datos.py

Genera dos archivos en data/:
    wdi_panel.csv     panel largo país-indicador-año (2015-2024), tal como lo entrega la API
    paises_meta.csv   nombre, región y grupo de ingreso de cada economía

La API pública del Banco Mundial no pide registro ni clave. Si falla una
consulta, el script la reintenta hasta cinco veces.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
API = "https://api.worldbank.org/v2"
CABECERAS = {"User-Agent": "Mozilla/5.0 (compatible; taller-acp-kmeans/1.0)"}
ANIOS = "2015:2024"

# Código WDI -> nombre corto que usa el proyecto
INDICADORES = {
    "NV.AGR.TOTL.ZS": "va_agro",          # Agricultura, silvicultura y pesca, valor agregado (% del PIB)
    "NV.IND.MANF.ZS": "va_manuf",         # Manufactura, valor agregado (% del PIB)
    "NV.IND.TOTL.ZS": "va_industria",     # Industria incluida construcción, valor agregado (% del PIB)
    "NV.SRV.TOTL.ZS": "va_servicios",     # Servicios, valor agregado (% del PIB)
    "NY.GDP.TOTL.RT.ZS": "rentas_rn",     # Rentas totales de recursos naturales (% del PIB)
    "SL.AGR.EMPL.ZS": "empleo_agro",      # Empleo en agricultura (% del empleo total, estimación OIT)
    "SL.IND.EMPL.ZS": "empleo_industria", # Empleo en industria (% del empleo total, estimación OIT)
    "SP.URB.TOTL.IN.ZS": "urbana",        # Población urbana (% del total)
    "NE.EXP.GNFS.ZS": "export_pib",       # Exportaciones de bienes y servicios (% del PIB)
    "NE.GDI.TOTL.ZS": "inversion",        # Formación bruta de capital (% del PIB)
    "TX.VAL.TECH.MF.ZS": "alta_tec",      # Exportaciones de alta tecnología (% de las exportaciones de manufacturas)
    "TX.VAL.FUEL.ZS.UN": "x_combust",     # Exportaciones de combustibles (% de las exportaciones de mercancías)
    "TX.VAL.MMTL.ZS.UN": "x_minerales",   # Exportaciones de minerales y metales (% de las exportaciones de mercancías)
    "TX.VAL.MANF.ZS.UN": "x_manuf",       # Exportaciones de manufacturas (% de las exportaciones de mercancías)
    "TX.VAL.FOOD.ZS.UN": "x_alimentos",   # Exportaciones de alimentos (% de las exportaciones de mercancías)
    "BX.TRF.PWKR.DT.GD.ZS": "remesas",    # Remesas personales recibidas (% del PIB)
    "ST.INT.RCPT.XP.ZS": "turismo",       # Ingresos por turismo internacional (% de las exportaciones totales)
    "NY.GDP.PCAP.PP.KD": "pib_pc_ppa",    # PIB per cápita, PPA (dólares internacionales constantes de 2021)
    "SP.POP.TOTL": "poblacion",           # Población total
}


def pedir(url: str, intentos: int = 5) -> list:
    for i in range(intentos):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=120)
            r.raise_for_status()
            datos = r.json()
            if isinstance(datos, list) and len(datos) == 2:
                return datos
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}")


# Si la API del Banco Mundial no responde, estas dos series se toman de Our World in Data, que
# publica la misma serie del Banco Mundial (PIB per cápita) o una equivalente (población, ONU).
ESPEJO_OWID = {
    "NY.GDP.PCAP.PP.KD": ("https://ourworldindata.org/grapher/gdp-per-capita-worldbank.csv"
                          "?v=1&csvType=full&useColumnShortNames=true", "ny_gdp_pcap_pp_kd"),
    "SP.POP.TOTL": ("https://ourworldindata.org/grapher/population.csv"
                    "?v=1&csvType=full&useColumnShortNames=true", "population"),
}


def desde_owid(codigo: str, corto: str, validos: set) -> pd.DataFrame:
    import io
    url, col = ESPEJO_OWID[codigo]
    r = requests.get(url, headers=CABECERAS, timeout=120)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    a0, a1 = (int(x) for x in ANIOS.split(":"))
    d = d[d["code"].isin(validos) & d["year"].between(a0, a1)].dropna(subset=[col])
    return pd.DataFrame({"iso3": d["code"], "indicador": corto, "codigo_wdi": codigo,
                         "anio": d["year"].astype(int), "valor": d[col].astype(float)})


INGRESO_ES = {"High income": "Ingreso alto", "Upper middle income": "Ingreso mediano alto",
              "Lower middle income": "Ingreso mediano bajo", "Low income": "Ingreso bajo", "Not classified": "Sin clasificar"}


def descargar_meta() -> pd.DataFrame:
    """Nombre, región y grupo de ingreso de cada economía, en inglés y en español."""
    _, en = pedir(f"{API}/country?format=json&per_page=400")
    _, es = pedir(f"{API}/es/country?format=json&per_page=400")
    es = {p["id"]: p for p in es}
    meta = pd.DataFrame(
        {
            "iso3": p["id"],
            "pais": es.get(p["id"], p)["name"],
            "pais_en": p["name"],
            "region": es.get(p["id"], p)["region"]["value"],
            "ingreso": INGRESO_ES.get(p["incomeLevel"]["value"], p["incomeLevel"]["value"]),
            "region_en": p["region"]["value"],
            "ingreso_en": p["incomeLevel"]["value"],
        }
        for p in en
        if p["region"]["value"] != "Aggregates"
    )
    for col in meta.columns:  # la API en español deja espacios al final de algunos nombres
        meta[col] = meta[col].str.strip()
    meta.to_csv(DATA / "paises_meta.csv", index=False)
    print(f"Economías (sin agregados regionales): {len(meta)}")
    return meta


def main() -> None:
    DATA.mkdir(exist_ok=True)
    cache = DATA / "_cache"   # un archivo por indicador: si la descarga se corta, no se repite lo ya bajado
    cache.mkdir(exist_ok=True)
    meta = descargar_meta()
    validos = set(meta.iso3)

    partes = []
    for codigo, corto in INDICADORES.items():
        archivo = cache / f"{corto}.csv"
        if archivo.exists():
            partes.append(pd.read_csv(archivo))
            print(f"  {codigo:22s} {corto:16s} (ya descargado)")
            continue
        try:
            cab, obs = pedir(f"{API}/country/all/indicator/{codigo}?format=json&date={ANIOS}&per_page=20000")
        except RuntimeError:
            if codigo not in ESPEJO_OWID:
                raise
            parte = desde_owid(codigo, corto, validos)
            parte.to_csv(archivo, index=False)
            partes.append(parte)
            print(f"  {codigo:22s} {corto:16s} (API sin respuesta: copia de Our World in Data)", flush=True)
            continue
        filas = [
            {"iso3": o["countryiso3code"], "indicador": corto, "codigo_wdi": codigo,
             "anio": int(o["date"]), "valor": float(o["value"])}
            for o in obs or []
            if o["value"] is not None and o["countryiso3code"] in validos
        ]
        parte = pd.DataFrame(filas)
        parte.to_csv(archivo, index=False)
        partes.append(parte)
        print(f"  {codigo:22s} {corto:16s} actualizado {cab.get('lastupdated')}", flush=True)
        time.sleep(2)  # pausa corta: el cortafuegos de la API bloquea ráfagas de consultas

    panel = pd.concat(partes, ignore_index=True).sort_values(["indicador", "iso3", "anio"])
    panel.to_csv(DATA / "wdi_panel.csv", index=False)
    print(f"Panel guardado: {len(panel):,} observaciones en {DATA / 'wdi_panel.csv'}")


if __name__ == "__main__":
    main()
