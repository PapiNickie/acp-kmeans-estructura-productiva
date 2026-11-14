"""Procesamiento y algoritmos del proyecto.

Todo lo que la aplicación muestra sale de estas funciones: el corte transversal a partir del panel
del Banco Mundial, la preparación de la matriz, el ACP, la comparación de valores de K y la
segmentación con K-Means. El script scripts/reproducir_resultados.py usa las mismas funciones para
escribir las tablas de la carpeta resultados/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
SEMILLA = 42

# --------------------------------------------------------------------------------------------
# Catálogo de variables
# --------------------------------------------------------------------------------------------
# bloque: qué pregunta responde la variable dentro del estudio
VARIABLES: dict[str, dict] = {
    "va_agro": {"etiqueta": "Agricultura (% PIB)", "larga": "Agricultura, silvicultura y pesca, valor agregado (% del PIB)",
                "codigo": "NV.AGR.TOTL.ZS", "bloque": "Qué produce"},
    "va_manuf": {"etiqueta": "Manufactura (% PIB)", "larga": "Manufactura, valor agregado (% del PIB)",
                 "codigo": "NV.IND.MANF.ZS", "bloque": "Qué produce"},
    "va_servicios": {"etiqueta": "Servicios (% PIB)", "larga": "Servicios, valor agregado (% del PIB)",
                     "codigo": "NV.SRV.TOTL.ZS", "bloque": "Qué produce"},
    "va_industria": {"etiqueta": "Industria con construcción (% PIB)", "larga": "Industria, incluida la construcción, valor agregado (% del PIB)",
                     "codigo": "NV.IND.TOTL.ZS", "bloque": "Qué produce"},
    "rentas_rn": {"etiqueta": "Rentas de recursos naturales (% PIB)", "larga": "Rentas totales de los recursos naturales (% del PIB)",
                  "codigo": "NY.GDP.TOTL.RT.ZS", "bloque": "Qué produce"},
    "empleo_agro": {"etiqueta": "Empleo agrícola (% empleo)", "larga": "Empleo en agricultura (% del empleo total, estimación modelada de la OIT)",
                    "codigo": "SL.AGR.EMPL.ZS", "bloque": "Dónde trabaja la gente"},
    "empleo_industria": {"etiqueta": "Empleo industrial (% empleo)", "larga": "Empleo en industria (% del empleo total, estimación modelada de la OIT)",
                         "codigo": "SL.IND.EMPL.ZS", "bloque": "Dónde trabaja la gente"},
    "urbana": {"etiqueta": "Población urbana (%)", "larga": "Población urbana (% de la población total)",
               "codigo": "SP.URB.TOTL.IN.ZS", "bloque": "Dónde trabaja la gente"},
    "export_pib": {"etiqueta": "Exportaciones (% PIB)", "larga": "Exportaciones de bienes y servicios (% del PIB)",
                   "codigo": "NE.EXP.GNFS.ZS", "bloque": "Cómo se inserta en el mundo"},
    "inversion": {"etiqueta": "Formación bruta de capital (% PIB)", "larga": "Formación bruta de capital (% del PIB)",
                  "codigo": "NE.GDI.TOTL.ZS", "bloque": "Cómo se inserta en el mundo"},
    "alta_tec": {"etiqueta": "Alta tecnología (% export. manuf.)", "larga": "Exportaciones de productos de alta tecnología (% de las exportaciones de manufacturas)",
                 "codigo": "TX.VAL.TECH.MF.ZS", "bloque": "Qué vende al mundo"},
    "x_manuf": {"etiqueta": "Manufacturas (% export. mercancías)", "larga": "Exportaciones de manufacturas (% de las exportaciones de mercancías)",
                "codigo": "TX.VAL.MANF.ZS.UN", "bloque": "Qué vende al mundo"},
    "x_extractivas": {"etiqueta": "Combustibles y minerales (% export. mercancías)",
                      "larga": "Exportaciones de combustibles más minerales y metales (% de las exportaciones de mercancías)",
                      "codigo": "TX.VAL.FUEL.ZS.UN + TX.VAL.MMTL.ZS.UN", "bloque": "Qué vende al mundo"},
    "x_alimentos": {"etiqueta": "Alimentos (% export. mercancías)", "larga": "Exportaciones de alimentos (% de las exportaciones de mercancías)",
                    "codigo": "TX.VAL.FOOD.ZS.UN", "bloque": "Qué vende al mundo"},
    "remesas": {"etiqueta": "Remesas recibidas (% PIB)", "larga": "Remesas personales recibidas (% del PIB)",
                "codigo": "BX.TRF.PWKR.DT.GD.ZS", "bloque": "Cómo se inserta en el mundo"},
    "turismo": {"etiqueta": "Turismo (% export. totales)", "larga": "Ingresos por turismo internacional (% de las exportaciones totales)",
                "codigo": "ST.INT.RCPT.XP.ZS", "bloque": "Cómo se inserta en el mundo"},
}

# Variables que no entran al modelo: se usan después para leer los grupos
EXTERNAS = {
    "pib_pc_ppa": {"etiqueta": "PIB per cápita PPA (US$ int. 2021)", "codigo": "NY.GDP.PCAP.PP.KD"},
    "poblacion": {"etiqueta": "Población (millones)", "codigo": "SP.POP.TOTL"},
}

# Configuración por defecto (justificada en la página de preparación y en el README)
POR_DEFECTO = {
    "anio_ini": 2019,
    "anio_fin": 2023,
    "pob_min": 1.0,  # millones de habitantes
    "variables": [],  # se completa más abajo
    "log": [],
}


# Nombres oficiales largos del Banco Mundial que se acortan para los gráficos
NOMBRES_CORTOS = {
    "Hong Kong, Región Administrativa Especial": "Hong Kong (RAE)", "Corea, República de": "Corea del Sur",
    "Corea, República Popular Democrática de": "Corea del Norte", "Egipto, República Árabe de": "Egipto",
    "Irán, República Islámica del": "Irán", "Congo, República Democrática del": "Congo (RD)",
    "Congo, República del": "Congo", "Federación de Rusia": "Rusia", "República Democrática Popular Lao": "Laos",
    "Yemen, Rep. del": "Yemen", "República Árabe Siria": "Siria", "República Federal de Somalia": "Somalia",
    "Ribera Occidental y Gaza": "Cisjordania y Gaza", "República de Moldova": "Moldavia", "Viet Nam": "Vietnam",
    "Nueva Zelandia": "Nueva Zelanda", "Belarús": "Bielorrusia", "Tanzanía": "Tanzania", "Kenya": "Kenia",
    "República Eslovaca": "Eslovaquia", "Côte d'Ivoire": "Costa de Marfil", "Puerto Rico (US)": "Puerto Rico",
}


# Etiquetas cortas para el biplot
CORTAS = {'va_agro': 'Agricultura', 'va_manuf': 'Manufactura', 'va_servicios': 'Servicios', 'va_industria': 'Industria', 'rentas_rn': 'Rentas de recursos', 'empleo_agro': 'Empleo agrícola', 'empleo_industria': 'Empleo industrial', 'urbana': 'Población urbana', 'export_pib': 'Exportaciones/PIB', 'inversion': 'Inversión', 'alta_tec': 'Alta tecnología', 'x_manuf': 'Export. de manufacturas', 'x_extractivas': 'Export. extractivas', 'x_alimentos': 'Export. de alimentos', 'remesas': 'Remesas', 'turismo': 'Turismo'}


def corta(v: str) -> str:
    return CORTAS.get(v, etiqueta(v))


def minus(t: str) -> str:
    """Primera letra en minúscula, sin tocar siglas como PIB."""
    return t[:1].lower() + t[1:] if t else t


def etiqueta(v: str) -> str:
    if v in VARIABLES:
        return VARIABLES[v]["etiqueta"]
    if v in EXTERNAS:
        return EXTERNAS[v]["etiqueta"]
    return v


# --------------------------------------------------------------------------------------------
# Carga y corte transversal
# --------------------------------------------------------------------------------------------
def cargar_datos(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(data_dir / "wdi_panel.csv")
    meta = pd.read_csv(data_dir / "paises_meta.csv")
    meta = meta.apply(lambda col: col.str.strip() if col.dtype == object or str(col.dtype).startswith("str") else col)
    return panel, meta


def corte_transversal(panel: pd.DataFrame, meta: pd.DataFrame, anio_ini: int, anio_fin: int) -> pd.DataFrame:
    """Promedio de cada indicador en la ventana [anio_ini, anio_fin] para cada economía.

    Promediar varios años suaviza choques de un solo año (la pandemia de 2020, un precio del
    petróleo atípico) y reduce los faltantes: basta con que el país reporte un año de la ventana.
    """
    v = panel[panel.anio.between(anio_ini, anio_fin)]
    ancho = v.pivot_table(index="iso3", columns="indicador", values="valor", aggfunc="mean")
    if {"x_combust", "x_minerales"}.issubset(ancho.columns):
        ancho["x_extractivas"] = ancho["x_combust"] + ancho["x_minerales"]
    if "poblacion" in ancho.columns:
        ancho["poblacion"] = ancho["poblacion"] / 1e6
    df = meta.merge(ancho.reset_index(), on="iso3", how="left")
    df["pais"] = df["pais"].replace(NOMBRES_CORTOS)
    return df


@dataclass
class Preparacion:
    base: pd.DataFrame            # economías que entran al modelo, variables en su escala original
    X: pd.DataFrame               # variables del modelo después de la transformación logarítmica
    Z: np.ndarray                 # X estandarizada (media 0, desviación 1)
    variables: list[str]
    log: list[str]
    excluidas_pob: pd.DataFrame   # economías fuera por tamaño
    excluidas_na: pd.DataFrame    # economías fuera por datos faltantes
    faltantes: pd.Series          # faltantes por variable antes de eliminar filas
    duplicados: int
    escalador: StandardScaler = field(repr=False, default=None)


def preparar(df: pd.DataFrame, variables: list[str], log: list[str], pob_min: float) -> Preparacion:
    """Filtra, revisa calidad, transforma y estandariza."""
    duplicados = int(df.duplicated(subset="iso3").sum())
    df = df.drop_duplicates(subset="iso3")
    grandes = df["poblacion"] >= pob_min
    excl_pob = df.loc[~grandes, ["iso3", "pais", "region", "poblacion"]].sort_values("poblacion", ascending=False)
    d = df.loc[grandes].copy()
    faltantes = d[variables].isna().sum().sort_values(ascending=False)
    completas = d[variables].notna().all(axis=1)
    excl_na = d.loc[~completas, ["iso3", "pais", "region"]].copy()
    excl_na["variables_faltantes"] = d.loc[~completas, variables].isna().apply(
        lambda r: ", ".join(etiqueta(c) for c in variables if r[c]), axis=1)
    base = d.loc[completas].reset_index(drop=True)
    X = base[variables].copy()
    for v in log:
        if v in X.columns:
            X[v] = np.log1p(X[v].clip(lower=0))
    esc = StandardScaler()
    Z = esc.fit_transform(X)
    return Preparacion(base, X, Z, variables, [v for v in log if v in variables], excl_pob, excl_na.reset_index(drop=True),
                       faltantes, duplicados, esc)


def asimetria(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    return df[variables].skew().sort_values(ascending=False)


# --------------------------------------------------------------------------------------------
# ACP
# --------------------------------------------------------------------------------------------
@dataclass
class ResultadoACP:
    modelo: PCA
    valores_propios: np.ndarray   # suman p (número de variables): escala de la matriz de correlaciones
    varianza: np.ndarray          # proporción explicada por componente
    acumulada: np.ndarray
    cargas: pd.DataFrame          # correlación variable-componente (vector propio por raíz del valor propio)
    vectores: pd.DataFrame        # vectores propios (components_ de sklearn)
    scores: pd.DataFrame          # coordenadas de las economías en todos los componentes


def acp(prep: Preparacion) -> ResultadoACP:
    p = len(prep.variables)
    m = PCA(n_components=p, random_state=SEMILLA).fit(prep.Z)
    vectores = m.components_.copy()
    scores = m.transform(prep.Z)
    # El signo de un componente es arbitrario: se orienta para que la variable con la carga de mayor
    # magnitud quede con signo positivo. Así la lectura no cambia de una corrida a otra.
    for j in range(p):
        if vectores[j, np.argmax(np.abs(vectores[j]))] < 0:
            vectores[j] *= -1
            scores[:, j] *= -1
    razon = m.explained_variance_ratio_
    lam = razon * p
    nombres = [f"CP{j + 1}" for j in range(p)]
    cargas = pd.DataFrame(vectores.T * np.sqrt(lam), index=prep.variables, columns=nombres)
    vec = pd.DataFrame(vectores.T, index=prep.variables, columns=nombres)
    sc = pd.DataFrame(scores, columns=nombres)
    return ResultadoACP(m, lam, razon, np.cumsum(razon), cargas, vec, sc)


def sugerir_componentes(res: ResultadoACP, umbral: float = 0.70) -> dict:
    """Tres criterios habituales; la decisión final la toma quien analiza."""
    kaiser = int((res.valores_propios > 1).sum())
    acum = int(np.argmax(res.acumulada >= umbral) + 1)
    # Codo del gráfico de sedimentación: componente después de la mayor caída relativa en la
    # pendiente (segunda diferencia de los valores propios)
    lam = res.valores_propios
    if len(lam) >= 3:
        d2 = np.diff(lam, 2)
        codo = int(np.argmax(d2) + 2)
    else:
        codo = 1
    return {"kaiser": kaiser, "acumulada": acum, "codo": codo, "umbral": umbral}


# --------------------------------------------------------------------------------------------
# K-Means
# --------------------------------------------------------------------------------------------
def evaluar_k(scores: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(scores)
        filas.append({"K": k, "inercia": km.inertia_, "silueta": silhouette_score(scores, km.labels_)})
    t = pd.DataFrame(filas)
    t["caida_inercia_pct"] = -t["inercia"].pct_change() * 100
    return t


LETRAS = "ABCDEFGHIJ"


@dataclass
class ResultadoKMeans:
    etiquetas: np.ndarray         # letras A, B, C... (A = grupo más numeroso; la letra no implica orden)
    centroides: pd.DataFrame
    silueta: np.ndarray
    silueta_media: float
    inercia: float


def segmentar(scores: np.ndarray, k: int) -> ResultadoKMeans:
    km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(scores)
    # Se renombran los grupos por tamaño para que la etiqueta sea estable entre corridas.
    tam = pd.Series(km.labels_).value_counts()
    orden = sorted(tam.index, key=lambda g: (-tam[g], km.cluster_centers_[g, 0]))
    mapa = {g: LETRAS[i] for i, g in enumerate(orden)}
    et = np.array([mapa[g] for g in km.labels_])
    cent = pd.DataFrame(km.cluster_centers_[orden], index=[mapa[g] for g in orden],
                        columns=[f"CP{j + 1}" for j in range(scores.shape[1])])
    sil = silhouette_samples(scores, km.labels_)
    return ResultadoKMeans(et, cent, sil, float(sil.mean()), float(km.inertia_))


def perfiles(prep: Preparacion, km: ResultadoKMeans) -> dict[str, pd.DataFrame]:
    """Describe cada grupo con las variables originales."""
    b = prep.base.copy()
    b["grupo"] = km.etiquetas
    v = prep.variables
    medias = b.groupby("grupo")[v].mean()
    medianas = b.groupby("grupo")[v].median()
    zs = pd.DataFrame(prep.Z, columns=v)
    zs["grupo"] = km.etiquetas
    z_medias = zs.groupby("grupo")[v].mean()        # medias estandarizadas: 0 = promedio de la muestra
    tam = b.groupby("grupo").size().rename("economías")
    ext = b.groupby("grupo").agg(pib_pc_mediano=("pib_pc_ppa", "median"), poblacion_total=("poblacion", "sum"))
    general = pd.DataFrame({"media": b[v].mean(), "mediana": b[v].median()}).T
    return {"medias": medias, "medianas": medianas, "z": z_medias, "tam": tam, "externas": ext, "general": general}


def rasgos(z_medias: pd.DataFrame, n: int = 2, umbral: float = 0.4) -> dict[str, str]:
    """Frase automática con lo que más distingue a cada grupo del promedio de la muestra."""
    out = {}
    for g, fila in z_medias.iterrows():
        altos = [etiqueta(c) for c, val in fila.sort_values(ascending=False).items() if val > umbral][:n]
        bajos = [etiqueta(c) for c, val in fila.sort_values().items() if val < -umbral][:n]
        partes = []
        if altos:
            partes.append("por encima del promedio en " + " y ".join(minus(x) for x in altos))
        if bajos:
            partes.append("por debajo en " + " y ".join(minus(x) for x in bajos))
        out[g] = "; ".join(partes) if partes else "cerca del promedio en todas las variables"
    return out


def vecinos(scores: pd.DataFrame, base: pd.DataFrame, iso3: str, n: int = 6) -> pd.DataFrame:
    """Economías más cercanas a una dada en el espacio de componentes retenidos (distancia euclídea)."""
    idx = base.index[base.iso3 == iso3]
    if len(idx) == 0:
        return pd.DataFrame()
    x = scores.iloc[idx[0]].values
    d = np.sqrt(((scores.values - x) ** 2).sum(axis=1))
    t = base[["iso3", "pais", "region"]].copy()
    t["distancia"] = d
    return t[t.iso3 != iso3].nsmallest(n, "distancia").reset_index(drop=True)
