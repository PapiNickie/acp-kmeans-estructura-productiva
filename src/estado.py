"""Configuración compartida entre páginas y cálculos con caché.

La aplicación recalcula todo a partir del panel de datos cada vez que cambia una decisión
(variables, ventana de años, filtro de tamaño, número de componentes o de grupos). Las funciones
con caché evitan repetir cálculos idénticos.
"""
from __future__ import annotations

import streamlit as st

from . import analisis as an

# Decisiones del estudio (justificadas en la página de preparación y en el README)
VARIABLES_ESTUDIO = [
    "va_agro", "va_manuf", "va_servicios", "rentas_rn",
    "empleo_agro", "empleo_industria", "urbana",
    "export_pib", "x_manuf", "x_extractivas", "alta_tec",
]
LOG_ESTUDIO = ["rentas_rn", "export_pib", "alta_tec"]
N_COMP_ESTUDIO = 3
K_ESTUDIO = 4
VENTANA_ESTUDIO = (2019, 2023)
POB_ESTUDIO = 1.0

DEFECTOS = {
    "ventana": VENTANA_ESTUDIO,
    "pob_min": POB_ESTUDIO,
    "variables": VARIABLES_ESTUDIO,
    "log": LOG_ESTUDIO,
    "n_comp": N_COMP_ESTUDIO,
    "k": K_ESTUDIO,
    "pais_foco": "COL",
}


def iniciar() -> None:
    for c, v in DEFECTOS.items():
        st.session_state.setdefault(c, list(v) if isinstance(v, list) else v)


def restablecer() -> None:
    for c, v in DEFECTOS.items():
        st.session_state[c] = list(v) if isinstance(v, list) else v
        st.session_state.pop("_" + c, None)


def _copiar(clave: str) -> None:
    st.session_state[clave] = st.session_state["_" + clave]


def control(widget, clave: str, *args, **kwargs):
    """Crea un widget cuyo valor sobrevive al cambio de página."""
    st.session_state["_" + clave] = st.session_state[clave]
    return widget(*args, key="_" + clave, on_change=_copiar, args=(clave,), **kwargs)


def config() -> dict:
    s = st.session_state
    return {"ventana": tuple(s["ventana"]), "pob_min": float(s["pob_min"]), "variables": tuple(s["variables"]),
            "log": tuple(v for v in s["log"] if v in s["variables"]), "n_comp": int(s["n_comp"]), "k": int(s["k"])}


# ------------------------------------------------------------------ cálculos con caché
@st.cache_data(show_spinner=False)
def datos():
    return an.cargar_datos()


@st.cache_data(show_spinner=False)
def corte(ventana: tuple[int, int]):
    panel, meta = datos()
    return an.corte_transversal(panel, meta, *ventana)


@st.cache_data(show_spinner="Preparando la matriz de datos...")
def preparacion(ventana, pob_min, variables, log):
    return an.preparar(corte(ventana), list(variables), list(log), pob_min)


@st.cache_data(show_spinner="Calculando componentes principales...")
def acp(ventana, pob_min, variables, log):
    return an.acp(preparacion(ventana, pob_min, variables, log))


@st.cache_data(show_spinner="Comparando valores de K...")
def tabla_k(ventana, pob_min, variables, log, n_comp, k_max=10):
    r = acp(ventana, pob_min, variables, log)
    return an.evaluar_k(r.scores.iloc[:, :n_comp].values, 2, k_max)


@st.cache_data(show_spinner="Ejecutando K-Means...")
def kmeans(ventana, pob_min, variables, log, n_comp, k):
    r = acp(ventana, pob_min, variables, log)
    return an.segmentar(r.scores.iloc[:, :n_comp].values, k)


def todo(c: dict | None = None):
    """Devuelve preparación, ACP y K-Means para la configuración actual."""
    c = c or config()
    a = (c["ventana"], c["pob_min"], c["variables"], c["log"])
    prep = preparacion(*a)
    res = acp(*a)
    n = min(c["n_comp"], len(c["variables"]))
    km = kmeans(*a, n, c["k"])
    return prep, res, km
