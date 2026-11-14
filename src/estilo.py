"""Identidad visual y piezas de interfaz que comparten las páginas."""
from __future__ import annotations

import streamlit as st

from . import analisis as an
from . import estado

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&display=swap');
h1, h2, h3, h4, .encab h1 {font-family: 'Source Serif 4', Georgia, serif !important; font-weight: 700 !important;}
.block-container {padding-top: 4rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.encab {border-bottom: 1px solid #DDD5C8; padding-bottom: .9rem; margin-bottom: 1.4rem;}
.encab .paso {font: 600 .78rem/1 'IBM Plex Sans', sans-serif; letter-spacing: .14em; text-transform: uppercase; color: #C4502E;}
.encab h1 {margin: .35rem 0 .35rem 0; font-size: 2.35rem; line-height: 1.12;}
.encab p {color: #4A555C; font-size: 1.06rem; margin: 0; max-width: 820px;}
.nota {background: #F1EBE1; border-left: 4px solid #C4502E; border-radius: 6px; padding: .85rem 1.1rem; margin: .4rem 0 1.1rem 0;
       color: #1F2A30; font-size: .97rem; line-height: 1.5;}
.nota.azul {border-left-color: #2F6DB5;}
.nota b {font-weight: 600;}
.lectura {font-size: 1rem; line-height: 1.6; color: #1F2A30; max-width: 860px;}
.cifra {background: #FFFFFF; border: 1px solid #E6E0D6; border-radius: 8px; padding: .9rem 1rem; height: 100%;}
.cifra .v {font: 700 clamp(1.35rem, 2.2vw, 1.9rem)/1.05 'Source Serif 4', Georgia, serif; color: #1F2A30; white-space: nowrap;}
.cifra .l {font-size: .86rem; color: #4A555C; margin-top: .35rem; line-height: 1.35;}
.grupo-tarjeta {background:#FFFFFF; border:1px solid #E6E0D6; border-top: 5px solid var(--c); border-radius: 8px; padding: .9rem 1rem; height: 100%;}
.grupo-tarjeta h4 {margin: 0 0 .3rem 0; font-size: 1.12rem;}
.grupo-tarjeta .n {color:#4A555C; font-size:.86rem;}
.grupo-tarjeta p {font-size: .92rem; line-height: 1.45; margin: .5rem 0 0 0;}
.pie-fuente {color:#8A9298; font-size:.82rem; margin-top: -.4rem;}
section[data-testid="stSidebar"] .resumen {font-size: .86rem; line-height: 1.55; color: #1F2A30;}
section[data-testid="stSidebar"] .resumen span {color:#C4502E; font-weight:600;}
</style>
"""


def _nb(t: str) -> str:
    """Espacio duro antes del signo de porcentaje para que no quede solo al inicio de una línea."""
    return t.replace(" %", "\u00a0%")


def fmt(d: int):
    """Formateador de tablas con coma decimal."""
    import math
    return lambda x: "" if x is None or (isinstance(x, float) and math.isnan(x)) else num(float(x), d)


def aplicar_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(paso: str, titulo: str, texto: str) -> None:
    st.markdown(_nb(f'<div class="encab"><div class="paso">{paso}</div><h1>{titulo}</h1><p>{texto}</p></div>'),
                unsafe_allow_html=True)


def nota(texto: str, tono: str = "") -> None:
    st.markdown(_nb(f'<div class="nota {tono}">{texto}</div>'), unsafe_allow_html=True)


def lectura(texto: str) -> None:
    st.markdown(_nb(f'<div class="lectura">{texto}</div>'), unsafe_allow_html=True)


def cifras(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (v, l) in zip(cols, items):
        col.markdown(_nb(f'<div class="cifra"><div class="v">{v}</div><div class="l">{l}</div></div>'), unsafe_allow_html=True)


def fuente(texto: str = "Fuente: Banco Mundial, Indicadores del Desarrollo Mundial (WDI). Cálculos propios.") -> None:
    st.markdown(f'<div class="pie-fuente">{texto}</div>', unsafe_allow_html=True)


def num(x: float, d: int = 1) -> str:
    """Formato colombiano: coma decimal, punto de miles."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, d: int = 1) -> str:
    return num(x * 100, d) + "\u00a0%"


def barra_lateral() -> None:
    """Decisiones sobre los datos y resumen del estado del análisis."""
    estado.iniciar()
    with st.sidebar:
        st.markdown("### Decisiones sobre los datos")
        estado.control(st.slider, "ventana", "Ventana de años (promedio por país)", 2015, 2024)
        estado.control(st.select_slider, "pob_min", "Población mínima (millones)", options=[0.0, 0.5, 1.0, 2.0, 5.0])
        opciones = list(an.VARIABLES)
        with st.expander(f"Variables del modelo ({len(st.session_state['variables'])})"):
            estado.control(st.multiselect, "variables", "Variables que entran al ACP", options=opciones,
                           format_func=an.etiqueta, help="Mínimo cuatro. Por defecto, las once del estudio.")
        with st.expander(f"Variables con logaritmo ({len([v for v in st.session_state['log'] if v in st.session_state['variables']])})"):
            estado.control(st.multiselect, "log", "Se transforman con log(1 + x)", options=opciones, format_func=an.etiqueta,
                           help="Para variables muy asimétricas; ver la página de preparación.")
        if len(st.session_state["variables"]) < 4:
            st.warning("Elige al menos cuatro variables.")
        st.button("Volver a la configuración del estudio", on_click=estado.restablecer, use_container_width=True)

        c = estado.config()
        if len(c["variables"]) >= 4:
            try:
                prep, res, km = estado.todo(c)
                n = min(c["n_comp"], len(c["variables"]))
                st.markdown("---")
                st.markdown(
                    '<div class="resumen">'
                    f'Economías en el modelo: <span>{len(prep.base)}</span><br>'
                    f'Variables: <span>{len(c["variables"])}</span><br>'
                    f'Componentes retenidos: <span>{n}</span> ({pct(res.acumulada[n - 1])} de la varianza)<br>'
                    f'Grupos (K): <span>{c["k"]}</span>, silueta <span>{num(km.silueta_media, 3)}</span>'
                    '</div>', unsafe_allow_html=True)
            except Exception as e:  # la app no debe caerse por una combinación rara de opciones
                st.error(f"No fue posible calcular con esta configuración: {e}")
        st.markdown("---")
        st.caption("Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali")


def requiere_variables() -> bool:
    if len(st.session_state.get("variables", [])) < 4:
        st.info("Elige al menos cuatro variables en la barra lateral para continuar.")
        return False
    return True
