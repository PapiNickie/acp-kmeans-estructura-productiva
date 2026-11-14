import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analisis as an
from src import estado
from src import graficos as gr
from src.estilo import fmt, cifras, encabezado, fuente, lectura, nota, num, pct, requiere_variables

encabezado("Paso 3 · Análisis de componentes principales", "Pocas dimensiones para once variables",
           "Varianza explicada, gráfico de sedimentación, cargas e interpretación de cada componente, y la posición de "
           "las economías en el espacio que forman.")
if not requiere_variables():
    st.stop()

c = estado.config()
prep = estado.preparacion(c["ventana"], c["pob_min"], c["variables"], c["log"])
res = estado.acp(c["ventana"], c["pob_min"], c["variables"], c["log"])
V = list(c["variables"])
p = len(V)
sug = an.sugerir_componentes(res, 0.70)

with st.expander("Detalle matemático del ACP"):
    st.markdown("Sea $Z \\in \\mathbb{R}^{n \\times p}$ la matriz estandarizada (n economías, p variables). Su matriz de "
                "correlaciones es")
    st.latex(r"R = \frac{1}{n} Z^\top Z = V \Lambda V^\top, \qquad \Lambda = \mathrm{diag}(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p)")
    st.markdown("Las columnas de $V$ son los vectores propios (las direcciones de los componentes) y los $\\lambda_j$ sus "
                "valores propios. Las coordenadas de las economías en los primeros $k$ componentes son $F = Z V_k$, y la "
                "proporción de varianza que explica el componente $j$ es")
    st.latex(r"\frac{\lambda_j}{\sum_{i=1}^{p} \lambda_i} = \frac{\lambda_j}{p}")
    st.markdown("porque con variables estandarizadas la varianza total es p. La carga de la variable $x_i$ en el componente "
                "$j$ es su correlación con él: $v_{ij}\\sqrt{\\lambda_j}$. El signo de cada componente es arbitrario; la "
                "aplicación lo orienta para que la variable de mayor carga absoluta quede positiva.")

st.markdown("## Varianza explicada")
tabla = pd.DataFrame({"componente": res.cargas.columns, "valor propio": res.valores_propios,
                      "% de la varianza": res.varianza * 100, "% acumulado": res.acumulada * 100})
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(gr.sedimentacion(res.valores_propios), use_container_width=True)
with cc2:
    st.plotly_chart(gr.varianza(res.varianza, res.acumulada, 0.70, min(c["n_comp"], p)), use_container_width=True)
st.dataframe(tabla.style.format({"valor propio": fmt(3), "% de la varianza": fmt(1), "% acumulado": fmt(1)}),
             hide_index=True, use_container_width=True)

st.markdown("## Cuántos componentes conservar")
cifras([
    (str(sug["kaiser"]), "componentes con valor propio mayor que 1 (criterio de Kaiser)"),
    (str(sug["acumulada"]), "componentes para superar el 70 % de la varianza acumulada"),
    (str(sug["codo"]), "componentes antes del codo del gráfico de sedimentación"),
])
st.markdown("")
estado.control(st.slider, "n_comp", "Componentes que se conservan para K-Means", 2, max(2, min(p, 6)))
n = min(c["n_comp"], p)
lectura(
    f"Con {n} componentes se conserva el {pct(res.acumulada[n - 1])} de la variación original y se pierde el "
    f"{pct(1 - res.acumulada[n - 1])}. "
    + ("En la configuración del estudio, el criterio de Kaiser y el umbral del 70 % coinciden en tres componentes. El "
       "codo del gráfico de sedimentación aparece antes, en dos, pero el tercer componente tiene un valor propio de "
       f"{num(res.valores_propios[2], 2)}: explica más que cualquiera de las variables originales por sí sola y, como se ve "
       "abajo, tiene una lectura económica propia (el peso de la industria). El cuarto ya está por debajo de 1. Se conservan "
       "tres." if (c["variables"] == tuple(estado.VARIABLES_ESTUDIO) and n == 3 and sug["kaiser"] == 3) else
       "La decisión debe apoyarse en los tres criterios de arriba y en que cada componente conservado tenga una lectura clara.")
)

st.markdown("## Qué significa cada componente")
st.plotly_chart(gr.cargas(res.cargas, n), use_container_width=True)


def lectura_auto(comp: str, umbral: float = 0.4) -> str:
    s = res.cargas[comp]
    pos = [an.minus(an.etiqueta(v)) for v, x in s.sort_values(ascending=False).items() if x >= umbral]
    neg = [an.minus(an.etiqueta(v)) for v, x in s.sort_values().items() if x <= -umbral]
    t = f"Valores altos de {comp} corresponden a economías con más " + ", ".join(pos) if pos else f"{comp} no tiene cargas positivas grandes"
    if neg:
        t += "; valores bajos, a economías con más " + ", ".join(neg)
    return t + "."


NOMBRES = {"CP1": "peso del campo", "CP2": "especialización extractiva", "CP3": "peso de la industria"}
estudio = c["variables"] == tuple(estado.VARIABLES_ESTUDIO) and set(c["log"]) == set(estado.LOG_ESTUDIO)
sel = st.radio("Componente", list(res.cargas.columns[:n]), horizontal=True)
cc1, cc2 = st.columns([1, 1])
with cc1:
    st.plotly_chart(gr.contribuciones(res.cargas, sel), use_container_width=True)
with cc2:
    st.markdown(f"#### {sel}" + (f": {NOMBRES[sel]}" if estudio and sel in NOMBRES else ""))
    lectura(lectura_auto(sel))
    st.markdown("")
    if estudio and sel == "CP1":
        lectura("Es el eje clásico de la transformación estructural y resume el "
                f"{pct(res.varianza[0])} de la varianza. A un lado están las economías donde la agricultura pesa en la producción "
                "y sobre todo en el empleo; al otro, las economías urbanas de servicios que exportan manufacturas y tecnología. "
                "Las rentas de recursos también cargan del lado agrario: en buena parte de África la minería convive con un "
                "campo que ocupa a la mayoría de los trabajadores.")
    elif estudio and sel == "CP2":
        lectura(f"Resume el {pct(res.varianza[1])} de la varianza y separa dos maneras de insertarse en el comercio mundial. "
                "Del lado positivo están las economías que exportan sobre todo combustibles y minerales, con rentas altas y "
                "población urbana; del negativo, las que exportan manufacturas o productos del campo. Este componente es "
                "el que distingue a Arabia Saudita, Chile o Colombia de Polonia o Vietnam.")
    elif estudio and sel == "CP3":
        lectura(f"Resume el {pct(res.varianza[2])} de la varianza. Mide cuánto pesa la industria, en el PIB y en el empleo, "
                "frente a los servicios. Separa a las plataformas manufactureras (China, Europa central, el sudeste asiático) "
                "de las economías de servicios con un nivel de urbanización parecido.")

st.markdown("## Las economías en el espacio de componentes")
pares_ejes = [(a, b) for i, a in enumerate(res.cargas.columns[:n]) for b in res.cargas.columns[i + 1:n]]
cc1, cc2, cc3 = st.columns([2, 2, 1])
with cc1:
    eje = st.selectbox("Plano", pares_ejes, format_func=lambda t: f"{t[0]} frente a {t[1]}")
with cc2:
    color_por = st.selectbox("Colorear por", ["sin color", "región", "grupo de ingreso"])
with cc3:
    flechas = st.toggle("Ver cargas", value=True)
cx, cy = eje
jx, jy = list(res.cargas.columns).index(cx), list(res.cargas.columns).index(cy)
if color_por == "sin color":
    fig = gr.plano(res.scores, prep.base, cx, cy, res.varianza[jx], res.varianza[jy],
                   cargas_df=res.cargas if flechas else None)
else:
    col = "region" if color_por == "región" else "ingreso"
    d = prep.base[["pais", col]].copy()
    d[cx], d[cy] = res.scores[cx].values, res.scores[cy].values
    fig = px.scatter(d, x=cx, y=cy, color=col, hover_name="pais", color_discrete_sequence=gr.GRUPOS,
                     symbol=col, symbol_sequence=gr.SIMBOLOS)
    fig.update_traces(marker={"size": 10, "line": {"color": gr.FONDO, "width": 1}})
    fig.update_xaxes(title=f"{cx} ({pct(res.varianza[jx])} de la varianza)")
    fig.update_yaxes(title=f"{cy} ({pct(res.varianza[jy])} de la varianza)")
    gr._base(fig, f"Economías en el plano {cx}-{cy} por {color_por}: resume {pct(res.varianza[jx] + res.varianza[jy])} de la información", alto=600)
    fig.update_layout(legend={"orientation": "h", "y": -0.18, "x": 0, "xanchor": "left"})
st.plotly_chart(fig, use_container_width=True)
lectura(
    f"Este plano muestra el {pct(res.varianza[jx] + res.varianza[jy])} de la variación de las once variables; el "
    f"{pct(1 - res.varianza[jx] - res.varianza[jy])} restante queda fuera de la vista. Dos países que se ven juntos pueden "
    "estar lejos en un componente que el plano no dibuja, y por eso K-Means trabaja con todos los componentes conservados, "
    "no solo con estos dos."
)

if "COL" in set(prep.base.iso3):
    i = prep.base.index[prep.base.iso3 == "COL"][0]
    fila = res.scores.iloc[i, :n]
    nota("Colombia en los componentes conservados: " + "; ".join(f"{k} = {num(v, 2)}" for k, v in fila.items()) +
         ". Un valor de 0 es el promedio de la muestra; las unidades son desviaciones estándar de cada componente.")

sc = pd.concat([prep.base[["iso3", "pais", "region"]], res.scores.iloc[:, :n].round(4)], axis=1)
st.download_button("Descargar coordenadas de las economías (CSV)", sc.to_csv(index=False).encode("utf-8"),
                   "coordenadas_acp.csv", "text/csv")
fuente()
