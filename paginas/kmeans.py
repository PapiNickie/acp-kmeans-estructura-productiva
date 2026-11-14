import pandas as pd
import streamlit as st

from src import analisis as an
from src import estado
from src import graficos as gr
from src.estilo import fmt, cifras, encabezado, fuente, lectura, nota, num, pct, requiere_variables

encabezado("Paso 4 · Segmentación con K-Means", "¿Cuántos tipos de economía hay?",
           "K-Means sobre los componentes conservados: comparación de valores de K con el método del codo y el "
           "coeficiente de silueta, grupos en el espacio de componentes y silueta de cada economía.")
if not requiere_variables():
    st.stop()

c = estado.config()
a = (c["ventana"], c["pob_min"], c["variables"], c["log"])
prep = estado.preparacion(*a)
res = estado.acp(*a)
n = min(c["n_comp"], len(c["variables"]))
t = estado.tabla_k(*a, n)

with st.expander("Detalle matemático de K-Means y de la silueta"):
    st.markdown("Dadas las coordenadas $f_i \\in \\mathbb{R}^{k}$ de cada economía en los $k$ componentes conservados, "
                "K-Means busca $K$ grupos $C_1, \\dots, C_K$ con centroides $\\mu_1, \\dots, \\mu_K$ que minimicen la inercia")
    st.latex(r"J = \sum_{g=1}^{K} \sum_{i \in C_g} \lVert f_i - \mu_g \rVert^2")
    st.markdown("El algoritmo de Lloyd alterna dos pasos hasta que las asignaciones no cambian: asignar cada economía al "
                "centroide más cercano y recalcular cada centroide como la media de su grupo. Converge a un mínimo local, "
                "así que se corre 10 veces desde arranques distintos (k-means++) y se conserva la solución con menor inercia "
                "(`KMeans(n_clusters=K, n_init=10, random_state=42)`).")
    st.markdown("La silueta de la economía $i$ compara su distancia media a los países de su grupo, $a(i)$, con la distancia "
                "media al grupo vecino más cercano, $b(i)$:")
    st.latex(r"s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \in [-1, 1]")
    st.markdown("Cerca de 1, la economía está bien metida en su grupo; cerca de 0, en la frontera entre dos; negativa, más "
                "cerca de otro grupo que del suyo. La referencia habitual para el promedio es la de Kaufman y Rousseeuw "
                "(1990): de 0,71 a 1, estructura fuerte; de 0,51 a 0,70, razonable; de 0,26 a 0,50, débil, que podría ser "
                "artificial; 0,25 o menos, sin estructura sustancial.")

st.markdown("## Comparación de valores de K")
st.caption(f"K-Means sobre los {n} componentes conservados ({pct(res.acumulada[n - 1])} de la varianza), K de 2 a 10.")
k_sel = int(c["k"])
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(gr.codo(t, k_sel), use_container_width=True)
with cc2:
    st.plotly_chart(gr.silueta_k(t, k_sel), use_container_width=True)
tt = t.rename(columns={"inercia": "inercia", "silueta": "silueta promedio", "caida_inercia_pct": "caída de la inercia (%)"})
st.dataframe(tt.style.format({"inercia": fmt(1), "silueta promedio": fmt(3), "caída de la inercia (%)": fmt(1)}, na_rep=""),
             hide_index=True, use_container_width=True)

k_mejor = int(t.loc[t.silueta.idxmax(), "K"])
estudio = (c["variables"] == tuple(estado.VARIABLES_ESTUDIO) and set(c["log"]) == set(estado.LOG_ESTUDIO)
           and n == estado.N_COMP_ESTUDIO and c["ventana"] == estado.VENTANA_ESTUDIO and c["pob_min"] == estado.POB_ESTUDIO)
if estudio:
    s3, s4 = t.set_index("K").loc[3, "silueta"], t.set_index("K").loc[4, "silueta"]
    d3, d4, d5, d6 = (t.set_index("K").loc[k, "caida_inercia_pct"] for k in (3, 4, 5, 6))
    lectura(
        f"Los dos criterios no apuntan al mismo número. La silueta es máxima con K = 3 ({num(s3, 3)}). La curva del codo no "
        f"tiene un quiebre único: pasar de 2 a 3 grupos reduce la inercia un {num(d3)} %, de 3 a 4 un {num(d4)} %, de 4 a 5 un "
        f"{num(d5)} % y de 5 a 6 apenas un {num(d6)} %; la curva se aplana después de 4 o 5. El estudio elige K = 4. Con "
        f"K = 3, uno de los grupos reúne a la mitad de la muestra y mezcla a Estados Unidos y Francia con China, México y "
        "Polonia: economías de servicios y plataformas manufactureras quedan juntas, justo la distinción que interesa para "
        f"hablar de desindustrialización. El cuarto grupo las separa a cambio de bajar la silueta a {num(s4, 3)}. Es una "
        "decisión entre la mejor métrica y la utilidad del resultado, y queda a la vista: el control de abajo permite ver la "
        "solución con K = 3."
    )
    lectura(
        "<br>Hay que decir también lo que la silueta advierte. Con cualquier K, el promedio queda entre 0,26 y 0,50, la franja "
        "que Kaufman y Rousseeuw llaman estructura débil. Las economías no forman islas separadas por mar abierto; forman un "
        "continuo con zonas más densas. Los grupos son una manera útil de ordenar ese continuo, no tipos puros, y las "
        "economías de frontera merecen leerse una por una."
    )
else:
    lectura(f"Con esta configuración, la silueta es máxima en K = {k_mejor}. Revisa en la curva del codo dónde deja de "
            "valer la pena agregar un grupo y si los grupos resultantes tienen una lectura clara.")

estado.control(st.slider, "k", "Número de grupos (K)", 2, 8)
k = int(st.session_state["k"])
km = estado.kmeans(*a, n, k)
pf = an.perfiles(prep, km)

st.markdown("## Los grupos")
nota("Las letras de los grupos no tienen orden ni jerarquía: A es simplemente el grupo con más economías. Lo que define a "
     "cada grupo es su perfil en las variables originales (página siguiente).", "azul")
negativas = int((km.silueta < 0).sum())
cifras([(str(k), "grupos"), (num(km.silueta_media, 3), "silueta promedio"),
        (str(negativas), "economías con silueta negativa"),
        (" · ".join(str(pf["tam"][g]) for g in pf["tam"].index), "economías en " + ", ".join(pf["tam"].index[:-1]) + " y " + pf["tam"].index[-1])])
st.markdown("")

pares_ejes = [(x, y) for i, x in enumerate(res.cargas.columns[:n]) for y in res.cargas.columns[i + 1:n]]
eje = st.selectbox("Plano", pares_ejes, format_func=lambda tt: f"{tt[0]} frente a {tt[1]}", key="plano_km")
cx, cy = eje
jx, jy = list(res.cargas.columns).index(cx), list(res.cargas.columns).index(cy)
st.plotly_chart(gr.plano(res.scores, prep.base, cx, cy, res.varianza[jx], res.varianza[jy],
                         color=pd.Series(km.etiquetas),
                         titulo=f"Grupos en el plano {cx}-{cy} ({pct(res.varianza[jx] + res.varianza[jy])} de la información)"),
                use_container_width=True)
lectura("Los grupos se formaron con los " + str(n) + " componentes; el plano muestra solo dos. Por eso algunos puntos "
        "parecen estar del lado equivocado de una frontera: en el componente que no se ve, están más cerca de su propio grupo.")

st.markdown("## Silueta de cada economía")
st.plotly_chart(gr.silueta_muestras(km.silueta, km.etiquetas, prep.base.pais), use_container_width=True)
b = prep.base[["iso3", "pais", "region"]].copy()
b["grupo"], b["silueta"] = km.etiquetas, km.silueta
dudosas = b.nsmallest(8, "silueta")
lectura("Las economías peor asignadas son " + ", ".join(f"{r.pais} ({r.grupo}, {num(r.silueta, 2)})" for r in dudosas.itertuples())
        + ". Son casos de frontera: comparten rasgos con dos perfiles y una pequeña variación en los datos podría cambiarlas de grupo.")
if "COL" in set(b.iso3):
    fc = b[b.iso3 == "COL"].iloc[0]
    rk = int((b.silueta < fc.silueta).sum()) + 1
    nota(f"Colombia queda en el grupo {fc.grupo} con una silueta de {num(fc.silueta, 3)}: es la economía número {rk} de "
         f"{len(b)} si se ordenan de peor a mejor asignada.")

st.markdown("## Centroides en el espacio de componentes")
st.dataframe(km.centroides.style.format(fmt(2)), use_container_width=True)
st.caption("Coordenadas del centro de cada grupo en cada componente (0 = promedio de la muestra).")
fuente()
