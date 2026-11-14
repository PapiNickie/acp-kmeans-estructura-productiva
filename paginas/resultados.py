import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analisis as an
from src import estado
from src import graficos as gr
from src.estilo import fmt, encabezado, fuente, lectura, nota, num, requiere_variables

encabezado("Paso 5 · Perfiles", "Qué distingue a cada grupo",
           "Los grupos descritos con las variables originales, su relación con el ingreso y la región (que no entraron "
           "al modelo), el mapa y la búsqueda de cualquier economía.")
if not requiere_variables():
    st.stop()

c = estado.config()
prep, res, km = estado.todo(c)
V = list(c["variables"])
n = min(c["n_comp"], len(V))
pf = an.perfiles(prep, km)
ras = an.rasgos(pf["z"])
cmap = gr.color_grupo(list(pf["tam"].index))
estudio = (c["variables"] == tuple(estado.VARIABLES_ESTUDIO) and set(c["log"]) == set(estado.LOG_ESTUDIO)
           and n == estado.N_COMP_ESTUDIO and c["k"] == estado.K_ESTUDIO and c["ventana"] == estado.VENTANA_ESTUDIO
           and c["pob_min"] == estado.POB_ESTUDIO)

# Nombres descriptivos: solo para la configuración del estudio, donde se revisaron a mano.
NOMBRES = {}
if estudio:
    z = pf["z"]
    NOMBRES = {
        z["va_servicios"].idxmax(): "Economías de servicios",
        z["empleo_agro"].idxmax(): "Economías agrarias",
        z["va_manuf"].idxmax(): "Plataformas manufactureras",
        z["x_extractivas"].idxmax(): "Exportadoras de recursos",
    }
    if len(set(NOMBRES)) < 4:
        NOMBRES = {}

b = prep.base.copy()
b["grupo"], b["silueta"] = km.etiquetas, km.silueta

st.markdown("## Los grupos en una frase")
cols = st.columns(len(pf["tam"]))
for col, g in zip(cols, pf["tam"].index):
    ejemplos = b[b.grupo == g].nlargest(5, "silueta").pais.tolist()
    nombre = NOMBRES.get(g, "")
    col.markdown(
        f'<div class="grupo-tarjeta" style="--c:{cmap[g]}"><h4>Grupo {g}{": " + nombre if nombre else ""}</h4>'
        f'<div class="n">{pf["tam"][g]} economías · PIB per cápita mediano US$ {num(pf["externas"].loc[g, "pib_pc_mediano"], 0)}</div>'
        f'<p>{ras[g].capitalize()}.</p><p style="color:#4A555C">Casos típicos: {", ".join(ejemplos)}.</p></div>',
        unsafe_allow_html=True)
st.caption("Casos típicos: las cinco economías con mayor silueta en cada grupo. El PIB per cápita no participó en la formación de los grupos.")

st.markdown("## Perfil de cada grupo")
st.plotly_chart(gr.perfil_z(pf["z"]), use_container_width=True)
med = pf["medianas"].T
med.columns = [f"Grupo {g}" for g in med.columns]
med["Toda la muestra"] = prep.base[V].median().values
med.index = [an.etiqueta(v) for v in med.index]
st.markdown("#### Medianas en la escala original")
st.dataframe(med.style.format(fmt(1)), use_container_width=True)
st.caption("Se muestran medianas porque varias variables tienen colas largas; las medias están en el CSV descargable.")

if estudio and NOMBRES:
    inv = {v: k for k, v in NOMBRES.items()}
    m = pf["medianas"]
    S, A, M, R = inv["Economías de servicios"], inv["Economías agrarias"], inv["Plataformas manufactureras"], inv["Exportadoras de recursos"]
    lectura(
        f"El grupo {A} reúne a las economías agrarias: la mediana del empleo agrícola es {num(m.loc[A, 'empleo_agro'])} % y la de "
        f"población urbana {num(m.loc[A, 'urbana'])} %; {int((b[b.grupo == A].region == 'África al sur del Sahara').sum())} de sus "
        f"{pf['tam'][A]} economías están en África al sur del Sahara. "
        f"Los grupos {S} y {M} son los dos destinos posibles de la transformación estructural. En el {S}, los servicios generan "
        f"una mediana de {num(m.loc[S, 'va_servicios'])} % del PIB y la manufactura {num(m.loc[S, 'va_manuf'])} %: son Estados "
        f"Unidos, Europa occidental y algunas economías pequeñas de servicios como Costa Rica o Uruguay. En el {M}, la "
        f"manufactura llega a {num(m.loc[M, 'va_manuf'])} % del PIB, el empleo industrial a {num(m.loc[M, 'empleo_industria'])} % y "
        f"las manufacturas son el {num(m.loc[M, 'x_manuf'])} % de las exportaciones de mercancías: China, Vietnam, México, Corea "
        "del Sur y Europa central. "
        f"El grupo {R} se define por lo que vende: la mediana de combustibles y minerales es {num(m.loc[R, 'x_extractivas'])} % de las "
        f"exportaciones de mercancías y la de rentas de recursos {num(m.loc[R, 'rentas_rn'])} % del PIB. Ahí conviven el Golfo, "
        "Noruega, Australia, Rusia, Chile, Perú y Colombia, con ingresos muy distintos pero una misma forma de insertarse en el mundo."
    )

st.markdown("#### Una variable a la vez")
v_sel = st.selectbox("Variable", V + ["pib_pc_ppa"], format_func=an.etiqueta)
st.plotly_chart(gr.caja_variable(b, km.etiquetas, v_sel), use_container_width=True)

st.markdown("## Ingreso y región: lo que el modelo no vio")
cc1, cc2 = st.columns(2)
orden_ing = ["Ingreso bajo", "Ingreso mediano bajo", "Ingreso mediano alto", "Ingreso alto"]
ing = pd.crosstab(b.grupo, b.ingreso).reindex(columns=[o for o in orden_ing if o in set(b.ingreso)])
reg = pd.crosstab(b.grupo, b.region)
with cc1:
    st.markdown("Economías por grupo de ingreso del Banco Mundial")
    st.dataframe(ing, use_container_width=True)
with cc2:
    st.markdown("Economías por región")
    st.dataframe(reg.T, use_container_width=True)
if estudio and NOMBRES:
    ext = pf["externas"]["pib_pc_mediano"]
    lectura(
        "Los grupos se formaron solo con variables de estructura, sin el ingreso ni la región. Aun así, el ingreso se ordena: el "
        f"grupo agrario tiene un PIB per cápita mediano de US$ {num(ext[A], 0)}, unas {num(ext.drop(A).min() / ext[A], 0)} veces "
        "menos que el de cualquier otro grupo. Pero la relación no es uno a uno. Las exportadoras de recursos "
        f"(US$ {num(ext[R], 0)}) y las plataformas manufactureras (US$ {num(ext[M], 0)}) tienen ingresos medianos casi iguales "
        "y estructuras opuestas, y dentro de las exportadoras conviven Qatar y Bolivia. La estructura productiva dice algo "
        "distinto del nivel de ingreso, que es justamente lo que se quería averiguar."
    )
else:
    lectura("Compara el PIB per cápita mediano de las tarjetas de arriba y estas tablas: si los grupos coincidieran con los "
            "grupos de ingreso, la estructura no estaría agregando información; si no coinciden, la agrega.")

st.markdown("## Mapa")
st.plotly_chart(gr.mapa(prep.base, km.etiquetas), use_container_width=True)

st.markdown("## Buscar una economía")
nombres = prep.base.sort_values("pais")[["iso3", "pais"]]
idx_col = int(nombres.reset_index(drop=True).index[nombres.reset_index(drop=True).iso3 == "COL"][0]) if "COL" in set(nombres.iso3) else 0
pais = st.selectbox("Economía", nombres.iso3.tolist(), index=idx_col,
                    format_func=lambda x: nombres.set_index("iso3").loc[x, "pais"])
f = b[b.iso3 == pais].iloc[0]
vec = an.vecinos(res.scores.iloc[:, :n], prep.base, pais, 6)
vec["grupo"] = vec.iso3.map(b.set_index("iso3").grupo)
cc1, cc2 = st.columns([1, 1])
with cc1:
    nota(f"{f.pais} está en el grupo {f.grupo}{' (' + NOMBRES[f.grupo] + ')' if f.grupo in NOMBRES else ''} con una silueta "
         f"de {num(f.silueta, 3)}. Sus seis vecinos más cercanos en el espacio de los {n} componentes son "
         + ", ".join(f"{r.pais} ({r.grupo})" for r in vec.itertuples()) + ".")
    comp = pd.DataFrame({"variable": [an.etiqueta(v) for v in V], f.pais: [f[v] for v in V],
                         f"mediana grupo {f.grupo}": [pf["medianas"].loc[f.grupo, v] for v in V],
                         "mediana muestra": prep.base[V].median().values})
    st.dataframe(comp.style.format({k: fmt(1) for k in comp.columns[1:]}), hide_index=True, use_container_width=True)
with cc2:
    zf = pd.Series(prep.Z[b.index[b.iso3 == pais][0]], index=V)
    zg = pf["z"].loc[f.grupo]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[an.etiqueta(v) for v in V], x=zg.values, orientation="h", name=f"Grupo {f.grupo} (media)",
                         marker_color="#D9CFC1"))
    fig.add_trace(go.Scatter(y=[an.etiqueta(v) for v in V], x=zf.values, mode="markers", name=f.pais,
                             marker={"size": 12, "color": cmap[f.grupo], "symbol": "diamond", "line": {"color": gr.TINTA, "width": 1}}))
    fig.add_vline(x=0, line_color=gr.TENUE)
    fig.update_xaxes(title="desviaciones estándar respecto al promedio de la muestra")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(gr._base(fig, f"{f.pais} frente a su grupo", alto=440), use_container_width=True)

st.markdown("## Descargar resultados")
salida = prep.base[["iso3", "pais", "region", "ingreso"] + V + ["pib_pc_ppa", "poblacion"]].copy()
salida = pd.concat([salida, res.scores.iloc[:, :n].round(4)], axis=1)
salida["grupo"], salida["silueta"] = km.etiquetas, km.silueta.round(4)
if NOMBRES:
    salida["nombre_grupo"] = salida.grupo.map(NOMBRES)
st.download_button("Descargar la base con componentes y grupo (CSV)", salida.to_csv(index=False).encode("utf-8"),
                   "economias_con_grupos.csv", "text/csv")
fuente()
