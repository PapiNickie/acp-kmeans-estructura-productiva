import pandas as pd
import streamlit as st

from src import analisis as an
from src import estado
from src import graficos as gr
from src.estilo import fmt, cifras, encabezado, fuente, lectura, nota, num, requiere_variables

encabezado("Paso 1 · Exploración", "Qué hay en la base",
           "Estructura del panel original, estadísticas descriptivas, datos faltantes y relaciones entre las variables "
           "antes de cualquier transformación.")
if not requiere_variables():
    st.stop()

c = estado.config()
panel, meta = estado.datos()
df = estado.corte(c["ventana"])
prep = estado.preparacion(c["ventana"], c["pob_min"], c["variables"], c["log"])
V = list(c["variables"])

st.markdown("## Estructura del panel original")
cifras([
    (f"{len(panel):,}".replace(",", "."), "observaciones país-indicador-año en el panel descargado"),
    (str(panel.iso3.nunique()), "economías (sin agregados regionales ni de ingreso)"),
    (str(panel.indicador.nunique()), "indicadores descargados del WDI"),
    (f"{panel.anio.min()}-{panel.anio.max()}", "años cubiertos por el panel"),
])
st.markdown("")
with st.expander("Ver las primeras filas del panel tal como lo entrega la API"):
    st.dataframe(panel.head(12), hide_index=True, use_container_width=True)
    st.caption("Formato largo: una fila por país, indicador y año. Columnas de texto: iso3, indicador, codigo_wdi; "
               "numéricas: anio (entero) y valor (decimal).")

st.markdown("## De panel a tabla por país")
lectura(
    f"Para cada economía se promedia cada indicador entre {c['ventana'][0]} y {c['ventana'][1]}. El resultado es una tabla "
    f"de {len(df)} filas, una por economía, con {len(an.VARIABLES)} variables candidatas, el PIB per cápita, la población y tres "
    "columnas de identificación (código, nombre, región) más el grupo de ingreso. Las columnas de identificación no entran al "
    "modelo: sirven para etiquetar y para leer los resultados."
)
tipos = pd.DataFrame({"columna": df.columns, "tipo": [str(t) for t in df.dtypes]})
tipos["uso"] = tipos.columna.map(lambda x: "variable del modelo" if x in V else
                                 ("candidata no usada" if x in an.VARIABLES else
                                  ("lectura posterior" if x in an.EXTERNAS else "identificación o auxiliar")))
cc1, cc2 = st.columns([1, 2])
with cc1:
    st.dataframe(tipos, hide_index=True, use_container_width=True, height=380)
with cc2:
    st.markdown("#### Revisión de calidad")
    st.markdown(
        f"- Filas duplicadas por código de país: {prep.duplicados}.\n"
        f"- Economías con menos de {num(c['pob_min'])} millones de habitantes: {len(prep.excluidas_pob)} (se excluyen, ver Preparación).\n"
        f"- Economías grandes con al menos un faltante en las variables del modelo: {len(prep.excluidas_na)}.\n"
        f"- Economías que entran al modelo: {len(prep.base)}.\n"
        "- Valores fuera de rango: ninguno negativo; las participaciones están entre 0 y 100, salvo las exportaciones, que "
        "superan el 100 % del PIB en economías de reexportación como Singapur o Hong Kong. Es un dato real, no un error."
    )
    st.plotly_chart(gr.faltantes(prep.faltantes, int((df.poblacion >= c["pob_min"]).sum())), use_container_width=True)

st.markdown("## Estadísticas descriptivas")
st.caption(f"Economías que entran al modelo (n = {len(prep.base)}), en la escala original de cada variable.")
desc = prep.base[V].describe().T
desc.index = [an.etiqueta(v) for v in desc.index]
desc = desc.rename(columns={"count": "n", "mean": "media", "std": "desv. estándar", "min": "mínimo", "25%": "p25",
                            "50%": "mediana", "75%": "p75", "max": "máximo"})
desc["asimetría"] = prep.base[V].skew().values
st.dataframe(desc.style.format(fmt(1)).format(fmt(0), subset=["n"]).format(fmt(2), subset=["asimetría"]),
             use_container_width=True)
if "COL" in set(prep.base.iso3):
    colf = prep.base.set_index("iso3").loc["COL", V]
    st.caption("Colombia: " + "; ".join(f"{an.etiqueta(v)} {num(colf[v])}" for v in V) + ".")

st.markdown("## Distribuciones")
st.plotly_chart(gr.histogramas(prep.base, V), use_container_width=True)
lectura(
    "Las rentas de los recursos naturales, las exportaciones como porcentaje del PIB y las exportaciones de alta tecnología "
    "tienen colas largas a la derecha: unas pocas economías concentran valores muy altos (los productores del Golfo, los "
    "centros de reexportación, los fabricantes de electrónica del sudeste asiático). En el ACP esas pocas observaciones "
    "pesarían demasiado; la página de preparación explica cómo se tratan."
)

st.markdown("## Relaciones entre variables")
st.plotly_chart(gr.correlaciones(prep.base[V]), use_container_width=True)
corr = prep.base[V].corr()
pares = pd.Series({(a, b): corr.loc[a, b] for i, a in enumerate(V) for b in V[i + 1:]})
top = pares.abs().sort_values(ascending=False).head(4)
frases = [f"{an.etiqueta(a)} y {an.etiqueta(b)} ({num(pares[(a, b)], 2)})" for a, b in top.index]
lectura(
    "Las correlaciones más fuertes en la escala original son " + ", ".join(frases[:-1]) + " y " + frases[-1] + ". "
    "Hay dos bloques visibles. El primero junta la agricultura (en producción y en empleo) con la población rural y la "
    "ausencia de servicios: es el eje clásico de la transformación estructural. El segundo enfrenta a las exportaciones de "
    "combustibles y minerales, y las rentas que dejan, con las exportaciones de manufacturas. Que existan bloques así es "
    "lo que justifica el ACP: varias columnas cuentan la misma historia y conviene resumirlas."
)
nota("Las correlaciones de este gráfico usan las variables en su escala original. Las del modelo, después del logaritmo, "
     "se muestran en la página de preparación.", "azul")
fuente()
