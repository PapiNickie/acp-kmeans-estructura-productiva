import numpy as np
import pandas as pd
import streamlit as st

from src import analisis as an
from src import estado
from src import graficos as gr
from src.estilo import fmt, encabezado, fuente, lectura, nota, num, requiere_variables

encabezado("Paso 2 · Preparación", "Decisiones antes de modelar",
           "Cada decisión de esta página cambia el resultado. Aquí está cada una con su razón; puedes modificarlas en la "
           "barra lateral y ver el efecto en las páginas siguientes.")
if not requiere_variables():
    st.stop()

c = estado.config()
df = estado.corte(c["ventana"])
prep = estado.preparacion(c["ventana"], c["pob_min"], c["variables"], c["log"])
V = list(c["variables"])
es_estudio = (c["variables"] == tuple(estado.VARIABLES_ESTUDIO) and c["ventana"] == estado.VENTANA_ESTUDIO
              and c["pob_min"] == estado.POB_ESTUDIO and set(c["log"]) == set(estado.LOG_ESTUDIO))
if not es_estudio:
    nota("Estás viendo una configuración distinta a la del estudio. Las cifras se recalculan, pero los textos de "
         "justificación se escribieron para la configuración original.", "azul")

st.markdown("### 1. Una fila por país: promedio de cinco años")
lectura(
    f"El panel tiene un valor por país y por año. Se usa el promedio de {c['ventana'][0]} a {c['ventana'][1]} porque la "
    "pregunta es estructural: interesa la forma de la economía, no el ciclo de un año. La ventana incluye la caída de 2020 y "
    "el repunte de precios de las materias primas de 2022; promediarlos evita que un solo año defina el perfil de un país. "
    "Además, un país que no reportó todos los años no se pierde si tiene al menos un dato en la ventana."
)

st.markdown("### 2. Solo economías de al menos un millón de habitantes")
lectura(
    f"Se excluyen {len(prep.excluidas_pob)} economías con menos de {num(c['pob_min'])} millones de habitantes. Son sobre "
    "todo islas del Caribe y del Pacífico, microestados europeos y territorios dependientes, con estructuras extremas: "
    "economías donde el turismo es casi todo o donde una sola refinería pesa más que el resto. Con un algoritmo basado en "
    "distancias, esos casos atípicos arrastrarían los componentes y formarían grupos de uno o dos países. El costo es "
    "explícito: las conclusiones valen para economías medianas y grandes."
)
with st.expander(f"Ver las {len(prep.excluidas_pob)} economías excluidas por tamaño"):
    t = prep.excluidas_pob.rename(columns={"pais": "economía", "region": "región", "poblacion": "población (millones)"})
    st.dataframe(t[["economía", "región", "población (millones)"]].style.format({"población (millones)": fmt(2)}),
                 hide_index=True, use_container_width=True)

st.markdown("### 3. Qué variables entran y cuáles no")
lectura(
    "El modelo usa once variables organizadas en tres preguntas: qué produce el país (agricultura, manufactura y servicios "
    "como porcentaje del PIB, y rentas de recursos naturales), dónde trabaja su gente (empleo agrícola, empleo industrial y "
    "población urbana) y qué vende al mundo (exportaciones como porcentaje del PIB, peso de las manufacturas y de los "
    "combustibles y minerales en las exportaciones de mercancías, y peso de la alta tecnología en las de manufacturas). "
    "Los combustibles y los minerales se suman en una sola variable porque los dos describen la misma especialización "
    "extractiva; separados, el petróleo y el cobre parecerían dos tipos de economía distintos."
)
st.markdown(
    "- Industria con construcción no entra: incluye la manufactura, la minería y los servicios públicos, y repetiría información "
    "de otras dos variables.\n"
    "- Formación bruta de capital no entra: depende mucho del ciclo y de grandes proyectos puntuales.\n"
    "- Remesas y turismo quedan como opciones: hablan de cómo se financia la economía más que de qué produce, y al turismo le "
    f"faltan datos en {int(df.loc[df.poblacion >= c['pob_min'], 'turismo'].isna().sum())} economías grandes.\n"
    "- El PIB per cápita no entra a propósito. Si entrara, el primer componente sería un índice de ingreso y los grupos "
    "repetirían la clasificación por ingreso del Banco Mundial. Fuera del modelo sirve para algo más útil: comprobar después "
    "si grupos formados solo por estructura coinciden o no con niveles de ingreso."
)

st.markdown("### 4. Datos faltantes: se eliminan filas, no se imputan")
lectura(
    f"Después del filtro de tamaño, {len(prep.excluidas_na)} economías tienen al menos un faltante en las variables del modelo. "
    "Se eliminan en lugar de imputarse. Los faltantes no aparecen al azar: se concentran en Estados frágiles o en conflicto "
    "(Siria, Yemen, Sudán, Somalia) y en países con estadísticas incompletas: Venezuela no tiene cuentas nacionales ni "
    "composición de exportaciones en la ventana, a Nigeria le falta el peso de las exportaciones en el PIB y a Iraq el dato "
    "de alta tecnología. Imputar con la media le asignaría a Yemen la estructura de un país promedio, y llenar con cero haría ver "
    "a Nigeria como un país que no exporta petróleo. Ninguna de las dos cosas es un dato. La eliminación tiene un costo que se "
    "discute al final: el grupo de exportadores de recursos pierde varios de sus casos más extremos."
)
with st.expander(f"Ver las {len(prep.excluidas_na)} economías excluidas por datos faltantes"):
    t = prep.excluidas_na.rename(columns={"pais": "economía", "region": "región", "variables_faltantes": "variables sin dato"})
    st.dataframe(t[["economía", "región", "variables sin dato"]], hide_index=True, use_container_width=True)

st.markdown("### 5. Logaritmo para las variables muy asimétricas")
asim_antes = prep.base[V].skew()
asim_desp = prep.X.skew()
tabla = pd.DataFrame({"variable": [an.etiqueta(v) for v in V], "asimetría original": asim_antes.values,
                      "con logaritmo": [num(asim_desp[v], 2) if v in prep.log else "" for v in V],
                      "transformada": ["sí" if v in prep.log else "no" for v in V]}).sort_values("asimetría original", ascending=False)
cc1, cc2 = st.columns([3, 2])
with cc1:
    st.dataframe(tabla.style.format({"asimetría original": fmt(2)}),
                 hide_index=True, use_container_width=True)
with cc2:
    lectura(
        "Las variables con asimetría mayor que 2 se transforman con log(1 + x). El logaritmo acerca a los casos extremos sin "
        "cambiar el orden de los países, y el 1 evita el problema de los ceros (hay países con rentas de recursos de 0,0 %). "
        "Con la transformación, Qatar o Singapur siguen arriba en sus variables, pero dejan de definir solos la dirección "
        "de un componente."
    )

st.markdown("### 6. Estandarización")
rx = prep.base["export_pib"] if "export_pib" in V else prep.base[V[0]]
rm = prep.base["va_manuf"] if "va_manuf" in V else prep.base[V[-1]]
lectura(
    "ACP y K-Means trabajan con varianzas y distancias, así que la escala importa. "
    f"{an.etiqueta(rx.name)} va de {num(rx.min(), 0)} a {num(rx.max(), 0)} y {an.etiqueta(rm.name)} de "
    f"{num(rm.min(), 0)} a {num(rm.max(), 0)}; sin estandarizar, la variable con más varianza dominaría el primer componente y las "
    "distancias entre países. Cada variable se transforma a z = (x − media) / desviación estándar, de modo que todas quedan "
    "con media 0 y desviación 1, y un valor de 1,5 significa lo mismo en cualquier columna: una desviación y media por encima "
    "del promedio de la muestra."
)
v_ej = st.selectbox("Variable para ver el efecto", V, index=V.index("export_pib") if "export_pib" in V else 0,
                    format_func=an.etiqueta)
f1, f2 = gr.antes_despues(prep.base[v_ej], prep.Z[:, V.index(v_ej)], v_ej, v_ej in prep.log)
cc1, cc2 = st.columns(2)
cc1.plotly_chart(f1, use_container_width=True)
cc2.plotly_chart(f2, use_container_width=True)
Zdf = pd.DataFrame(prep.Z, columns=[an.etiqueta(v) for v in V])
st.caption(f"Comprobación: medias de la matriz estandarizada entre {Zdf.mean().min():.1e} y {Zdf.mean().max():.1e}; "
           f"desviaciones (con n en el denominador) iguales a {num(Zdf.std(ddof=0).mean(), 3)}.")

st.markdown("### 7. La matriz que entra al modelo")
st.plotly_chart(gr.correlaciones(prep.X), use_container_width=True)
st.caption("Correlaciones después del logaritmo. Son las que resume el ACP: estandarizar no las cambia.")
mat = pd.concat([prep.base[["iso3", "pais"]], Zdf.round(3)], axis=1)
st.dataframe(mat, hide_index=True, use_container_width=True, height=300)
st.download_button("Descargar la matriz estandarizada (CSV)", mat.to_csv(index=False).encode("utf-8"),
                   "matriz_estandarizada.csv", "text/csv")
fuente()
