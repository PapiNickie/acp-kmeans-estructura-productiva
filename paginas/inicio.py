import pandas as pd
import streamlit as st

from src import analisis as an
from src import estado
from src.estilo import cifras, encabezado, lectura, nota, num, pct

c = estado.config()
prep, res, km = estado.todo(c)
col = prep.base.set_index("iso3").loc["COL"] if "COL" in set(prep.base.iso3) else None

encabezado(
    "Taller individual · ACP y K-Means",
    "¿A qué economías se parece la estructura productiva de Colombia?",
    f"Un recorrido por {len(prep.base)} economías con datos del Banco Mundial: qué producen, dónde trabaja su "
    "gente y qué venden al mundo. El análisis de componentes principales resume esas diferencias en pocas "
    "dimensiones y K-Means busca grupos de países con estructuras parecidas.",
)

st.markdown("## La pregunta")
lectura(
    "La economía del desarrollo describe el crecimiento como un cambio de estructura. Kuznets (1966) y Chenery y "
    "Syrquin (1975) documentaron el patrón clásico: a medida que sube el ingreso, el empleo sale del campo, la "
    "manufactura gana peso y más tarde cede espacio a los servicios. Rodrik (2016) mostró que muchos países de "
    "ingreso medio, en América Latina y en África, empezaron a perder manufactura con ingresos mucho más bajos que "
    "los de Europa cuando le ocurrió lo mismo, y llamó al fenómeno desindustrialización prematura. Un país puede "
    "entonces tener una economía de servicios sin haber pasado por la fábrica, o una economía que vive de vender "
    "petróleo y minerales."
)
if col is not None:
    lectura(
        f"<br>Colombia es un caso para mirar con cuidado. Entre {c['ventana'][0]} y {c['ventana'][1]}, los servicios "
        f"generaron en promedio el {num(col['va_servicios'])} % de su PIB y la manufactura el {num(col['va_manuf'])} %, "
        f"pero el {num(col['x_extractivas'])} % de lo que exportó en mercancías fueron combustibles y minerales. "
        "Por la estructura de su producción se parece a una economía de servicios; por lo que vende, a una "
        "exportadora de recursos. El ejercicio busca con qué países comparte estructura cuando se miran todas las "
        "dimensiones a la vez."
    )

st.markdown("#### Preguntas que orientan el ejercicio")
st.markdown(
    "1. ¿Qué dimensiones comunes resumen las diferencias de estructura productiva entre países?\n"
    "2. ¿Se forman grupos de economías con estructuras parecidas, y qué distingue a cada uno?\n"
    "3. ¿En qué grupo queda Colombia, qué tan nítida es esa asignación y cuáles son sus vecinos más cercanos?"
)

cifras([
    (str(len(prep.base)), "economías en el modelo (cada fila es un país)"),
    (str(len(c["variables"])), "variables cuantitativas de estructura"),
    (f"{c['ventana'][0]}-{c['ventana'][1]}", "ventana de años: cada valor es el promedio del país"),
    (pct(res.acumulada[min(c['n_comp'], len(c['variables'])) - 1], 0), f"de la varianza en {min(c['n_comp'], len(c['variables']))} componentes"),
])

st.markdown("## Los datos")
lectura(
    "Todos los indicadores vienen de los Indicadores del Desarrollo Mundial (WDI) del Banco Mundial y se descargan "
    "de su API pública con <code>scripts/descargar_datos.py</code>. La base original es un panel de 217 economías "
    "entre 2015 y 2024. Para cada país se calcula el promedio de la ventana de años elegida; promediar suaviza "
    "choques de un solo año, como la caída de 2020, y permite usar el dato de un país que no reporta todos los años. "
    "Las variables se agrupan en cuatro preguntas:"
)
filas = [{"Bloque": d["bloque"], "Variable": d["etiqueta"], "Definición del Banco Mundial": d["larga"], "Código WDI": d["codigo"],
          "En el modelo": "sí" if v in c["variables"] else "no"} for v, d in an.VARIABLES.items()]
st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
st.caption("El PIB per cápita (PPA) y la población no entran al modelo: la población sirve para el filtro de tamaño y el "
           "ingreso se usa al final para leer los grupos, sin que haya intervenido en formarlos.")

st.markdown("#### Limitaciones de la fuente")
st.markdown(
    "- El empleo por sector es una estimación modelada de la OIT, no una medición directa en todos los países.\n"
    "- La composición de las exportaciones viene de UN Comtrade y se reporta con rezago; varios países no tienen dato reciente.\n"
    "- Todas las variables son proporciones: describen la forma de la economía, no su tamaño ni su ingreso.\n"
    "- Un promedio de cinco años es una fotografía; no muestra hacia dónde se mueve cada país.\n"
    "- Las cuentas nacionales registran mal la economía informal, que en Colombia ocupa a cerca del 55 % de los trabajadores "
    "según la Gran Encuesta Integrada de Hogares del DANE (2025), y eso afecta sobre todo a los servicios."
)

st.markdown("## Por qué ACP y K-Means")
lectura(
    "Las once variables no son independientes. Un país con mucho empleo agrícola suele tener poca población urbana "
    "y una participación baja de los servicios; esas tres columnas cuentan casi la misma historia. El ACP aprovecha "
    "esa redundancia: encuentra combinaciones de variables que concentran la mayor parte de la variación y que se "
    "pueden leer como ejes de transformación estructural. Si las variables entraran tal cual a K-Means, las que "
    "repiten la misma información pesarían doble en la distancia entre países; sobre los componentes, cada dimensión "
    "cuenta una sola vez."
)
lectura(
    "<br>K-Means responde a la otra mitad de la pregunta. La literatura habla de tipos de economías (exportadoras de "
    "recursos, plataformas manufactureras, economías de servicios), pero nadie entrega esa etiqueta en los datos. "
    "El algoritmo agrupa a los países por cercanía en el espacio de componentes sin saber nada de su región ni de "
    "su ingreso. Si los grupos que salen tienen sentido económico, la tipología tiene sustento; si los países forman "
    "un continuo sin cortes claros, eso también es un hallazgo y la silueta lo va a mostrar."
)

nota("Cómo recorrer la aplicación: en la barra lateral están las decisiones sobre los datos (ventana, tamaño mínimo, "
     "variables y transformación). Las páginas siguen el orden del análisis: exploración, preparación, componentes "
     "principales, K-Means, perfiles de los grupos y discusión. Cada gráfico se recalcula con la configuración que elijas; "
     "el botón de la barra lateral devuelve todo a la configuración del estudio.")

st.markdown("#### Referencias")
st.markdown(
    "- Banco Mundial (2026). World Development Indicators. API v2, consultada el 14 de noviembre de 2026.\n"
    "- DANE (2025). Gran Encuesta Integrada de Hogares: informalidad laboral, trimestre octubre-diciembre de 2025. Boletín técnico.\n"
    "- Chenery, H. y Syrquin, M. (1975). Patterns of Development, 1950-1970. Oxford University Press.\n"
    "- Kuznets, S. (1966). Modern Economic Growth: Rate, Structure and Spread. Yale University Press.\n"
    "- Rodrik, D. (2016). Premature deindustrialization. Journal of Economic Growth, 21(1), 1-33."
)
