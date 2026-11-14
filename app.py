"""Punto de entrada de la aplicación.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Estructuras productivas · ACP y K-Means", page_icon=":material/public:", layout="wide")

from src import estilo  # noqa: E402

paginas = [
    st.Page("paginas/inicio.py", title="Contexto y pregunta", icon=":material/public:", default=True),
    st.Page("paginas/datos.py", title="Exploración de los datos", icon=":material/table_chart:"),
    st.Page("paginas/preparacion.py", title="Preparación", icon=":material/tune:"),
    st.Page("paginas/acp.py", title="Componentes principales", icon=":material/scatter_plot:"),
    st.Page("paginas/kmeans.py", title="Segmentación K-Means", icon=":material/hub:"),
    st.Page("paginas/resultados.py", title="Perfiles y mapa", icon=":material/map:"),
    st.Page("paginas/conclusiones.py", title="Discusión y conclusiones", icon=":material/menu_book:"),
]
pg = st.navigation(paginas)
estilo.aplicar_css()
estilo.barra_lateral()
pg.run()
