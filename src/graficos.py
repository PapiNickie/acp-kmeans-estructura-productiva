"""Gráficos de la aplicación (Plotly). Cada función recibe resultados de src/analisis.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analisis import corta, etiqueta


def _pc(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"

# Identidad visual: papel cálido, tinta pizarra, óxido como color principal
FONDO = "#FBF8F3"
TINTA = "#1F2A30"
TINTA_2 = "#4A555C"
TENUE = "#8A9298"
REJILLA = "#E6E0D6"
OXIDO = "#C4502E"
AZUL = "#2F6DB5"
NEUTRO = "#ECE8E2"
# Colores por grupo: los cuatro primeros se distinguen entre sí también con daltonismo (validados
# en todos los pares); desde el quinto, cada grupo además tiene su propio símbolo.
GRUPOS = ["#C4502E", "#2F6DB5", "#1A9E77", "#D99A1E", "#7E57C2", "#A8652B", "#C9578A", "#5A7D1E"]
SIMBOLOS = ["circle", "square", "diamond", "triangle-up", "x", "star", "hexagon", "triangle-down"]
DIVERGENTE = [[0.0, AZUL], [0.5, NEUTRO], [1.0, OXIDO]]
FUENTE = "IBM Plex Sans, Segoe UI, sans-serif"


def _base(fig: go.Figure, titulo: str = "", alto: int = 420, leyenda: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": titulo, "x": 0, "xanchor": "left", "font": {"size": 17, "color": TINTA, "family": FUENTE}},
        paper_bgcolor=FONDO, plot_bgcolor=FONDO, height=alto,
        font={"family": FUENTE, "color": TINTA_2, "size": 13},
        margin={"l": 10, "r": 10, "t": 60 if titulo else 20, "b": 10},
        showlegend=leyenda,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "title": None},
        hoverlabel={"bgcolor": "white", "font": {"family": FUENTE, "color": TINTA}},
    )
    fig.update_xaxes(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=TENUE, tickfont={"color": TINTA_2})
    fig.update_yaxes(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=TENUE, tickfont={"color": TINTA_2})
    return fig


def color_grupo(letras: list[str]) -> dict[str, str]:
    return {g: GRUPOS[i % len(GRUPOS)] for i, g in enumerate(sorted(letras))}


def simbolo_grupo(letras: list[str]) -> dict[str, str]:
    return {g: SIMBOLOS[i % len(SIMBOLOS)] for i, g in enumerate(sorted(letras))}


# ------------------------------------------------------------------ exploración
def faltantes(serie: pd.Series, total: int) -> go.Figure:
    s = serie.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[etiqueta(v) for v in s.index], orientation="h", marker_color=OXIDO,
                           hovertemplate="%{y}: %{x} economías sin dato<extra></extra>"))
    fig.update_xaxes(title=f"economías sin dato en la ventana (de {total})")
    return _base(fig, "Datos faltantes por variable", alto=60 + 26 * len(s), leyenda=False)


def histogramas(df: pd.DataFrame, variables: list[str], destacar: str | None = "COL") -> go.Figure:
    largo = df.melt(id_vars=["iso3", "pais"], value_vars=variables, var_name="variable", value_name="valor")
    largo["variable"] = largo["variable"].map(etiqueta)
    ncol = 3
    fig = px.histogram(largo, x="valor", facet_col="variable", facet_col_wrap=ncol, nbins=25,
                       color_discrete_sequence=[AZUL], facet_col_spacing=0.06, facet_row_spacing=0.12)
    fig.update_traces(marker_line_color=FONDO, marker_line_width=1, hovertemplate="%{x}: %{y} economías<extra></extra>")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font={"size": 12, "color": TINTA}))
    fig.update_xaxes(matches=None, showticklabels=True, title=None)
    fig.update_yaxes(title=None)
    if destacar is not None and destacar in set(df.iso3):
        fila = df[df.iso3 == destacar].iloc[0]
        orden = [etiqueta(v) for v in variables]
        for k, v in enumerate(variables):
            r, c = divmod(k, ncol)
            nfil = int(np.ceil(len(variables) / ncol))
            fig.add_vline(x=fila[v], line_color=OXIDO, line_width=2, row=nfil - r, col=c + 1)
        _ = orden
    filas = int(np.ceil(len(variables) / ncol))
    return _base(fig, "Distribución de cada variable (línea roja: Colombia)", alto=40 + 210 * filas, leyenda=False)


def correlaciones(X: pd.DataFrame) -> go.Figure:
    c = X.corr()
    et = [etiqueta(v) for v in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=et, y=et, zmin=-1, zmax=1, colorscale=DIVERGENTE,
                               text=np.round(c.values, 2), texttemplate="%{text}", textfont={"size": 11},
                               hovertemplate="%{y} y %{x}: %{z:.2f}<extra></extra>",
                               colorbar={"title": "r", "thickness": 12}))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    return _base(fig, "Matriz de correlaciones de Pearson (variables del modelo)", alto=200 + 38 * len(et), leyenda=False)


def antes_despues(X_orig: pd.Series, Z: np.ndarray, nombre: str, con_log: bool = False) -> tuple[go.Figure, go.Figure]:
    def caja(x, color, titulo, eje):
        f = go.Figure(go.Box(x=x, marker_color=color, boxpoints="all", jitter=0.45, pointpos=0, name="",
                             line_color=color, fillcolor="rgba(0,0,0,0.04)", hovertemplate="%{x:.2f}<extra></extra>"))
        f.update_xaxes(title=eje)
        f.update_yaxes(showticklabels=False)
        return _base(f, titulo, alto=260, leyenda=False)
    a = caja(X_orig, TENUE, f"{etiqueta(nombre)}: escala original", "porcentaje")
    b = caja(Z, OXIDO, "Después del logaritmo y la estandarización" if con_log else "Después de estandarizar",
             "desviaciones estándar respecto a la media")
    return a, b


# ------------------------------------------------------------------ ACP
def sedimentacion(valores_propios: np.ndarray) -> go.Figure:
    k = np.arange(1, len(valores_propios) + 1)
    fig = go.Figure(go.Scatter(x=k, y=valores_propios, mode="lines+markers", line={"color": OXIDO, "width": 2},
                               marker={"size": 9, "color": OXIDO, "line": {"color": FONDO, "width": 2}},
                               hovertemplate="CP%{x}: valor propio %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line_dash="dot", line_color=TENUE, annotation_text="criterio de Kaiser (valor propio = 1)",
                  annotation_position="top right", annotation_font_color=TINTA_2)
    fig.update_xaxes(title="componente", dtick=1)
    fig.update_yaxes(title="valor propio", rangemode="tozero")
    return _base(fig, "Gráfico de sedimentación", alto=380, leyenda=False)


def varianza(varianza: np.ndarray, acumulada: np.ndarray, umbral: float, n_sel: int) -> go.Figure:
    k = np.arange(1, len(varianza) + 1)
    col = [OXIDO if i <= n_sel else "#D9CFC1" for i in k]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=varianza * 100, name="por componente", marker_color=col,
                         hovertemplate="CP%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=acumulada * 100, name="acumulada", mode="lines+markers",
                             line={"color": TINTA, "width": 2}, marker={"size": 8},
                             hovertemplate="hasta CP%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=umbral * 100, line_dash="dot", line_color=TENUE, annotation_text=f"{umbral * 100:.0f} % de la varianza",
                  annotation_position="bottom right", annotation_font_color=TINTA_2)
    fig.update_xaxes(title="componente", dtick=1)
    fig.update_yaxes(title="% de la varianza total", range=[0, 105])
    return _base(fig, "Varianza explicada por componente y acumulada", alto=380)


def cargas(c: pd.DataFrame, n: int) -> go.Figure:
    d = c.iloc[:, :n]
    et = [etiqueta(v) for v in d.index]
    fig = go.Figure(go.Heatmap(z=d.values, x=list(d.columns), y=et, zmin=-1, zmax=1, colorscale=DIVERGENTE,
                               text=np.round(d.values, 2), texttemplate="%{text}", textfont={"size": 12},
                               hovertemplate="%{y} en %{x}: %{z:.2f}<extra></extra>", colorbar={"title": "carga", "thickness": 12}))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, "Cargas: correlación de cada variable con cada componente", alto=140 + 34 * len(et), leyenda=False)


def contribuciones(c: pd.DataFrame, comp: str) -> go.Figure:
    s = c[comp].sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[etiqueta(v) for v in s.index], orientation="h",
                           marker_color=[OXIDO if v > 0 else AZUL for v in s.values],
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=TENUE)
    fig.update_xaxes(title="carga", range=[-1, 1])
    return _base(fig, f"Qué variables definen {comp}", alto=80 + 30 * len(s), leyenda=False)


def plano(scores: pd.DataFrame, base: pd.DataFrame, cx: str, cy: str, var_x: float, var_y: float,
          color: pd.Series | None = None, cargas_df: pd.DataFrame | None = None, destacar: str = "COL",
          titulo: str | None = None) -> go.Figure:
    d = base[["iso3", "pais", "region"]].copy()
    d[cx], d[cy] = scores[cx].values, scores[cy].values
    fig = go.Figure()
    if color is None:
        fig.add_trace(go.Scatter(x=d[cx], y=d[cy], mode="markers", text=d.pais, customdata=d.region,
                                 marker={"size": 9, "color": AZUL, "opacity": 0.75, "line": {"color": FONDO, "width": 1}},
                                 hovertemplate="%{text}<br>%{customdata}<br>" + cx + " %{x:.2f} · " + cy + " %{y:.2f}<extra></extra>",
                                 showlegend=False))
    else:
        d["grupo"] = color.values
        cmap, smap = color_grupo(sorted(d.grupo.unique())), simbolo_grupo(sorted(d.grupo.unique()))
        for g in sorted(d.grupo.unique()):
            s = d[d.grupo == g]
            fig.add_trace(go.Scatter(x=s[cx], y=s[cy], mode="markers", name=f"Grupo {g}", text=s.pais, customdata=s.region,
                                     marker={"size": 10, "color": cmap[g], "symbol": smap[g], "opacity": 0.85,
                                             "line": {"color": FONDO, "width": 1.5}},
                                     hovertemplate="%{text}<br>%{customdata}<br>Grupo " + g + "<extra></extra>"))
    if cargas_df is not None:
        esc = 0.9 * min(np.abs(d[cx]).max(), np.abs(d[cy]).max())
        etiquetas_pos = []
        for v, fila in cargas_df.iterrows():
            x, y = fila[cx] * esc, fila[cy] * esc
            fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=1, arrowwidth=1.6, arrowcolor=TINTA_2, text="")
            etiquetas_pos.append([x * 1.12, y * 1.12, corta(v)])
        # separa verticalmente las etiquetas que caen casi en el mismo lugar
        rango_y = float(d[cy].max() - d[cy].min()) or 1.0
        rango_x = float(d[cx].max() - d[cx].min()) or 1.0
        etiquetas_pos.sort(key=lambda e: e[1])
        for i in range(1, len(etiquetas_pos)):
            for j in range(i):
                ea, eb = etiquetas_pos[i], etiquetas_pos[j]
                if abs(ea[0] - eb[0]) < 0.16 * rango_x and abs(ea[1] - eb[1]) < 0.05 * rango_y:
                    ea[1] = eb[1] + 0.05 * rango_y
        for x, y, t in etiquetas_pos:
            fig.add_annotation(x=x, y=y, text=t, showarrow=False, font={"size": 12, "color": TINTA},
                               bgcolor="rgba(251,248,243,.8)")
    if destacar in set(d.iso3):
        f = d[d.iso3 == destacar].iloc[0]
        fig.add_trace(go.Scatter(x=[f[cx]], y=[f[cy]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker={"size": 18, "color": "rgba(0,0,0,0)", "line": {"color": TINTA, "width": 2.5}}))
        fig.add_annotation(x=f[cx], y=f[cy], text=f["pais"], showarrow=True, arrowhead=0, ax=40, ay=-34,
                           font={"size": 13, "color": TINTA}, arrowcolor=TINTA, bgcolor="rgba(251,248,243,.9)")
    fig.update_xaxes(title=f"{cx} ({_pc(var_x)} de la varianza)", zeroline=True)
    fig.update_yaxes(title=f"{cy} ({_pc(var_y)} de la varianza)", zeroline=True)
    t = titulo or f"Economías en el plano {cx}-{cy}: resume {_pc(var_x + var_y)} de la información"
    return _base(fig, t, alto=560)


# ------------------------------------------------------------------ K-Means
def codo(t: pd.DataFrame, k_sel: int | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=t.K, y=t.inercia, mode="lines+markers", line={"color": AZUL, "width": 2},
                               marker={"size": 9, "color": AZUL, "line": {"color": FONDO, "width": 2}},
                               hovertemplate="K = %{x}: inercia %{y:.1f}<extra></extra>"))
    if k_sel is not None:
        f = t[t.K == k_sel].iloc[0]
        fig.add_trace(go.Scatter(x=[k_sel], y=[f.inercia], mode="markers", marker={"size": 16, "color": OXIDO}, hoverinfo="skip"))
    fig.update_xaxes(title="número de grupos (K)", dtick=1)
    fig.update_yaxes(title="inercia (suma de distancias al cuadrado)")
    return _base(fig, "Método del codo", alto=360, leyenda=False)


def silueta_k(t: pd.DataFrame, k_sel: int | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=t.K, y=t.silueta, mode="lines+markers", line={"color": AZUL, "width": 2},
                               marker={"size": 9, "color": AZUL, "line": {"color": FONDO, "width": 2}},
                               hovertemplate="K = %{x}: silueta %{y:.3f}<extra></extra>"))
    if k_sel is not None:
        f = t[t.K == k_sel].iloc[0]
        fig.add_trace(go.Scatter(x=[k_sel], y=[f.silueta], mode="markers", marker={"size": 16, "color": OXIDO}, hoverinfo="skip"))
    fig.update_xaxes(title="número de grupos (K)", dtick=1)
    fig.update_yaxes(title="coeficiente de silueta promedio")
    return _base(fig, "Coeficiente de silueta", alto=360, leyenda=False)


def silueta_muestras(sil: np.ndarray, etiquetas: np.ndarray, nombres: pd.Series) -> go.Figure:
    d = pd.DataFrame({"s": sil, "g": etiquetas, "pais": nombres.values}).sort_values(["g", "s"], ascending=[True, True])
    cmap = color_grupo(sorted(d.g.unique()))
    d["pos"] = np.arange(len(d))
    fig = go.Figure()
    for g in sorted(d.g.unique()):
        s = d[d.g == g]
        fig.add_trace(go.Bar(x=s.s, y=s.pos, orientation="h", name=f"Grupo {g}", marker_color=cmap[g], customdata=s.pais,
                             hovertemplate="%{customdata}: %{x:.2f}<extra></extra>", marker_line_width=0))
    fig.add_vline(x=sil.mean(), line_dash="dot", line_color=TINTA, annotation_text=f"promedio {sil.mean():.2f}".replace(".", ","),
                  annotation_position="top")
    fig.add_vline(x=0, line_color=TENUE)
    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_xaxes(title="silueta de cada economía (negativa = más cerca de otro grupo)")
    fig.update_layout(bargap=0.05)
    return _base(fig, "Silueta por economía", alto=520)


def perfil_z(z: pd.DataFrame) -> go.Figure:
    et = [etiqueta(v) for v in z.columns]
    lim = max(1.5, float(np.ceil(np.abs(z.values).max() * 2) / 2))
    fig = go.Figure(go.Heatmap(z=z.values.T, x=[f"Grupo {g}" for g in z.index], y=et, zmin=-lim, zmax=lim,
                               colorscale=DIVERGENTE, text=np.round(z.values.T, 2), texttemplate="%{text}",
                               hovertemplate="%{x} · %{y}: %{z:.2f} desviaciones<extra></extra>",
                               colorbar={"title": "desv.", "thickness": 12}))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, "Media estandarizada de cada variable por grupo (0 = promedio de la muestra)", alto=140 + 34 * len(et), leyenda=False)


def mapa(base: pd.DataFrame, etiquetas: np.ndarray) -> go.Figure:
    d = base[["iso3", "pais"]].copy()
    d["grupo"] = [f"Grupo {g}" for g in etiquetas]
    cmap = {f"Grupo {g}": c for g, c in color_grupo(sorted(set(etiquetas))).items()}
    fig = px.choropleth(d, locations="iso3", color="grupo", hover_name="pais", color_discrete_map=cmap,
                        category_orders={"grupo": sorted(cmap)})
    fig.update_traces(marker_line_color=FONDO, marker_line_width=0.5, hovertemplate="%{hovertext}<br>%{location}<extra></extra>")
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor=FONDO,
                    landcolor="#E2DCD2", showland=True, showcountries=False, lataxis_range=[-58, 85])
    return _base(fig, "Dónde quedó cada grupo (en gris claro: economías fuera de la muestra)", alto=520)


def caja_variable(base: pd.DataFrame, etiquetas: np.ndarray, var: str) -> go.Figure:
    d = base[["pais", var]].copy()
    d["grupo"] = etiquetas
    cmap = color_grupo(sorted(d.grupo.unique()))
    fig = go.Figure()
    for g in sorted(d.grupo.unique()):
        s = d[d.grupo == g]
        fig.add_trace(go.Box(y=s[var], name=f"Grupo {g}", marker_color=cmap[g], boxpoints="all", jitter=0.35,
                             pointpos=0, text=s.pais, hovertemplate="%{text}: %{y:.1f}<extra></extra>"))
    fig.update_yaxes(title=etiqueta(var))
    return _base(fig, f"{etiqueta(var)} por grupo", alto=380, leyenda=False)
