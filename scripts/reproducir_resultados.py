"""Reproduce, sin la interfaz, todas las tablas y cifras del estudio.

Uso:
    python scripts/reproducir_resultados.py

Usa las mismas funciones de src/analisis.py que la aplicación y la configuración del estudio
definida en src/estado.py. Escribe en resultados/:

    economias_modelo.csv        las 128 economías con sus variables, componentes, grupo y silueta
    excluidas.csv               economías fuera del modelo y por qué
    varianza_acp.csv            valores propios, varianza explicada y acumulada
    cargas_acp.csv              correlación de cada variable con cada componente
    comparacion_k.csv           inercia y silueta para K de 2 a 10
    perfiles_medianas.csv       medianas de las variables originales por grupo
    perfiles_z.csv              medias estandarizadas por grupo
    robustez_colombia.csv       grupo de Colombia con 2 a 4 componentes y K de 3 a 6
    metricas.json               cifras clave que cita el README
    figuras/*.png               gráficos estáticos para el README
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import analisis as an  # noqa: E402

# Configuración del estudio (la misma que src/estado.py, repetida para no depender de Streamlit)
VARIABLES = ["va_agro", "va_manuf", "va_servicios", "rentas_rn", "empleo_agro", "empleo_industria", "urbana",
             "export_pib", "x_manuf", "x_extractivas", "alta_tec"]
LOG = ["rentas_rn", "export_pib", "alta_tec"]
VENTANA = (2019, 2023)
POB_MIN = 1.0
N_COMP = 3
K = 4

SAL = RAIZ / "resultados"
FIG = SAL / "figuras"


def figuras(prep, res, km, tk, pf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.facecolor": "#FBF8F3", "figure.facecolor": "#FBF8F3"})
    colores = ["#C4502E", "#2F6DB5", "#1A9E77", "#D99A1E"]
    marcas = ["o", "s", "D", "^"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    k = np.arange(1, len(res.valores_propios) + 1)
    ax[0].plot(k, res.valores_propios, "o-", color="#C4502E")
    ax[0].axhline(1, ls=":", color="#8A9298")
    ax[0].set(title="Gráfico de sedimentación", xlabel="componente", ylabel="valor propio", xticks=k)
    ax[1].bar(k, res.varianza * 100, color=["#C4502E" if i <= N_COMP else "#D9CFC1" for i in k])
    ax[1].plot(k, res.acumulada * 100, "o-", color="#1F2A30")
    ax[1].axhline(70, ls=":", color="#8A9298")
    ax[1].set(title="Varianza explicada y acumulada (%)", xlabel="componente", xticks=k, ylim=(0, 105))
    fig.tight_layout(); fig.savefig(FIG / "acp_varianza.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(tk.K, tk.inercia, "o-", color="#2F6DB5"); ax[0].plot([K], tk.set_index("K").loc[K, "inercia"], "o", ms=12, color="#C4502E")
    ax[0].set(title="Método del codo", xlabel="K", ylabel="inercia", xticks=tk.K)
    ax[1].plot(tk.K, tk.silueta, "o-", color="#2F6DB5"); ax[1].plot([K], tk.set_index("K").loc[K, "silueta"], "o", ms=12, color="#C4502E")
    ax[1].set(title="Coeficiente de silueta", xlabel="K", ylabel="silueta promedio", xticks=tk.K)
    fig.tight_layout(); fig.savefig(FIG / "kmeans_k.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    nombres = {"A": "A: economías de servicios", "B": "B: economías agrarias", "C": "C: plataformas manufactureras",
               "D": "D: exportadoras de recursos"}
    for i, g in enumerate(sorted(set(km.etiquetas))):
        m = km.etiquetas == g
        ax.scatter(res.scores.loc[m, "CP1"], res.scores.loc[m, "CP2"], s=42, c=colores[i], marker=marcas[i],
                   edgecolor="#FBF8F3", linewidth=.6, label=nombres.get(g, g))
    i = prep.base.index[prep.base.iso3 == "COL"][0]
    ax.scatter(res.scores.loc[i, "CP1"], res.scores.loc[i, "CP2"], s=230, facecolor="none", edgecolor="#1F2A30", lw=2)
    ax.annotate("Colombia", (res.scores.loc[i, "CP1"], res.scores.loc[i, "CP2"]), xytext=(18, 16), textcoords="offset points",
                fontsize=11, arrowprops={"arrowstyle": "-", "color": "#1F2A30"})
    ax.axhline(0, color="#E6E0D6", zorder=0); ax.axvline(0, color="#E6E0D6", zorder=0)
    ax.set(xlabel=f"CP1, peso del campo ({res.varianza[0] * 100:.1f} %)".replace(".", ","),
           ylabel=f"CP2, especialización extractiva ({res.varianza[1] * 100:.1f} %)".replace(".", ","),
           title=f"Grupos de K-Means en el plano CP1-CP2 ({(res.varianza[0] + res.varianza[1]) * 100:.1f} % de la información)".replace(".", ","))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "grupos_plano.png", dpi=150); plt.close(fig)

    z = pf["z"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    lim = float(np.ceil(np.abs(z.values).max() * 2) / 2)
    im = ax.imshow(z.values.T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(z.index)), [f"Grupo {g}" for g in z.index])
    ax.set_yticks(range(len(z.columns)), [an.etiqueta(v) for v in z.columns])
    for a in range(z.shape[0]):
        for b in range(z.shape[1]):
            ax.text(a, b, f"{z.values[a, b]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="desviaciones respecto al promedio")
    ax.set_title("Media estandarizada de cada variable por grupo")
    fig.tight_layout(); fig.savefig(FIG / "perfiles_grupos.png", dpi=150); plt.close(fig)


def main():
    SAL.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    panel, meta = an.cargar_datos()
    df = an.corte_transversal(panel, meta, *VENTANA)
    prep = an.preparar(df, VARIABLES, LOG, POB_MIN)
    res = an.acp(prep)
    S = res.scores.iloc[:, :N_COMP]
    tk = an.evaluar_k(S.values, 2, 10)
    km = an.segmentar(S.values, K)
    pf = an.perfiles(prep, km)

    base = prep.base[["iso3", "pais", "region", "ingreso"] + VARIABLES + ["pib_pc_ppa", "poblacion"]].copy()
    base = pd.concat([base, S.round(4)], axis=1)
    base["grupo"], base["silueta"] = km.etiquetas, km.silueta.round(4)
    base.to_csv(SAL / "economias_modelo.csv", index=False)
    excl = pd.concat([prep.excluidas_pob.assign(motivo="menos de 1 millón de habitantes"),
                      prep.excluidas_na.assign(motivo="datos faltantes")], ignore_index=True)
    excl.to_csv(SAL / "excluidas.csv", index=False)
    pd.DataFrame({"componente": res.cargas.columns, "valor_propio": res.valores_propios.round(4),
                  "varianza_pct": (res.varianza * 100).round(2), "acumulada_pct": (res.acumulada * 100).round(2)}
                 ).to_csv(SAL / "varianza_acp.csv", index=False)
    res.cargas.round(4).to_csv(SAL / "cargas_acp.csv")
    tk.round(4).to_csv(SAL / "comparacion_k.csv", index=False)
    pf["medianas"].round(2).to_csv(SAL / "perfiles_medianas.csv")
    pf["z"].round(3).to_csv(SAL / "perfiles_z.csv")

    rob = []
    for n in (2, 3, 4):
        for kk in (3, 4, 5, 6):
            k2 = an.segmentar(res.scores.iloc[:, :n].values, kk)
            bb = prep.base.assign(g=k2.etiquetas, s=k2.silueta)
            gc = bb[bb.iso3 == "COL"].g.iloc[0]
            mates = bb[bb.g == gc]
            rob.append({"componentes": n, "K": kk, "silueta_promedio": round(k2.silueta_media, 4),
                        "tam_grupo_colombia": len(mates), "silueta_colombia": round(float(bb[bb.iso3 == "COL"].s.iloc[0]), 4),
                        "con_mexico": "MEX" in set(mates.iso3),
                        "latinoamericanos": ", ".join(sorted(mates[(mates.region == "América Latina y el Caribe") & (mates.iso3 != "COL")].pais))})
    pd.DataFrame(rob).to_csv(SAL / "robustez_colombia.csv", index=False)

    col = base[base.iso3 == "COL"].iloc[0]
    vec = an.vecinos(S, prep.base, "COL", 6)
    metricas = {
        "n_panel_obs": int(len(panel)), "n_economias_panel": int(panel.iso3.nunique()),
        "n_modelo": int(len(prep.base)), "n_excluidas_pob": int(len(prep.excluidas_pob)), "n_excluidas_na": int(len(prep.excluidas_na)),
        "varianza_pct": [round(float(x) * 100, 1) for x in res.varianza[:4]],
        "acumulada_3_pct": round(float(res.acumulada[2]) * 100, 1),
        "valores_propios": [round(float(x), 3) for x in res.valores_propios[:4]],
        "silueta_por_k": {int(r.K): round(float(r.silueta), 3) for r in tk.itertuples()},
        "silueta_k4": round(km.silueta_media, 3), "negativas_k4": [p for p in base[base.silueta < 0].pais],
        "tam_grupos": {g: int(v) for g, v in pf["tam"].items()},
        "pib_pc_mediano": {g: round(float(v)) for g, v in pf["externas"]["pib_pc_mediano"].items()},
        "colombia": {"grupo": col.grupo, "silueta": round(float(col.silueta), 3),
                     "vecinos": vec.pais.tolist(), **{v: round(float(col[v]), 1) for v in VARIABLES + ["pib_pc_ppa"]}},
        "medianas_grupo": pf["medianas"].round(1).to_dict(orient="index"),
    }
    (SAL / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8")
    figuras(prep, res, km, tk, pf)
    print(json.dumps({k: metricas[k] for k in ("n_modelo", "varianza_pct", "acumulada_3_pct", "silueta_por_k", "tam_grupos", "colombia")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
