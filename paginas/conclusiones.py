import pandas as pd
import streamlit as st

from src import analisis as an
from src import estado
from src.estilo import fmt, encabezado, fuente, lectura, nota, num, pct, requiere_variables

encabezado("Paso 6 · Discusión", "Qué aprendimos de los datos",
           "Respuestas a las preguntas del estudio, la posición de Colombia, la solidez de esa conclusión y los límites "
           "que hay que tener presentes al usar los resultados.")
if not requiere_variables():
    st.stop()

c = estado.config()
a = (c["ventana"], c["pob_min"], c["variables"], c["log"])
prep, res, km = estado.todo(c)
V = list(c["variables"])
n = min(c["n_comp"], len(V))
pf = an.perfiles(prep, km)
b = prep.base.copy()
b["grupo"], b["silueta"] = km.etiquetas, km.silueta
estudio = (c["variables"] == tuple(estado.VARIABLES_ESTUDIO) and set(c["log"]) == set(estado.LOG_ESTUDIO)
           and n == estado.N_COMP_ESTUDIO and c["k"] == estado.K_ESTUDIO and c["ventana"] == estado.VENTANA_ESTUDIO
           and c["pob_min"] == estado.POB_ESTUDIO)
if not estudio:
    nota("Esta discusión se escribió para la configuración del estudio (once variables, logaritmo en tres, tres componentes, "
         "K = 4, ventana 2019-2023, un millón de habitantes). Con la configuración actual, las cifras de esta página se "
         "recalculan, pero la redacción puede no corresponder. Usa el botón de la barra lateral para volver.", "azul")

z = pf["z"]
G = {"serv": z["va_servicios"].idxmax(), "agro": z["empleo_agro"].idxmax(), "manuf": z["va_manuf"].idxmax(),
     "rec": z["x_extractivas"].idxmax()} if {"va_servicios", "empleo_agro", "va_manuf", "x_extractivas"} <= set(V) else {}
m = pf["medianas"]
ext = pf["externas"]["pib_pc_mediano"]

st.markdown("## 1. Qué dimensiones resumen la estructura productiva")
c1 = res.cargas["CP1"]
lectura(
    f"Tres componentes conservan el {pct(res.acumulada[2])} de la variación de las once variables. El primero, con el "
    f"{pct(res.varianza[0])}, es el eje de la transformación estructural: lo definen el empleo agrícola (carga de "
    f"{num(c1.get('empleo_agro', float('nan')), 2)}) y la agricultura en el PIB ({num(c1.get('va_agro', float('nan')), 2)}), frente a "
    f"los servicios ({num(c1.get('va_servicios', float('nan')), 2)}) y las exportaciones de manufacturas "
    f"({num(c1.get('x_manuf', float('nan')), 2)}). El segundo, con el {pct(res.varianza[1])}, separa a las economías que exportan "
    "combustibles y minerales de las que exportan manufacturas. El tercero, con el "
    f"{pct(res.varianza[2])}, mide el peso de la industria frente a los servicios. Con tres ejes se pierde un "
    f"{pct(1 - res.acumulada[2])} de la información; a cambio, cada eje tiene una lectura económica que se puede defender."
)

st.markdown("## 2. Qué grupos aparecen y qué tan nítidos son")
if G and len(set(G.values())) == 4:
    lectura(
        f"Con K = {c['k']} aparecen cuatro perfiles. Las economías agrarias (grupo {G['agro']}, {pf['tam'][G['agro']]} países) tienen "
        f"una mediana de {num(m.loc[G['agro'], 'empleo_agro'])} % del empleo en el campo y un PIB per cápita mediano de "
        f"US$ {num(ext[G['agro']], 0)}. Las economías de servicios (grupo {G['serv']}, {pf['tam'][G['serv']]}) y las plataformas "
        f"manufactureras (grupo {G['manuf']}, {pf['tam'][G['manuf']]}) son urbanas y exportan manufacturas, pero en las primeras la "
        f"manufactura pesa {num(m.loc[G['serv'], 'va_manuf'])} % del PIB y en las segundas {num(m.loc[G['manuf'], 'va_manuf'])} %. "
        f"Las exportadoras de recursos (grupo {G['rec']}, {pf['tam'][G['rec']]}) venden sobre todo combustibles y minerales "
        f"(mediana de {num(m.loc[G['rec'], 'x_extractivas'])} % de sus exportaciones de mercancías)."
    )
negs = b[b.silueta < 0].sort_values("silueta")
lectura(
    f"<br>Los grupos no son islas. La silueta promedio es {num(km.silueta_media, 3)}, dentro de la franja que Kaufman y "
    "Rousseeuw (1990) llaman estructura débil, y "
    + (f"{len(negs)} economías tienen silueta negativa: " + ", ".join(f"{r.pais} ({r.grupo})" for r in negs.itertuples()) + ". "
       if len(negs) else "ninguna economía tiene silueta negativa. ")
    + ("Brasil y Sudáfrica, por ejemplo, exportan minerales, pero también tienen bases manufactureras grandes; quedan en la "
       "frontera porque las dos descripciones les sirven a medias. " if {"BRA", "ZAF"} <= set(negs.iso3) else "")
    + "La lectura correcta es la de un continuo con zonas densas: "
    "los grupos ordenan ese continuo y ayudan a compararlo, pero no son categorías naturales."
)

st.markdown("## 3. Dónde queda Colombia")
if "COL" in set(b.iso3):
    fc = b[b.iso3 == "COL"].iloc[0]
    g = fc.grupo
    vec = an.vecinos(res.scores.iloc[:, :n], prep.base, "COL", 6)
    pos_x = int((prep.base["export_pib"] < fc["export_pib"]).sum()) + 1 if "export_pib" in V else None
    lectura(
        f"Colombia queda en el grupo {g}, el de las exportadoras de recursos, con una silueta de {num(fc.silueta, 3)}. Sus vecinos "
        "más cercanos en el espacio de componentes son " + ", ".join(vec.pais.head(5)) + ". La compañía es más "
        "sudamericana que petrolera: Colombia comparte grupo con Arabia Saudita y Noruega, pero se parece más a Brasil, Perú o "
        "Ecuador que a ellos."
    )
    if {"x_extractivas", "rentas_rn", "x_manuf", "va_manuf", "export_pib"} <= set(V) and G:
        lectura(
            f"<br>Frente a la mediana de su grupo, Colombia es una exportadora de recursos atenuada. Los combustibles y minerales son "
            f"el {num(fc['x_extractivas'])} % de sus exportaciones de mercancías (mediana del grupo: {num(m.loc[g, 'x_extractivas'])} %) y "
            f"las rentas de recursos el {num(fc['rentas_rn'])} % del PIB (mediana: {num(m.loc[g, 'rentas_rn'])} %). Exporta más "
            f"manufacturas que sus pares ({num(fc['x_manuf'])} % frente a {num(m.loc[g, 'x_manuf'])} %). Y exporta poco en total: el "
            f"{num(fc['export_pib'])} % del PIB, cuando la mediana de la muestra es {num(prep.base['export_pib'].median())} %; solo "
            f"{pos_x - 1} de las {len(b)} economías exportan una fracción menor de su producción."
        )
        lectura(
            f"<br>El dato que conecta con la pregunta de partida es la manufactura. En Colombia genera el {num(fc['va_manuf'])} % del PIB, "
            f"casi lo mismo que en la mediana de las economías de servicios ({num(m.loc[G['serv'], 'va_manuf'])} %), pero con un PIB "
            f"per cápita de US$ {num(fc['pib_pc_ppa'], 0)} frente a US$ {num(ext[G['serv']], 0)}. Las plataformas manufactureras, con un "
            f"ingreso mediano parecido al colombiano (US$ {num(ext[G['manuf']], 0)}), tienen una manufactura de "
            f"{num(m.loc[G['manuf'], 'va_manuf'])} %. Es el patrón que Rodrik (2016) describió como desindustrialización prematura: "
            "una participación industrial de país rico con un ingreso de país medio. El ejercicio no dice por qué ocurrió; dice que "
            "Colombia está más cerca de ese patrón que del de México o Polonia."
        )

st.markdown("#### ¿Cambia la conclusión si cambian las decisiones?")
filas = []
for nn in (2, 3, 4):
    if nn > len(V):
        continue
    for kk in (3, 4, 5, 6):
        kmx = estado.kmeans(*a, nn, kk)
        bb = prep.base.assign(g=kmx.etiquetas, s=kmx.silueta)
        if "COL" not in set(bb.iso3):
            continue
        gc = bb[bb.iso3 == "COL"].g.iloc[0]
        mates = bb[bb.g == gc]
        lat = sorted(mates[(mates.region == "América Latina y el Caribe") & (mates.iso3 != "COL")].pais)
        filas.append({"componentes": nn, "K": kk, "silueta promedio": kmx.silueta_media, "economías en el grupo de Colombia": len(mates),
                      "silueta de Colombia": bb[bb.iso3 == "COL"].s.iloc[0], "con México": "sí" if "MEX" in set(mates.iso3) else "no",
                      "latinoamericanos en el mismo grupo": ", ".join(lat)})
rob = pd.DataFrame(filas)
st.dataframe(rob.style.format({"silueta promedio": fmt(3), "silueta de Colombia": fmt(2)}), hide_index=True, use_container_width=True)
if len(rob):
    sets = [set(x.split(", ")) if x else set() for x in rob["latinoamericanos en el mismo grupo"]]
    siempre = sorted(set.intersection(*sets))
    con_mex = int((rob["con México"] == "sí").sum())
    casi = sorted(p for p in set.union(*sets) if sum(p in s for s in sets) == len(sets) - 1)
    lectura(
        f"En las {len(rob)} combinaciones de componentes y grupos, Colombia queda junto a " + ", ".join(siempre[:-1]) + " y "
        + siempre[-1] + (f"; en {len(rob) - 1} de ellas, también junto a " + " y ".join(casi) if casi else "") + ". "
        + (f"En ninguna queda junto a México. " if con_mex == 0 else f"Queda junto a México en {con_mex}. ")
        + f"Lo que sí cambia es la nitidez: la silueta de Colombia va de {num(rob['silueta de Colombia'].min(), 2)} "
        f"a {num(rob['silueta de Colombia'].max(), 2)} y baja cuando se piden más grupos. La pertenencia es estable; la distancia a la "
        "frontera, no tanto." if estudio else
        "La tabla muestra si el grupo de Colombia se mantiene al cambiar el número de componentes y de grupos."
    )

st.markdown("## 4. Para qué sirven estos resultados y para qué no")
lectura(
    "Sirven para elegir con quién compararse. Cuando se discute la política industrial o la transición energética de Colombia, "
    "los referentes naturales que sugieren los datos son Perú, Chile, Brasil o Ecuador, no México, Corea del Sur o Vietnam, que "
    "tienen otra estructura. También sirven para ubicar rápidamente a un país nuevo: basta con proyectarlo en los componentes. "
    "No sirven para afirmar causas. Pertenecer al grupo de las exportadoras de recursos no explica el ingreso de un país ni "
    "predice su crecimiento, y ningún resultado de esta aplicación debería leerse así."
)

st.markdown("## 5. Limitaciones")
st.markdown(
    f"- Muestra seleccionada por disponibilidad. {len(prep.excluidas_na)} economías grandes quedan fuera por datos faltantes, entre "
    "ellas varias exportadoras de petróleo (Venezuela, Nigeria, Iraq, Libia, Argelia). El grupo de recursos está "
    "subrepresentado justo en sus casos más extremos.\n"
    "- Las variables son proporciones. Describen la forma de la economía, no su tamaño ni la calidad de lo que produce. La "
    "alta tecnología, por ejemplo, se mide por el tipo de producto exportado: un país que ensambla computadores con piezas "
    "importadas aparece igual que uno que las diseña.\n"
    "- Es una fotografía de 2019-2023. No muestra trayectorias: no distingue a un país que se está industrializando de uno "
    "que se está desindustrializando si hoy tienen la misma estructura.\n"
    "- K-Means supone grupos compactos y de tamaño parecido, y la silueta indica una estructura débil. La elección de K = 4 "
    "fue una decisión de interpretación sobre una métrica que prefería K = 3.\n"
    "- El empleo sectorial es una estimación modelada de la OIT y la economía informal está mal medida en las cuentas "
    "nacionales, lo que afecta sobre todo a los países de ingreso bajo y medio."
)

st.markdown("## Conclusión")
lat_vec = []
if "COL" in set(b.iso3):
    vv = an.vecinos(res.scores.iloc[:, :n], prep.base, "COL", 15)
    lat_vec = vv[vv.region == "América Latina y el Caribe"].pais.head(3).tolist()
lectura(
    "Las diferencias de estructura productiva entre países se resumen bien en tres ejes: el paso del campo a la ciudad, la "
    "especialización extractiva y el peso de la industria. Sobre esos ejes, los países se ordenan en cuatro perfiles que no "
    "repiten la clasificación por ingreso, aunque se relacionan con ella. Colombia aparece de manera estable entre las "
    "exportadoras de recursos sudamericanas, con una economía cerrada al comercio y una manufactura del tamaño de la de un "
    "país rico de servicios. Si la pregunta era a qué economías se parece, la respuesta de los datos es "
    + (", ".join(lat_vec[:-1]) + " y " + lat_vec[-1] if len(lat_vec) >= 2 else "la de sus vecinos sudamericanos") + "."
)
fuente()
