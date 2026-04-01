import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import os
import unicodedata
import requests
import base64

# 🔤 Quitar acentos
def quitar_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

# 🔤 Orden consistente (alfabético ignorando acentos)
def ordenar_nombres(lista_nombres):
    return sorted(lista_nombres, key=lambda x: quitar_acentos(x).lower())

# 🎨 Mapa de colores fijo (consistencia entre gráficas)
def crear_mapa_colores(nombres):
    colores = plt.cm.tab10.colors
    return {nombre: colores[i % len(colores)] for i, nombre in enumerate(nombres)}

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Liga Padel Guimaneta",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CARGA DE DATOS (Adaptada específicamente a tus archivos)
# ─────────────────────────────────────────────

@st.cache_data(ttl=10)
def cargar_datos():
    base_url = "https://raw.githubusercontent.com/AlvarG12/padel_guimaneta/main/data/"
    print("🔍 DEBUG: Cargando datos...")
    jugadores = pd.read_csv(base_url + "jugadores.csv")
    print(f"✅ Jugadores cargados: {len(jugadores)} filas")

    def leer_temp(suffix, label):
        url_p = base_url + f"partidos_{suffix}.csv"
        url_pj = base_url + f"partido_jugadores_{suffix}.csv"

        try:
            p = pd.read_csv(url_p)
            pj = pd.read_csv(url_pj)

            p["temporada"] = label
            pj["temporada"] = label

            print(f"✅ Temporada {label}: {len(p)} partidos, {len(pj)} PJ")
            return p, pj

        except Exception as e:
            print(f"❌ Error cargando {suffix}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    p24, pj24 = leer_temp("24_25", "2024/25")
    p25, pj25 = leer_temp("25_26", "2025/26")

    partidos = pd.concat([p24, p25], ignore_index=True)
    partido_jugadores = pd.concat([pj24, pj25], ignore_index=True)

    print(f"📊 Total: {len(partidos)} partidos, {len(partido_jugadores)} PJ")
    print(f"🏷️ Temporadas en partidos: {partidos['temporada'].unique() if 'temporada' in partidos.columns else 'NO'}")
    print(f"🏷️ Temporadas en PJ: {partido_jugadores['temporada'].unique() if 'temporada' in partido_jugadores.columns else 'NO'}")

    if "equipo_ganador" not in partidos.columns:
        partidos["equipo_ganador"] = partidos.apply(
            lambda x: 1 if x["juegos_equipo1"] > x["juegos_equipo2"] else 2, axis=1
        )

    return jugadores, partidos, partido_jugadores

@st.cache_data(ttl=10)
def construir_df(jugadores, partidos, partido_jugadores):
    print("🔍 DEBUG: Construyendo DF CORREGIDO...")

    # 🔑 CREAR ID ÚNICO GLOBAL (clave para evitar colisiones)
    partidos = partidos.copy()
    partido_jugadores = partido_jugadores.copy()

    partidos["id_partido"] = partidos["id_partido"].astype(str) + "_" + partidos["temporada"].astype(str)
    partido_jugadores["id_partido"] = partido_jugadores["id_partido"].astype(str) + "_" + partido_jugadores["temporada"].astype(str)

    # LIMPIAR DUPLICADOS
    partidos_clean = partidos.drop_duplicates(subset=['id_partido'])
    pj_clean = partido_jugadores.drop_duplicates()

    print(f"🔍 Partidos limpios: {len(partidos_clean)}")
    print(f"🔍 PJ limpios: {len(pj_clean)}")

    # MERGE PASO A PASO
    df = pj_clean.merge(jugadores[['id_jugador', 'nombre']], on="id_jugador", how='left')
    df = df.merge(partidos_clean, on="id_partido", how='left')

    # TEMPORADA LIMPIA
    if 'temporada_x' in df.columns:
        df['temporada'] = df['temporada_x'].fillna(df.get('temporada_y', 'Sin temporada'))
    else:
        df['temporada'] = df['temporada'].fillna('Sin temporada')

    # CÁLCULOS VECTORIZADOS (RÁPIDOS Y CORRECTOS)
    df['victoria'] = (df['equipo'] == df['equipo_ganador']).astype(int)
    df['juegos_ganados'] = np.where(df['equipo'] == 1, df['juegos_equipo1'], df['juegos_equipo2'])
    df['juegos_perdidos'] = np.where(df['equipo'] == 1, df['juegos_equipo2'], df['juegos_equipo1'])

    # DIAGNÓSTICO FINAL (SIN ASSERT)
    partidos_unicos = df['id_partido'].nunique()
    filas_total = len(df)
    pj_promedio = filas_total / partidos_unicos if partidos_unicos > 0 else 0

    print(f"✅ FINAL: {filas_total} filas, {partidos_unicos} partidos, {pj_promedio:.1f} PJ/partido")

    return df

@st.cache_data(ttl=10)
def calcular_clasificacion(df):
    clas = df.groupby("nombre").agg(
        partidos_jugados=("id_partido", "count"),
        victorias=("victoria", "sum"),
        juegos_ganados=("juegos_ganados", "sum"),
        juegos_perdidos=("juegos_perdidos", "sum"),
        jornadas_participadas=("id_jornada", "nunique")
    ).reset_index()

    clas["diferencia_juegos"] = clas["juegos_ganados"] - clas["juegos_perdidos"]
    clas["derrotas"] = clas["partidos_jugados"] - clas["victorias"]
    clas["porcentaje_victorias"] = (clas["victorias"] / clas["partidos_jugados"] * 100).round(2)
    total_jornadas = df["id_jornada"].nunique()
    clas["jornadas"] = clas["jornadas_participadas"].astype(str) + "/" + str(total_jornadas)

    # Orden con desempates en cascada
    clas = clas.sort_values(
        by=[
            "porcentaje_victorias",
            "diferencia_juegos",
            "victorias",
            "jornadas_participadas",
            "juegos_ganados"
        ],
        ascending=False
    ).reset_index(drop=True)

    clas.index = clas.index + 1
    return clas

@st.cache_data(ttl=10)
def calcular_ranking_por_jornada(df):
    jornadas = sorted(df["id_jornada"].unique())
    clasificacion_jornadas = []

    for j in jornadas:
        df_j = df[df["id_jornada"] <= j]
        clas_j = df_j.groupby("nombre").agg(
            partidos_jugados=("id_partido", "count"),
            victorias=("victoria", "sum"),
            juegos_ganados=("juegos_ganados", "sum"),
            juegos_perdidos=("juegos_perdidos", "sum"),
            jornadas_participadas=("id_jornada", "nunique")
        ).reset_index()

        clas_j["diferencia_juegos"] = clas_j["juegos_ganados"] - clas_j["juegos_perdidos"]
        clas_j["porcentaje_victorias"] = (clas_j["victorias"] / clas_j["partidos_jugados"] * 100).round(2)

        # Orden con desempates en cascada
        clas_j = clas_j.sort_values(
            by=[
                "porcentaje_victorias",
                "diferencia_juegos",
                "victorias",
                "jornadas_participadas",
                "juegos_ganados"
            ],
            ascending=False
        ).reset_index(drop=True)

        clas_j["rank"] = range(1, len(clas_j) + 1)
        clas_j["hasta_jornada"] = j
        clasificacion_jornadas.append(clas_j)

    return pd.concat(clasificacion_jornadas, ignore_index=True)

@st.cache_data(ttl=10)
def calcular_ranking_por_partido(df):
    """Calcula el ranking acumulado tras cada partido individual (no por jornada)."""
    df = df.copy()
    df["_id_num"] = df["id_partido"].astype(str).str.split("_").str[0].astype(int)
    partidos_ord = df.drop_duplicates("id_partido").sort_values(["id_jornada", "_id_num"])

    resultados = []
    for i, (_, partido_row) in enumerate(partidos_ord.iterrows(), start=1):
        pid = partido_row["id_partido"]
        df_hasta = df[df["id_partido"].isin(partidos_ord.iloc[:i]["id_partido"])]

        clas = df_hasta.groupby("nombre").agg(
            victorias=("victoria", "sum"),
            partidos_jugados=("id_partido", "count"),
            juegos_ganados=("juegos_ganados", "sum"),
            juegos_perdidos=("juegos_perdidos", "sum"),
        ).reset_index()
        clas["diferencia_juegos"] = clas["juegos_ganados"] - clas["juegos_perdidos"]
        clas["porcentaje_victorias"] = (clas["victorias"] / clas["partidos_jugados"] * 100).round(1)
        clas = clas.sort_values(
            by=["porcentaje_victorias", "diferencia_juegos", "juegos_ganados"], ascending=False
        )
        clas["rank"] = range(1, len(clas) + 1)
        clas["hasta_partido"] = i
        clas["id_jornada"] = partido_row["id_jornada"]
        resultados.append(clas)

    return pd.concat(resultados, ignore_index=True)

@st.cache_data(ttl=10)
def calcular_enfrentamientos(df):
    jugadores_unicos = df["nombre"].unique()
    rows = []
    for p1, p2 in itertools.combinations(jugadores_unicos, 2):
        a = df[(df["nombre"] == p1) & (df["equipo"] == 1)].merge(
            df[(df["nombre"] == p2) & (df["equipo"] == 2)], on="id_partido", suffixes=("_p1", "_p2"))
        b = df[(df["nombre"] == p1) & (df["equipo"] == 2)].merge(
            df[(df["nombre"] == p2) & (df["equipo"] == 1)], on="id_partido", suffixes=("_p1", "_p2"))
        combined = pd.concat([a, b])
        if not combined.empty:
            rows.append({
                "jugador1": p1, "jugador2": p2,
                "partidos_totales": combined["id_partido"].nunique(),
                "victorias_jugador1": combined["victoria_p1"].sum(),
                "victorias_jugador2": combined["victoria_p2"].sum(),
                "juegos_ganados_jugador1": combined["juegos_ganados_p1"].sum(),
                "juegos_ganados_jugador2": combined["juegos_ganados_p2"].sum(),
            })
    df_enf = pd.DataFrame(rows)
    if not df_enf.empty:
        df_enf["pct_j1"] = (df_enf["victorias_jugador1"] / df_enf["partidos_totales"] * 100).round(1)
        df_enf["pct_j2"] = (df_enf["victorias_jugador2"] / df_enf["partidos_totales"] * 100).round(1)
    return df_enf

@st.cache_data(ttl=10)
def calcular_parejas(df):
    jugadores_unicos = df["nombre"].unique()
    rows = []
    for p1, p2 in itertools.combinations(jugadores_unicos, 2):
        df_p1 = df[df["nombre"] == p1]
        df_p2 = df[df["nombre"] == p2]
        juntos = pd.merge(df_p1, df_p2, on=["id_partido", "id_jornada", "equipo"], suffixes=("_p1", "_p2"))
        if not juntos.empty:
            total = juntos["id_partido"].nunique()
            victorias = juntos["victoria_p1"].sum()
            rows.append({
                "jugador1": p1, "jugador2": p2,
                "partidos_juntos": total,
                "victorias_juntos": victorias,
                "derrotas_juntos": total - victorias,
                "juegos_ganados_juntos": juntos["juegos_ganados_p1"].sum(),
                "juegos_perdidos_juntos": juntos["juegos_perdidos_p1"].sum(),
            })
    df_pj = pd.DataFrame(rows)
    if not df_pj.empty:
        df_pj["porcentaje_victorias"] = (df_pj["victorias_juntos"] / df_pj["partidos_juntos"] * 100).round(1)
        df_pj = df_pj.sort_values(
            by=["porcentaje_victorias", "victorias_juntos", "partidos_juntos"], ascending=False
        )
    return df_pj

@st.cache_data(ttl=10)
def calcular_rachas(df):
    rachas_activas, rachas_max_v, rachas_max_d = [], [], []

    df = df.copy()
    df["_id_partido_num"] = df["id_partido"].astype(str).str.split("_").str[0].astype(int)

    for nombre in df["nombre"].unique():
        df_j = df[df["nombre"] == nombre].sort_values(by=["id_jornada", "_id_partido_num"])
        if df_j.empty:
            continue

        # Racha activa
        last = df_j.iloc[-1]["victoria"]
        tipo = "victorias" if last else "derrotas"
        count = 0
        for _, row in df_j.iloc[::-1].iterrows():
            if (row["victoria"] and tipo == "victorias") or (not row["victoria"] and tipo == "derrotas"):
                count += 1
            else:
                break
        rachas_activas.append({"nombre": nombre, "tipo_racha": tipo, "longitud": count})

        # Mejor racha victorias
        max_v, cur_v = 0, 0
        for _, row in df_j.iterrows():
            if row["victoria"]:
                cur_v += 1
                max_v = max(max_v, cur_v)
            else:
                cur_v = 0
        rachas_max_v.append({"nombre": nombre, "max_racha_victorias": max_v})

        # Peor racha derrotas
        max_d, cur_d = 0, 0
        for _, row in df_j.iterrows():
            if not row["victoria"]:
                cur_d += 1
                max_d = max(max_d, cur_d)
            else:
                cur_d = 0
        rachas_max_d.append({"nombre": nombre, "max_racha_derrotas": max_d})

    return (
        pd.DataFrame(rachas_activas).sort_values("longitud", ascending=False),
        pd.DataFrame(rachas_max_v).sort_values("max_racha_victorias", ascending=False),
        pd.DataFrame(rachas_max_d).sort_values("max_racha_derrotas", ascending=False),
    )

@st.cache_data(ttl=10)
def calcular_rachas_historicas(df):
    rachas_victorias = []
    rachas_derrotas = []

    # Extraer el número real del id_partido (antes del "_") para ordenar correctamente
    df = df.copy()
    df["_id_partido_num"] = df["id_partido"].astype(str).str.split("_").str[0].astype(int)

    for nombre_jugador in df['nombre'].unique():
        df_jugador = df[df['nombre'] == nombre_jugador].sort_values(
            by=["id_jornada", "_id_partido_num"]  # ← orden correcto: jornada numérica + partido numérico
        )

        # 🔥 RACHAS DE VICTORIAS
        current_racha = 0
        current_partidos = []

        for _, row in df_jugador.iterrows():
            partido_limpio = str(row['id_partido']).split("_")[0]
            if row['victoria']:
                current_racha += 1
                current_partidos.append(partido_limpio)
            else:
                if current_racha > 0:
                    rachas_victorias.append({
                        "nombre": nombre_jugador,
                        "racha": current_racha,
                        "partidos": list(current_partidos)
                    })
                current_racha = 0
                current_partidos = []

        if current_racha > 0:
            rachas_victorias.append({
                "nombre": nombre_jugador,
                "racha": current_racha,
                "partidos": list(current_partidos)
            })

        # ❄️ RACHAS DE DERROTAS
        current_racha = 0
        current_partidos = []

        for _, row in df_jugador.iterrows():
            partido_limpio = str(row['id_partido']).split("_")[0]
            if not row['victoria']:
                current_racha += 1
                current_partidos.append(partido_limpio)
            else:
                if current_racha > 0:
                    rachas_derrotas.append({
                        "nombre": nombre_jugador,
                        "racha": current_racha,
                        "partidos": list(current_partidos)
                    })
                current_racha = 0
                current_partidos = []

        if current_racha > 0:
            rachas_derrotas.append({
                "nombre": nombre_jugador,
                "racha": current_racha,
                "partidos": list(current_partidos)
            })

    df_v = pd.DataFrame(rachas_victorias).sort_values(by="racha", ascending=False)
    df_d = pd.DataFrame(rachas_derrotas).sort_values(by="racha", ascending=False)

    return df_v, df_d

# @st.cache_data
# def construir_features_ml(df_hist):
#     """
#     Para cada partido histórico, calcula las features de ambos equipos
#     usando SOLO datos anteriores a ese partido (no data leakage).
#     Devuelve X (features) e y (quien ganó: 1 = equipo1, 0 = equipo2).
#     """
#     partidos_ordenados = df_hist.drop_duplicates("id_partido").copy()
#     partidos_ordenados["_id_num"] = (
#         partidos_ordenados["id_partido"].astype(str).str.split("_").str[0].astype(int)
#     )
#     partidos_ordenados = partidos_ordenados.sort_values(["id_jornada", "_id_num"])
 
#     filas = []
#     ids_vistos = []
 
#     for _, partido_row in partidos_ordenados.iterrows():
#         pid = partido_row["id_partido"]
#         jornada = partido_row["id_jornada"]
 
#         jugadores_partido = df_hist[df_hist["id_partido"] == pid]
#         eq1 = jugadores_partido[jugadores_partido["equipo"] == 1]["nombre"].tolist()
#         eq2 = jugadores_partido[jugadores_partido["equipo"] == 2]["nombre"].tolist()
 
#         if len(eq1) != 2 or len(eq2) != 2:
#             continue
 
#         # Datos ANTERIORES a este partido (excluimos el actual)
#         df_antes = df_hist[df_hist["id_partido"].isin(ids_vistos)]
 
#         def winrate(nombre):
#             sub = df_antes[df_antes["nombre"] == nombre]
#             if len(sub) == 0:
#                 return 0.5
#             return sub["victoria"].mean()
 
#         def forma_reciente(nombre, n=5):
#             sub = df_antes[df_antes["nombre"] == nombre].copy()
#             sub["_id_num"] = sub["id_partido"].astype(str).str.split("_").str[0].astype(int)
#             sub = sub.sort_values(["id_jornada", "_id_num"]).tail(n)
#             if len(sub) == 0:
#                 return 0.5
#             return sub["victoria"].mean()
 
#         def winrate_pareja(j1, j2):
#             sub1 = df_antes[df_antes["nombre"] == j1][["id_partido", "equipo", "victoria"]]
#             sub2 = df_antes[df_antes["nombre"] == j2][["id_partido", "equipo"]]
#             juntos = sub1.merge(sub2, on=["id_partido", "equipo"])
#             if len(juntos) == 0:
#                 return 0.5
#             return juntos["victoria"].mean()
 
#         def h2h(atacantes, defensores):
#             """% victorias de atacantes contra defensores en enfrentamientos directos"""
#             wins, total = 0, 0
#             for a in atacantes:
#                 for d in defensores:
#                     sub_a = df_antes[(df_antes["nombre"] == a) & (df_antes["equipo"] == 1)]
#                     sub_d = df_antes[(df_antes["nombre"] == d) & (df_antes["equipo"] == 2)]
#                     enf = sub_a.merge(sub_d, on="id_partido", suffixes=("_a", "_d"))
#                     wins += enf["victoria_a"].sum()
#                     total += len(enf)
 
#                     sub_a2 = df_antes[(df_antes["nombre"] == a) & (df_antes["equipo"] == 2)]
#                     sub_d2 = df_antes[(df_antes["nombre"] == d) & (df_antes["equipo"] == 1)]
#                     enf2 = sub_a2.merge(sub_d2, on="id_partido", suffixes=("_a", "_d"))
#                     wins += enf2["victoria_a"].sum()
#                     total += len(enf2)
 
#             return wins / total if total > 0 else 0.5
 
#         def diff_juegos_norm(nombre):
#             sub = df_antes[df_antes["nombre"] == nombre]
#             if len(sub) == 0:
#                 return 0.0
#             diff = (sub["juegos_ganados"].sum() - sub["juegos_perdidos"].sum())
#             total = sub["juegos_ganados"].sum() + sub["juegos_perdidos"].sum()
#             return diff / total if total > 0 else 0.0
 
#         # ── Calcular features ──
#         wr_eq1 = (winrate(eq1[0]) + winrate(eq1[1])) / 2
#         wr_eq2 = (winrate(eq2[0]) + winrate(eq2[1])) / 2
 
#         forma_eq1 = (forma_reciente(eq1[0]) + forma_reciente(eq1[1])) / 2
#         forma_eq2 = (forma_reciente(eq2[0]) + forma_reciente(eq2[1])) / 2
 
#         pareja_eq1 = winrate_pareja(eq1[0], eq1[1])
#         pareja_eq2 = winrate_pareja(eq2[0], eq2[1])
 
#         h2h_eq1 = h2h(eq1, eq2)
#         h2h_eq2 = h2h(eq2, eq1)
 
#         diff_eq1 = (diff_juegos_norm(eq1[0]) + diff_juegos_norm(eq1[1])) / 2
#         diff_eq2 = (diff_juegos_norm(eq2[0]) + diff_juegos_norm(eq2[1])) / 2
 
#         fila = {
#             # Diferencias entre equipos (el modelo aprende de estas diferencias)
#             "wr_diff":      wr_eq1 - wr_eq2,
#             "forma_diff":   forma_eq1 - forma_eq2,
#             "pareja_diff":  pareja_eq1 - pareja_eq2,
#             "h2h_diff":     h2h_eq1 - h2h_eq2,
#             "diff_juegos":  diff_eq1 - diff_eq2,
#             # Target
#             "y": 1 if partido_row["equipo_ganador"] == 1 else 0
#         }
#         filas.append(fila)
#         ids_vistos.append(pid)
 
#     df_features = pd.DataFrame(filas)
#     return df_features
 
 
# @st.cache_data
# def entrenar_modelo(df_features):
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import cross_val_score
#     from sklearn.pipeline import Pipeline
#     import numpy as np
 
#     feature_cols = ["wr_diff", "forma_diff", "pareja_diff", "h2h_diff", "diff_juegos"]
#     X = df_features[feature_cols].values
#     y = df_features["y"].values
 
#     pipeline = Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
#     ])
 
#     # Cross-validation para estimar accuracy real
#     scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
#     pipeline.fit(X, y)
 
#     coefs = pipeline.named_steps["clf"].coef_[0]
 
#     return pipeline, scores.mean(), scores.std(), feature_cols, coefs
 
 
# def predecir_partido(pipeline, df_hist, eq1, eq2, feature_cols):
#     """Calcula features con TODO el histórico disponible y predice."""
#     import numpy as np
 
#     def winrate(nombre):
#         sub = df_hist[df_hist["nombre"] == nombre]
#         return sub["victoria"].mean() if len(sub) > 0 else 0.5
 
#     def forma_reciente(nombre, n=5):
#         sub = df_hist[df_hist["nombre"] == nombre].copy()
#         sub["_id_num"] = sub["id_partido"].astype(str).str.split("_").str[0].astype(int)
#         sub = sub.sort_values(["id_jornada", "_id_num"]).tail(n)
#         return sub["victoria"].mean() if len(sub) > 0 else 0.5
 
#     def winrate_pareja(j1, j2):
#         sub1 = df_hist[df_hist["nombre"] == j1][["id_partido", "equipo", "victoria"]]
#         sub2 = df_hist[df_hist["nombre"] == j2][["id_partido", "equipo"]]
#         juntos = sub1.merge(sub2, on=["id_partido", "equipo"])
#         return juntos["victoria"].mean() if len(juntos) > 0 else 0.5
 
#     def h2h(atacantes, defensores):
#         wins, total = 0, 0
#         for a in atacantes:
#             for d in defensores:
#                 sub_a = df_hist[(df_hist["nombre"] == a) & (df_hist["equipo"] == 1)]
#                 sub_d = df_hist[(df_hist["nombre"] == d) & (df_hist["equipo"] == 2)]
#                 enf = sub_a.merge(sub_d, on="id_partido", suffixes=("_a", "_d"))
#                 wins += enf["victoria_a"].sum()
#                 total += len(enf)
#                 sub_a2 = df_hist[(df_hist["nombre"] == a) & (df_hist["equipo"] == 2)]
#                 sub_d2 = df_hist[(df_hist["nombre"] == d) & (df_hist["equipo"] == 1)]
#                 enf2 = sub_a2.merge(sub_d2, on="id_partido", suffixes=("_a", "_d"))
#                 wins += enf2["victoria_a"].sum()
#                 total += len(enf2)
#         return wins / total if total > 0 else 0.5
 
#     def diff_juegos_norm(nombre):
#         sub = df_hist[df_hist["nombre"] == nombre]
#         if len(sub) == 0:
#             return 0.0
#         diff = sub["juegos_ganados"].sum() - sub["juegos_perdidos"].sum()
#         total = sub["juegos_ganados"].sum() + sub["juegos_perdidos"].sum()
#         return diff / total if total > 0 else 0.0
 
#     wr_eq1 = (winrate(eq1[0]) + winrate(eq1[1])) / 2
#     wr_eq2 = (winrate(eq2[0]) + winrate(eq2[1])) / 2
#     forma_eq1 = (forma_reciente(eq1[0]) + forma_reciente(eq1[1])) / 2
#     forma_eq2 = (forma_reciente(eq2[0]) + forma_reciente(eq2[1])) / 2
#     pareja_eq1 = winrate_pareja(eq1[0], eq1[1])
#     pareja_eq2 = winrate_pareja(eq2[0], eq2[1])
#     h2h_eq1 = h2h(eq1, eq2)
#     h2h_eq2 = h2h(eq2, eq1)
#     diff_eq1 = (diff_juegos_norm(eq1[0]) + diff_juegos_norm(eq1[1])) / 2
#     diff_eq2 = (diff_juegos_norm(eq2[0]) + diff_juegos_norm(eq2[1])) / 2
 
#     X_pred = np.array([[
#         wr_eq1 - wr_eq2,
#         forma_eq1 - forma_eq2,
#         pareja_eq1 - pareja_eq2,
#         h2h_eq1 - h2h_eq2,
#         diff_eq1 - diff_eq2,
#     ]])
 
#     prob = pipeline.predict_proba(X_pred)[0]  # [prob_eq2_gana, prob_eq1_gana]
 
#     desglose = {
#         "Winrate histórico": (wr_eq1, wr_eq2),
#         "Forma reciente (5 partidos)": (forma_eq1, forma_eq2),
#         "Rendimiento como pareja": (pareja_eq1, pareja_eq2),
#         "Head-to-head directo": (h2h_eq1, h2h_eq2),
#         "Diferencia de juegos": (
#             (diff_eq1 + 1) / 2,  # normalizar a [0,1] para mostrar
#             (diff_eq2 + 1) / 2
#         ),
#     }
 
#     return prob[1], prob[0], desglose  # prob_eq1, prob_eq2, desglose


PESOS = {
    "H2H pareja exacta":         0.35,
    "H2H individual":            0.10,
    "Rendimiento como pareja":   0.32,
    "Forma reciente (5 partidos)":0.03,
    "Winrate histórico":         0.20,
    "Diferencia de juegos":      0.00,
}
 
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE FEATURES (reutilizables)
# ─────────────────────────────────────────────────────────────────────────────
 
def _winrate(df, nombre):
    sub = df[df["nombre"] == nombre]
    return sub["victoria"].mean() if len(sub) > 0 else 0.5
 
def _forma_reciente(df, nombre, n=5):
    sub = df[df["nombre"] == nombre].copy()
    sub["_n"] = sub["id_partido"].astype(str).str.split("_").str[0].astype(int)
    sub = sub.sort_values(["id_jornada", "_n"]).tail(n)
    return sub["victoria"].mean() if len(sub) > 0 else 0.5
 
def _winrate_pareja(df, j1, j2):
    s1 = df[df["nombre"] == j1][["id_partido", "equipo", "victoria"]]
    s2 = df[df["nombre"] == j2][["id_partido", "equipo"]]
    juntos = s1.merge(s2, on=["id_partido", "equipo"])
    return juntos["victoria"].mean() if len(juntos) > 0 else 0.5
 
def _h2h_pareja_exacta(df, eq1, eq2):
    """
    Busca partidos donde eq1[0]+eq1[1] jugaron JUNTOS contra eq2[0]+eq2[1].
    Prueba ambas combinaciones de equipos (1vs2 y 2vs1).
    Devuelve (winrate_eq1, n_partidos).
    """
    wins, total = 0, 0
 
    for e1, e2 in [(1, 2), (2, 1)]:
        p_j1a = set(df[(df["nombre"] == eq1[0]) & (df["equipo"] == e1)]["id_partido"])
        p_j1b = set(df[(df["nombre"] == eq1[1]) & (df["equipo"] == e1)]["id_partido"])
        p_j2a = set(df[(df["nombre"] == eq2[0]) & (df["equipo"] == e2)]["id_partido"])
        p_j2b = set(df[(df["nombre"] == eq2[1]) & (df["equipo"] == e2)]["id_partido"])
 
        partidos_comunes = p_j1a & p_j1b & p_j2a & p_j2b
 
        for pid in partidos_comunes:
            fila = df[df["id_partido"] == pid].iloc[0]
            gano_eq1 = (fila["equipo_ganador"] == e1)
            wins += int(gano_eq1)
            total += 1
 
    if total == 0:
        return 0.5, 0
    return wins / total, total
 
 
def _h2h_individual(df, eq1, eq2):
    """
    Fallback: promedia enfrentamientos individuales cruzados.
    ej: Gonzalo vs Esteban + Gonzalo vs Ivan + Danisu vs Esteban + Danisu vs Ivan
    """
    wins, total = 0, 0
    for a in eq1:
        for d in eq2:
            for e_a, e_d in [(1, 2), (2, 1)]:
                sa = df[(df["nombre"] == a) & (df["equipo"] == e_a)]
                sd = df[(df["nombre"] == d) & (df["equipo"] == e_d)]
                enf = sa.merge(sd, on="id_partido", suffixes=("_a", "_d"))
                wins += enf["victoria_a"].sum()
                total += len(enf)
    return wins / total if total > 0 else 0.5
 
def _diff_juegos_norm(df, nombre):
    sub = df[df["nombre"] == nombre]
    if len(sub) == 0:
        return 0.5
    g = sub["juegos_ganados"].sum()
    p = sub["juegos_perdidos"].sum()
    total = g + p
    return (g / total) if total > 0 else 0.5

def _peso_h2h_dinamico(n_partidos, peso_base):
    """
    Ajusta el peso del H2H pareja exacta según número de partidos.
    """
    if n_partidos == 0:
        return 0.0
    elif n_partidos == 1:
        return peso_base * 0.35
    elif n_partidos < 3:
        return peso_base * 0.8
    else:
        return peso_base * 1.5
 
def calcular_score(df_ref, eq1, eq2):
    vals = {}

    # 1. Obtener datos de H2H Pareja Exacta
    exacto_eq1, n = _h2h_pareja_exacta(df_ref, eq1, eq2)
    exacto_eq2, _ = _h2h_pareja_exacta(df_ref, eq2, eq1)

    # Peso dinámico H2H pareja exacta
    peso_base_h2h = PESOS.get("H2H pareja exacta", 0.0)
    peso_h2h_real = _peso_h2h_dinamico(n, peso_base_h2h)

    # Solo añadimos si hay datos suficientes
    if peso_h2h_real > 0:
        vals["H2H pareja exacta"] = (exacto_eq1, exacto_eq2)

    # 2. Recopilar resto de métricas
    vals["H2H individual"] = (
        _h2h_individual(df_ref, eq1, eq2),
        _h2h_individual(df_ref, eq2, eq1)
    )

    vals["Rendimiento como pareja"] = (
        _winrate_pareja(df_ref, eq1[0], eq1[1]),
        _winrate_pareja(df_ref, eq2[0], eq2[1])
    )

    vals["Forma reciente (5 partidos)"] = (
        (_forma_reciente(df_ref, eq1[0]) + _forma_reciente(df_ref, eq1[1])) / 2,
        (_forma_reciente(df_ref, eq2[0]) + _forma_reciente(df_ref, eq2[1])) / 2
    )

    vals["Winrate histórico"] = (
        (_winrate(df_ref, eq1[0]) + _winrate(df_ref, eq1[1])) / 2,
        (_winrate(df_ref, eq2[0]) + _winrate(df_ref, eq2[1])) / 2
    )

    vals["Diferencia de juegos"] = (
        (_diff_juegos_norm(df_ref, eq1[0]) + _diff_juegos_norm(df_ref, eq1[1])) / 2,
        (_diff_juegos_norm(df_ref, eq2[0]) + _diff_juegos_norm(df_ref, eq2[1])) / 2
    )

    # 🔢 CÁLCULO FINAL
    score_eq1, score_eq2 = 0.0, 0.0

    for feature, (v1, v2) in vals.items():
        # Definir el peso de la característica
        if feature == "H2H pareja exacta":
            peso = peso_h2h_real
        elif feature == "Winrate histórico" and n >= 3:
            # 📉 Si ya han jugado 3+ partidos juntos, el pasado individual importa menos
            peso = PESOS.get(feature, 0.0) * 0.3
        else:
            peso = PESOS.get(feature, 0.0)

        # Proporción de la métrica
        total = v1 + v2
        p1, p2 = (v1 / total, v2 / total) if total > 0 else (0.5, 0.5)

        # 🛡️ Lógica Anti-Dilución (Penalización por 0% en pareja)
        # Si una pareja tiene 0 victorias en H2H o Rendimiento tras varios partidos, se castiga el score
        if feature in ["Rendimiento como pareja", "H2H pareja exacta"]:
            # Si el equipo 1 tiene 0 victorias pero el equipo 2 sí tiene
            if v1 == 0 and v2 > 0:
                p1 = p1 * 0.15  # El 0% ahora es una losa pesada
                p2 = 1 - p1
            elif v2 == 0 and v1 > 0:
                p2 = p2 * 0.15
                p1 = 1 - p2

        score_eq1 += peso * p1
        score_eq2 += peso * p2

    total_score = score_eq1 + score_eq2

    # Protección extra
    if total_score == 0:
        return 0.5, 0.5, vals

    score_eq1 /= total_score
    score_eq2 /= total_score
    
    # 🏁 Hachazo final: Suelo de seguridad
    # Si una pareja ha perdido los 3 o más partidos jugados contra la otra, 
    # no permitimos que el azar de otras stats les de mucha esperanza.
    if n >= 3:
        if exacto_eq1 == 0 and exacto_eq2 > 0:
            score_eq1 = min(score_eq1, 0.20)
            score_eq2 = 1 - score_eq1
        elif exacto_eq2 == 0 and exacto_eq1 > 0:
            score_eq2 = min(score_eq2, 0.20)
            score_eq1 = 1 - score_eq2

    return score_eq1, score_eq2, vals
 
 
# @st.cache_data
# def validacion_historica(df_hist):
#     """
#     Aplica el modelo a cada partido histórico usando solo datos anteriores.
#     Devuelve (accuracy, n_partidos_validados).
#     """
#     partidos_ord = df_hist.drop_duplicates("id_partido").copy()
#     partidos_ord["_n"] = partidos_ord["id_partido"].astype(str).str.split("_").str[0].astype(int)
#     partidos_ord = partidos_ord.sort_values(["id_jornada", "_n"])
 
#     correctos, total, ids_vistos = 0, 0, []
 
#     for _, row in partidos_ord.iterrows():
#         pid = row["id_partido"]
#         jugadores_p = df_hist[df_hist["id_partido"] == pid]
#         eq1 = jugadores_p[jugadores_p["equipo"] == 1]["nombre"].tolist()
#         eq2 = jugadores_p[jugadores_p["equipo"] == 2]["nombre"].tolist()
 
#         if len(eq1) != 2 or len(eq2) != 2:
#             ids_vistos.append(pid)
#             continue
 
#         df_antes = df_hist[df_hist["id_partido"].isin(ids_vistos)]
 
#         if len(df_antes) < 8:
#             ids_vistos.append(pid)
#             continue
 
#         s1, s2, _ = calcular_score(df_antes, eq1, eq2)
#         prediccion = 1 if s1 >= s2 else 2
#         if prediccion == row["equipo_ganador"]:
#             correctos += 1
#         total += 1
#         ids_vistos.append(pid)
 
#     return (correctos / total if total > 0 else 0.0), total

# ─────────────────────────────────────────────
# SIDEBAR (VERSIÓN OPTIMIZADA)
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <p class="main-title">🎾 PÁDEL GUIMANETA</p>
    <p class="sub-title">LIGA DE PÁDEL - RESULTADOS</p>
    """, unsafe_allow_html=True)
    st.divider()

    try:
        # Carga de datos
        jugadores, partidos, partido_jugadores = cargar_datos()
        df_completo = construir_df(jugadores, partidos, partido_jugadores)

        # Obtener temporadas reales (SIN "Todas")
        if 'temporada' in df_completo.columns:
            temporadas = sorted(df_completo["temporada"].dropna().unique())
        else:
            st.error("❌ No existe la columna 'temporada'")
            st.stop()

        # Selector de temporada
        temporada_sel = st.selectbox("📅 Temporada", temporadas, index=len(temporadas)-1)

        # Filtrado directo (SIEMPRE una temporada)
        df = df_completo[df_completo["temporada"] == temporada_sel].copy()

        # Seguridad extra (por si algo raro pasa)
        if df.empty:
            st.warning("⚠️ No hay datos para esta temporada")
            st.stop()

    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.stop()

    st.divider()

    # Navegación
    st.markdown("**🧭 Navegación**")
    seccion = st.radio(
        label="",
        options=[
            "🏆 Clasificación",
            "📋 Detalle",
            "👤 Perfil Jugador",
            "⚔️ Enfrentamientos",
            "🤝 Parejas",
            "🔥 Rachas",
            "📊 Gráficas",
            "💻 Predictor",
            "🧮 Calculadora",
            "🔍 Buscador",
            "🔐 Admin"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # Métricas rápidas
    col1, col2 = st.columns(2)
    col1.metric("📊 Partidos", df["id_partido"].nunique())
    col2.metric("🗓️ Jornadas", df["id_jornada"].nunique())

# ─────────────────────────────────────────────
# CÁLCULOS GLOBALES
# ─────────────────────────────────────────────
clasificacion = calcular_clasificacion(df)
ranking_jornada = calcular_ranking_por_jornada(df)
df_enf = calcular_enfrentamientos(df)
df_parejas = calcular_parejas(df)
rachas_activas_df, rachas_max_v_df, rachas_max_d_df = calcular_rachas(df)
df_rachas_v, df_rachas_d = calcular_rachas_historicas(df)
ranking_partido = calcular_ranking_por_partido(df)
nombres = ordenar_nombres(df["nombre"].unique())
mapa_colores = crear_mapa_colores(nombres)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
if seccion == "🏆 Clasificación":
    st.markdown(f"## 🎾 PÁDEL GUIMANETA {temporada_sel}")

    # Métricas top
    lider = clasificacion.iloc[0]

    # Valores
    num_partidos = df["id_partido"].nunique()
    num_jornadas = df["id_jornada"].nunique()
    num_jugadores = len(clasificacion)

    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid #30363d; border-radius:12px; padding:16px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; gap:16px;">
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">🥇 Líder</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{lider['nombre']}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">📊 Partidos jugados</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{num_partidos}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">🗓️ Jornadas</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{num_jornadas}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">👥 Jugadores</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{num_jugadores}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TERMÓMETRO: TENDENCIAS Y RACHAS ACTUALES

    # 1. CÁLCULO DE TENDENCIAS (Subidón y Bajón de %)
    jor_actual = ranking_jornada["hasta_jornada"].max()
    jor_previa = ranking_jornada[ranking_jornada["hasta_jornada"] < jor_actual]["hasta_jornada"].max()

    subidon_txt, bajon_txt = "N/A", "N/A"

    if pd.notna(jor_previa):
        stats_ult = ranking_jornada[ranking_jornada["hasta_jornada"] == jor_actual][["nombre", "porcentaje_victorias"]]
        stats_pen = ranking_jornada[ranking_jornada["hasta_jornada"] == jor_previa][["nombre", "porcentaje_victorias"]]
        
        tendencia = stats_ult.merge(stats_pen, on="nombre", suffixes=('_ult', '_pen'))
        tendencia["dif"] = tendencia["porcentaje_victorias_ult"] - tendencia["porcentaje_victorias_pen"]
        
        s = tendencia.sort_values("dif", ascending=False).iloc[0]
        b = tendencia.sort_values("dif", ascending=True).iloc[0]
        
        subidon_txt = f"{s['nombre']} (+{s['dif']:.1f}%)"
        bajon_txt = f"{b['nombre']} ({b['dif']:.1f}%)"

    # 2. CÁLCULO DE RACHAS ACTUALES (Usando rachas_activas_df)
    racha_v = rachas_activas_df[rachas_activas_df["tipo_racha"] == "victorias"]
    racha_d = rachas_activas_df[rachas_activas_df["tipo_racha"] == "derrotas"]

    top_v_nom = racha_v.iloc[0]["nombre"] if not racha_v.empty else "-"
    top_v_val = racha_v.iloc[0]["longitud"] if not racha_v.empty else 0

    top_d_nom = racha_d.iloc[0]["nombre"] if not racha_d.empty else "-"
    top_d_val = racha_d.iloc[0]["longitud"] if not racha_d.empty else 0

    # 3. RENDERIZADO HTML
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px;">
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px; border-left: 5px solid #238636;">
            <div style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">📈 El Subidón</div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 700; margin-top: 4px;">{subidon_txt}</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px; border-left: 5px solid #da3633;">
            <div style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">📉 El Bajón</div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 700; margin-top: 4px;">{bajon_txt}</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px; border-left: 5px solid #f1e05a;">
            <div style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">🔥 On Fire</div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 700; margin-top: 4px;">{top_v_nom}</div>
            <div style="color: #f1e05a; font-size: 0.85rem; font-weight: 600;">{top_v_val} victorias seguidas</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px; border-left: 5px solid #8b949e;">
            <div style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">🧊 En el pozo</div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 700; margin-top: 4px;">{top_d_nom}</div>
            <div style="color: #8b949e; font-size: 0.85rem; font-weight: 600;">{top_d_val} derrotas seguidas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("## 🏆 Clasificación General")

    # Tabla de clasificación estilizada
    tabla = clasificacion[[
        "nombre", "partidos_jugados", "victorias", "derrotas",
        "diferencia_juegos", "juegos_ganados", "juegos_perdidos",
        "porcentaje_victorias", "jornadas"
    ]].copy()

    tabla.columns = ["Jugador", "PJ", "V", "D", "+/-", "JG", "JP", "% V", "Jornadas"]

    # FUNCIONES DE COLOR SUAVE TIPO GRADIENTE
    def color_top(nombre, rank):
        if rank == 1:  # 🥇
            return "background-color: rgba(255, 215, 0, 0.15); color: #FFD700; font-weight: 600;"
        elif rank == 2:  # 🥈
            return "background-color: rgba(192, 192, 192, 0.12); color: #C0C0C0; font-weight: 600;"
        elif rank == 3:  # 🥉
            return "background-color: rgba(205, 127, 50, 0.12); color: #CD7F32; font-weight: 600;"
        return ""

    # Aplicar estilo correctamente usando el índice REAL de la tabla
    def estilo_fila(row):
        rank = row.name  # posición real en el DataFrame ya ordenado
        return [color_top(row["Jugador"], rank)] * len(row)

    st.dataframe(
        tabla.style
        .apply(estilo_fila, axis=1)
        .background_gradient(subset=["% V"], cmap="RdYlGn", vmin=0, vmax=100)
        .format({"% V": "{:.2f}%"}),
        use_container_width=True,
        height=320
    )

    st.divider()
    st.markdown("## 📈 Evolución del Ranking")

    # Toggle jornada / partido
    vista_ranking = st.radio(
        "Ver evolución por:",
        ["🗓️ Jornada", "🎾 Partido"],
        horizontal=True,
        key="vista_ranking"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    colores = plt.cm.tab10.colors

    if vista_ranking == "🗓️ Jornada":
        for nombre in nombres:
            datos = ranking_jornada[
                ranking_jornada["nombre"] == nombre
            ].sort_values("hasta_jornada")

            ax.plot(
                datos["hasta_jornada"],
                datos["rank"],
                marker="o",
                linewidth=2.5,
                markersize=6,
                color=mapa_colores[nombre],
                label=nombre
            )

        ax.set_xlabel("Jornada", color="#8b949e")
        ax.set_xticks(sorted(ranking_jornada["hasta_jornada"].unique()))
        n_jugadores = ranking_jornada["rank"].max()

    else:
        for nombre in nombres:
            datos = ranking_partido[
                ranking_partido["nombre"] == nombre
            ].sort_values("hasta_partido")

            ax.plot(
                datos["hasta_partido"],
                datos["rank"],
                marker="o",
                linewidth=2,
                markersize=4,
                color=mapa_colores[nombre],
                label=nombre
            )

        # Líneas verticales separando jornadas
        jornada_cambios = ranking_partido.drop_duplicates("id_jornada").sort_values("hasta_partido")
        for _, jrow in jornada_cambios.iterrows():
            ax.axvline(x=jrow["hasta_partido"] - 0.5, color="#30363d", linewidth=1.6, linestyle="--")
            ax.text(jrow["hasta_partido"] - 0.5, 0.3, f"J{int(jrow['id_jornada'])}",
                    color="#8b949e", fontsize=7, ha="center")

        ax.set_xlabel("Nº partido acumulado", color="#8b949e")
        ax.set_xticks(sorted(ranking_partido["hasta_partido"].unique()))
        ax.tick_params(axis='x', labelsize=7)
        n_jugadores = ranking_partido["rank"].max()

    ax.invert_yaxis()
    ax.set_ylabel("Posición", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.set_yticks(range(1, int(n_jugadores) + 1))
    ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: DETALLE
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "📋 Detalle":
    st.markdown("## 📋 Detalle de Jornada")

    # 🔽 Selección de jornada
    jornadas = sorted(df["id_jornada"].unique())
    jornada_sel = st.select_slider(
        "Selecciona jornada",
        options=jornadas,
        value=jornadas[-1]  # valor inicial
    )

    # Filtrar datos de la jornada
    df_jornada_simple = df[df["id_jornada"] == jornada_sel].copy()

    # Tomamos una fila representativa
    fila_info = df_jornada_simple.drop_duplicates("id_partido").iloc[0]

    # --- LÓGICA DE FECHA FORMATEADA ---
    fecha_raw = fila_info["fecha"] if "fecha" in fila_info else None
    fecha_formateada = "N/A"

    if fecha_raw and fecha_raw != "N/A":
        try:
            # Intentamos convertir el string (asumiendo formato YYYY-MM-DD)
            fecha_dt = pd.to_datetime(fecha_raw)
            # Formato: día, mes abreviado en minúsculas (o como prefieras) y año
            fecha_formateada = fecha_dt.strftime("%d %b %Y").lower() 
        except:
            fecha_formateada = fecha_raw

    sede = fila_info["sede"] if "sede" in fila_info else "N/A"
    num_partidos = df_jornada_simple["id_partido"].nunique()

    # ── Contenedor estilizado ──
    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid #30363d; border-radius:12px; padding:16px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; gap:16px;">
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">🗓️ Jornada</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{jornada_sel}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">📍 Sede</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{sede}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">📅 Fecha</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{fecha_formateada}</div>
            </div>
            <div style="flex:1; text-align:center;">
                <div style="color:#8b949e; font-size:0.8rem;">🎾 Partidos</div>
                <div style="color:#ffffff; font-size:1.2rem; font-weight:700;">{num_partidos}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 🏆 CLASIFICACIÓN DE ESA JORNADA (SOLO ESA JORNADA)
    st.markdown("### 🏆 Clasificación de la jornada")

    df_jornada_simple = df[df["id_jornada"] == jornada_sel].copy()

    clasif_simple = calcular_clasificacion(df_jornada_simple)

    tabla_simple = clasif_simple[[
        "nombre", "partidos_jugados", "victorias", "derrotas",
        "diferencia_juegos", "juegos_ganados", "juegos_perdidos",
        "porcentaje_victorias"
    ]].copy()

    tabla_simple.columns = ["Jugador", "PJ", "V", "D", "+/-", "JG", "JP", "% V"]

    st.dataframe(
        tabla_simple.style
        .background_gradient(subset=["% V"], cmap="RdYlGn", vmin=0, vmax=100)
        .format({"% V": "{:.2f}%"}),
        use_container_width=True
    )

    st.divider()

    # 🏆 CLASIFICACIÓN ACUMULADA HASTA ESA JORNADA
    st.markdown("### 🏆 Clasificación acumulada")

    df_jornada_acum = df[df["id_jornada"] <= jornada_sel].copy()

    clasif_acum = calcular_clasificacion(df_jornada_acum)

    tabla_acum = clasif_acum[[
        "nombre", "partidos_jugados", "victorias", "derrotas",
        "diferencia_juegos", "juegos_ganados", "juegos_perdidos",
        "porcentaje_victorias"
    ]].copy()

    tabla_acum.columns = ["Jugador", "PJ", "V", "D", "+/-", "JG", "JP", "% V"]

    st.dataframe(
        tabla_acum.style
        .background_gradient(subset=["% V"], cmap="RdYlGn", vmin=0, vmax=100)
        .format({"% V": "{:.2f}%"}),
        use_container_width=True
    )

    st.divider()

    # ⚔️ PARTIDOS DE ESA JORNADA
    st.markdown("### ⚔️ Partidos de la jornada")

    partidos_jornada = df[
        (df["id_jornada"] == jornada_sel)
    ].copy()

    st.write(f"Partidos encontrados: {partidos_jornada['id_partido'].nunique()}")

    # ORDENAR PARTIDOS POR NÚMERO REAL
    partidos_jornada["_num"] = partidos_jornada["id_partido"].astype(str).str.split("_").str[0].astype(int)
    partidos_jornada = partidos_jornada.sort_values("_num")

    # Agrupar por partido (ya ordenado)
    for partido_id, grupo in partidos_jornada.groupby("id_partido", sort=False):

        # 🔢 Número limpio del partido
        num_partido = str(partido_id).split("_")[0]

        # Equipos
        equipo1 = grupo[grupo["equipo"] == 1]["nombre"].tolist()
        equipo2 = grupo[grupo["equipo"] == 2]["nombre"].tolist()

        if len(equipo1) < 1 or len(equipo2) < 1:
            continue

        eq1 = " y ".join(equipo1)
        eq2 = " y ".join(equipo2)

        # Juegos
        p_row = grupo.iloc[0]
        g1 = int(p_row["juegos_equipo1"])
        g2 = int(p_row["juegos_equipo2"])

        ganador = "equipo1" if p_row["equipo_ganador"] == 1 else "equipo2"

        if ganador == "equipo1":
            texto_ganador = f"{eq1} ganan el partido"
        else:
            texto_ganador = f"{eq2} ganan el partido"

        st.markdown(
            f"**{num_partido}. {eq1} vs {eq2}: {g1}-{g2}**  \n"
            f"_{texto_ganador}_"
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: PERFIL JUGADOR
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "👤 Perfil Jugador":
    st.markdown("## 👤 Perfil de Jugador")
    nombre_sel = st.selectbox(
        "Selecciona jugador",
        nombres
    )

    fila = clasificacion[clasificacion["nombre"] == nombre_sel].iloc[0]
    pos = clasificacion.index[clasificacion["nombre"] == nombre_sel][0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📍 Posición", f"#{pos}")
    col2.metric("🎾 Partidos", fila["partidos_jugados"])
    col3.metric("✅ Victorias", int(fila["victorias"]))
    col4.metric("❌ Derrotas", int(fila["derrotas"]))
    col5.metric("📈 % Victorias", f"{fila['porcentaje_victorias']}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Evolución del ranking")

        modo_rank = st.radio(
            "Tipo de ranking",
            ["🗓️ Jornada", "⚡ Partido"],
            horizontal=True
        )

        if modo_rank == "🗓️ Jornada":
            datos_jugador = ranking_jornada[
                ranking_jornada["nombre"] == nombre_sel
            ].sort_values("hasta_jornada")

            x = datos_jugador["hasta_jornada"]
            y = datos_jugador["rank"]
            xlabel = "Jornada"

        else:
            datos_jugador = ranking_partido[
                ranking_partido["nombre"] == nombre_sel
            ].sort_values("hasta_partido")

            x = datos_jugador["hasta_partido"]
            y = datos_jugador["rank"]
            xlabel = "Partido"

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        ax.plot(
            x, y,
            marker="o",
            linewidth=2,
            markersize=3,
            color="#238636"
        )

        ax.fill_between(x, y, alpha=0.15, color="#238636")

        ax.invert_yaxis()

        ax.set_xlabel(xlabel, color="#8b949e")
        ax.set_ylabel("Posición", color="#8b949e")

        if modo_rank == "🗓️ Jornada":
            ax.set_xticks(sorted(x.unique()))
        else:
            ax.set_xticks([])
        ax.set_yticks(range(1, ranking_jornada["rank"].max() + 1))

        ax.tick_params(colors="#8b949e")
        ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Victorias vs Derrotas")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        vals = [int(fila["victorias"]), int(fila["derrotas"])]
        labels = ["Victorias", "Derrotas"]
        colors = ["#238636", "#da3633"]
        wedges, texts, autotexts = ax.pie(
            vals, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#e6edf3"},
            wedgeprops={"linewidth": 2, "edgecolor": "#0d1117"}
        )
        for at in autotexts:
            at.set_fontsize(13)
            at.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("#### ⚔️ Enfrentamientos directos")

    if not df_enf.empty:
        rows = df_enf[(df_enf["jugador1"] == nombre_sel) | (df_enf["jugador2"] == nombre_sel)].copy()
        resultado = []

        for _, r in rows.iterrows():
            if r["jugador1"] == nombre_sel:
                rival = r["jugador2"]
                v_yo, v_rival = r["victorias_jugador1"], r["victorias_jugador2"]
                jg_yo, jg_rival = r["juegos_ganados_jugador1"], r["juegos_ganados_jugador2"]
                jp_yo, jp_rival = r["juegos_ganados_jugador2"], r["juegos_ganados_jugador1"]
                pct = r["pct_j1"]
            else:
                rival = r["jugador1"]
                v_yo, v_rival = r["victorias_jugador2"], r["victorias_jugador1"]
                jg_yo, jg_rival = r["juegos_ganados_jugador2"], r["juegos_ganados_jugador1"]
                jp_yo, jp_rival = r["juegos_ganados_jugador1"], r["juegos_ganados_jugador2"]
                pct = r["pct_j2"]

            resultado.append({
                "Rival": rival,
                "Mis victorias": int(v_yo),
                "Sus victorias": int(v_rival),
                "Partidos": int(r["partidos_totales"]),
                "Juegos a favor": int(jg_yo),
                "Juegos en contra": int(jp_yo),
                "Mi %": pct
            })

        df_res = pd.DataFrame(resultado).sort_values("Mi %", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_res.style.background_gradient(subset=["Mi %"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True
        )

    st.markdown("#### 🤝 Rendimiento con parejas")

    if not df_parejas.empty:
        rows_p = df_parejas[(df_parejas["jugador1"] == nombre_sel) | (df_parejas["jugador2"] == nombre_sel)].copy()
        resultado_p = []

        for _, r in rows_p.iterrows():
            socio = r["jugador2"] if r["jugador1"] == nombre_sel else r["jugador1"]

            resultado_p.append({
                "Pareja": socio,
                "Partidos": int(r["partidos_juntos"]),
                "Victorias": int(r["victorias_juntos"]),
                "Derrotas": int(r["derrotas_juntos"]),
                "Juegos a favor": int(r["juegos_ganados_juntos"]),
                "Juegos en contra": int(r["juegos_perdidos_juntos"]),
                "% Victorias": r["porcentaje_victorias"]
            })

        df_rp = pd.DataFrame(resultado_p).sort_values("% Victorias", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_rp.style.background_gradient(subset=["% Victorias"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: ENFRENTAMIENTOS
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "⚔️ Enfrentamientos":
    st.markdown("## ⚔️ Enfrentamientos Directos")
 
    tab1, tab2 = st.tabs(["🔍 Buscar enfrentamiento", "🗺️ Heatmap completo"])
 
    with tab1:
        col1, col2 = st.columns(2)

        j1 = col1.selectbox(
            "Jugador 1",
            nombres,
            key="enf_j1"
        )

        j2 = col2.selectbox(
            "Jugador 2",
            [j for j in nombres if j != j1],
            key="enf_j2"
        )
 
        if not df_enf.empty:
            fila = df_enf[
                ((df_enf["jugador1"] == j1) & (df_enf["jugador2"] == j2)) |
                ((df_enf["jugador1"] == j2) & (df_enf["jugador2"] == j1))
            ]
            if not fila.empty:
                r = fila.iloc[0]
                if r["jugador1"] == j1:
                    v1, v2 = int(r["victorias_jugador1"]), int(r["victorias_jugador2"])
                    jg1, jg2 = int(r["juegos_ganados_jugador1"]), int(r["juegos_ganados_jugador2"])
                    p1_name, p2_name = j1, j2
                else:
                    v1, v2 = int(r["victorias_jugador2"]), int(r["victorias_jugador1"])
                    jg1, jg2 = int(r["juegos_ganados_jugador2"]), int(r["juegos_ganados_jugador1"])
                    p1_name, p2_name = j1, j2
 
                total = int(r["partidos_totales"])

                pct1 = v1 / total * 100 if total > 0 else 0
                pct2 = v2 / total * 100 if total > 0 else 0

                color1 = "#238636" if pct1 > 50 else "#da3633"
                color2 = "#238636" if pct2 > 50 else "#da3633"

                st.divider()
                c1, c2, c3 = st.columns([2, 1, 2])

                c1.markdown(f"### {p1_name}")
                c1.metric("Victorias", v1)
                c1.markdown(f"<span style='color:{color1}; font-size:18px; font-weight:600;'>{pct1:.1f}%</span>", unsafe_allow_html=True)
                c1.metric("Juegos ganados", jg1)

                c2.markdown("### VS", unsafe_allow_html=False)
                c2.metric("Partidos", total)

                c3.markdown(f"### {p2_name}")
                c3.metric("Victorias", v2)
                c3.markdown(f"<span style='color:{color2}; font-size:18px; font-weight:600;'>{pct2:.1f}%</span>", unsafe_allow_html=True)
                c3.metric("Juegos ganados", jg2)

            else:
                st.info("Estos jugadores no se han enfrentado todavía.")
 
        st.divider()
        st.markdown("#### 📋 Tabla completa de enfrentamientos")
        if not df_enf.empty:
            
            df_enf_sorted = df_enf.copy()

            # Asegurar que jugador1 es el de mejor %
            cond = df_enf_sorted["pct_j2"] > df_enf_sorted["pct_j1"]

            cols_swap = [
                ("jugador1", "jugador2"),
                ("victorias_jugador1", "victorias_jugador2"),
                ("juegos_ganados_jugador1", "juegos_ganados_jugador2"),
                ("pct_j1", "pct_j2"),
            ]

            for col1, col2 in cols_swap:
                df_enf_sorted.loc[cond, [col1, col2]] = df_enf_sorted.loc[cond, [col2, col1]].values

            # Orden por enfrentamientos más igualados
            df_enf_sorted["pct_diff"] = abs(
                df_enf_sorted["pct_j1"] - df_enf_sorted["pct_j2"]
            )

            df_enf_sorted = df_enf_sorted.sort_values(by="pct_diff", ascending=False)
            df_enf_sorted = df_enf_sorted.drop(columns=["pct_diff"])

            tabla_enf = df_enf_sorted[[
                "jugador1", "jugador2", "partidos_totales",
                "victorias_jugador1", "victorias_jugador2",
                "pct_j1", "pct_j2"
            ]].copy()

            tabla_enf.columns = [
                "Jugador 1", "Jugador 2", "Partidos",
                "V Jugador 1", "V Jugador 2", "% J1", "% J2"
            ]

            st.dataframe(
                tabla_enf,
                use_container_width=True,
                height=770
            )
 
    with tab2:
        if not df_enf.empty:
            jugadores_unicos = nombres
            matriz = pd.DataFrame(0.0, index=jugadores_unicos, columns=jugadores_unicos)
            for _, r in df_enf.iterrows():
                j1n, j2n = r["jugador1"], r["jugador2"]
                if j1n in matriz.index and j2n in matriz.columns:
                    matriz.loc[j1n, j2n] = r["victorias_jugador1"]
                    matriz.loc[j2n, j1n] = r["victorias_jugador2"]
            np.fill_diagonal(matriz.values, np.nan)
 
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")
            sns.heatmap(matriz, annot=True, fmt=".0f", cmap="RdYlGn",
                        ax=ax, linewidths=0.5, linecolor="#0d1117",
                        annot_kws={"size": 12, "color": "white"})
            ax.set_title("Victorias (fila vs columna)", color="#e6edf3", pad=15)
            ax.tick_params(colors="#e6edf3")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: PAREJAS
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "🤝 Parejas":
    st.markdown("## 🤝 Rendimiento Jugando Juntos")

    if not df_parejas.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            df_parejas_sorted = df_parejas.copy()

            # criterio de desempate
            df_parejas_sorted["diff_juegos"] = (
                df_parejas_sorted["juegos_ganados_juntos"] - df_parejas_sorted["juegos_perdidos_juntos"]
            )

            df_parejas_sorted = df_parejas_sorted.sort_values(
                by=["porcentaje_victorias", "diff_juegos", "partidos_juntos"],
                ascending=[False, False, False]
            )

            tabla_parejas = df_parejas_sorted[[
                "jugador1", "jugador2", "partidos_juntos",
                "victorias_juntos", "derrotas_juntos",
                "juegos_ganados_juntos", "juegos_perdidos_juntos",
                "porcentaje_victorias"
            ]].copy()

            tabla_parejas.columns = ["J1", "J2", "PJ", "V", "D", "JG", "JP", "% V"]

            st.dataframe(
                tabla_parejas.style.background_gradient(
                    subset=["% V"], cmap="RdYlGn", vmin=0, vmax=100
                ),
                use_container_width=True,
                height=770
            )

        with col2:
            st.markdown("#### 🏅 Mejor pareja")

            mejor = df_parejas[df_parejas["partidos_juntos"] >= 3].copy()

            # criterio de desempate
            mejor["diff_juegos"] = (
                mejor["juegos_ganados_juntos"] - mejor["juegos_perdidos_juntos"]
            )

            mejor = mejor.sort_values(
                by=["porcentaje_victorias", "diff_juegos", "partidos_juntos"],
                ascending=[False, False, False]
            )

            if not mejor.empty:
                top = mejor.iloc[0]
                st.success(
                    f"**{top['jugador1']} & {top['jugador2']}**\n\n"
                    f"🏆 {top['porcentaje_victorias']}% victorias\n\n"
                    f"📊 {int(top['partidos_juntos'])} partidos juntos"
                )

            st.markdown("#### 💔 Peor pareja")

            peor = df_parejas[df_parejas["partidos_juntos"] >= 3].copy()

            # mismo criterio de desempate
            peor["diff_juegos"] = (
                peor["juegos_ganados_juntos"] - peor["juegos_perdidos_juntos"]
            )

            peor = peor.sort_values(
                by=["porcentaje_victorias", "diff_juegos", "partidos_juntos"],
                ascending=[True, True, True]
            )

            if not peor.empty:
                bot = peor.iloc[0]
                st.error(
                    f"**{bot['jugador1']} & {bot['jugador2']}**\n\n"
                    f"📉 {bot['porcentaje_victorias']}% victorias\n\n"
                    f"📊 {int(bot['partidos_juntos'])} partidos juntos"
                )

    else:
        st.info("No hay datos de parejas disponibles.")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: RACHAS
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "🔥 Rachas":
    st.markdown("## 🔥 Rachas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ⚡ Racha activa")
        for _, r in rachas_activas_df.iterrows():
            emoji = "🔥" if r["tipo_racha"] == "victorias" else "❄️"
            color = "#238636" if r["tipo_racha"] == "victorias" else "#da3633"

            st.markdown(
                f'''
                <div style="
                    background:#161b22;
                    border-left:4px solid {color};
                    border-radius:8px;
                    padding:10px 14px;
                    margin-bottom:8px;
                ">
                    <div style="color:#ffffff; font-weight:600; font-size:15px;">
                        {emoji} {r["nombre"]}
                    </div>
                    <div style="color:#8b949e; font-size:14px;">
                        {r["longitud"]} {r["tipo_racha"]}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### 🏆 10 Mejores rachas de victorias")

        # TOP 10
        st.dataframe(
            df_rachas_v.head(10),
            use_container_width=True,
            hide_index=True
        )

        # MAX por jugador
        max_v = df_rachas_v.groupby("nombre")["racha"].max().reset_index()
        max_v = max_v.sort_values(by="racha", ascending=False)

        st.markdown("##### Máxima racha por jugador (Victoria)")
        st.dataframe(max_v, use_container_width=True, hide_index=True)

    with col3:
        st.markdown("#### 💀 10 Peores rachas de derrotas")

        # TOP 10
        st.dataframe(
            df_rachas_d.head(10),
            use_container_width=True,
            hide_index=True
        )

        # MAX por jugador
        max_d = df_rachas_d.groupby("nombre")["racha"].max().reset_index()
        max_d = max_d.sort_values(by="racha", ascending=False)

        st.markdown("##### Máxima racha por jugador (Derrota)")
        st.dataframe(max_d, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "📊 Gráficas":
    st.markdown("## 📊 Gráficas")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 % Victorias por jornada",
        "⚡ % Victorias por partido",
        "📊 V/D por jugador",
        "🎯 Diferencia juegos"
    ])

    with tab3:
        plot_data = clasificacion.sort_values("partidos_jugados", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        x = range(len(plot_data))
        ax.bar(x, plot_data["derrotas"], color="#da3633", label="Derrotas", width=0.6)
        ax.bar(x, plot_data["victorias"], bottom=plot_data["derrotas"], color="#238636", label="Victorias", width=0.6)

        for i, (idx, row) in enumerate(plot_data.iterrows()):
            ax.text(i, row["derrotas"] / 2, str(int(row["derrotas"])),
                    ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            ax.text(i, row["derrotas"] + row["victorias"] / 2, str(int(row["victorias"])),
                    ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            ax.text(i, row["partidos_jugados"] + 0.3, f"{row['porcentaje_victorias']}%",
                    ha="center", va="bottom", color="#e6edf3", fontsize=9)

        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_data["nombre"], rotation=30, ha="right", color="#e6edf3")
        ax.set_ylabel("Partidos", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#30363d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab1:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        for nombre in nombres:
            datos = ranking_jornada[
                ranking_jornada["nombre"] == nombre
            ].sort_values("hasta_jornada")

            ax.plot(
                datos["hasta_jornada"],
                datos["porcentaje_victorias"],
                marker="o",
                linewidth=2.5,   # 👈 como ranking
                markersize=6,    # 👈 como ranking
                color=mapa_colores[nombre],
                label=nombre
            )

        # Líneas verticales por jornada (más sutil aquí pero mismo estilo)
        jornadas = sorted(ranking_jornada["hasta_jornada"].unique())

        for j in jornadas:
            ax.axvline(
                x=j,
                color="#30363d",
                linewidth=1,
                linestyle="--",
                alpha=0.4
            )

        # Ejes
        ax.set_xlabel("Jornada", color="#8b949e")
        ax.set_ylabel("% Victorias", color="#8b949e")

        ax.set_ylim(-5, 105)

        ax.set_xticks(jornadas)
        ax.tick_params(colors="#8b949e")

        ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        ax.legend(
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#e6edf3",
            fontsize=9
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        top_diff = clasificacion.sort_values("diferencia_juegos", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        colores_bar = ["#238636" if v >= 0 else "#da3633" for v in top_diff["diferencia_juegos"]]
        ax.bar(top_diff["nombre"], top_diff["diferencia_juegos"], color=colores_bar, width=0.6)
        for i, (idx, row) in enumerate(top_diff.iterrows()):
            va = "bottom" if row["diferencia_juegos"] >= 0 else "top"
            ax.text(i, row["diferencia_juegos"], str(int(row["diferencia_juegos"])),
                    ha="center", va=va, color="#e6edf3", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#30363d", linewidth=1)
        ax.set_ylabel("Diferencia de juegos", color="#8b949e")
        ax.set_xticklabels(top_diff["nombre"], rotation=30, ha="right", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#30363d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        ranking_partido = calcular_ranking_por_partido(df)

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        for nombre in nombres:
            datos = ranking_partido[ranking_partido["nombre"] == nombre].sort_values("hasta_partido")

            ax.plot(
                datos["hasta_partido"],
                datos["porcentaje_victorias"],
                marker="o",
                linewidth=2,
                markersize=4,
                color=mapa_colores[nombre],
                label=nombre
            )

        jornada_cambios = ranking_partido.drop_duplicates("id_jornada").sort_values("hasta_partido")

        for _, jrow in jornada_cambios.iterrows():
            ax.axvline(
                x=jrow["hasta_partido"] - 0.5,
                color="#30363d",
                linewidth=1.6,
                linestyle="--"
            )
            ax.text(
                jrow["hasta_partido"] - 0.5,
                -2,  # 👈 abajo del todo
                f"J{int(jrow['id_jornada'])}",
                color="#8b949e",
                fontsize=7,
                ha="center"
            )

        # Ejes
        ax.set_xlabel("Nº partido acumulado", color="#8b949e")
        ax.set_ylabel("% Victorias", color="#8b949e")

        ax.set_ylim(-5, 105)

        ax.set_xticks(sorted(ranking_partido["hasta_partido"].unique()))
        ax.tick_params(axis='x', labelsize=7, colors="#8b949e")
        ax.tick_params(axis='y', colors="#8b949e")

        ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        ax.legend(
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#e6edf3",
            fontsize=9
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "💻 Predictor":
    st.markdown("## 💻 Predictor de Partido")
 
    df_hist = df_completo.copy()
 
    # with st.spinner("⚙️ Validando modelo con datos históricos..."):
    #     acc, n_validados = validacion_historica(df_hist)
 
    # nombres_todos = sorted(df_hist["nombre"].unique())
 
    st.divider()

    col_eq1, col_vs, col_eq2 = st.columns([5, 1, 5])

    with col_eq1:
        st.markdown("#### 🔵 Equipo 1")

        j1a = st.selectbox(
            "Jugador A",
            nombres,
            key="j1a"
        )

        j1b = st.selectbox(
            "Jugador B",
            [j for j in nombres if j != j1a],
            key="j1b"
        )

    with col_vs:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### VS")

    with col_eq2:
        st.markdown("#### 🔴 Equipo 2")

        disp2 = [j for j in nombres if j not in [j1a, j1b]]

        j2a = st.selectbox(
            "Jugador C",
            disp2,
            key="j2a"
        )

        j2b = st.selectbox(
            "Jugador D",
            [j for j in disp2 if j != j2a],
            key="j2b"
        )

    eq1 = [j1a, j1b]
    eq2 = [j2a, j2b]

    st.divider()
 
    if st.button("🎯 Calcular predicción", use_container_width=True, type="primary"):
 
        prob_eq1, prob_eq2, desglose = calcular_score(df_completo, eq1, eq2)
 
        pct1 = round(prob_eq1 * 100, 1)
        pct2 = round(prob_eq2 * 100, 1)
        nombre_eq1 = f"{eq1[0]} & {eq1[1]}"
        nombre_eq2 = f"{eq2[0]} & {eq2[1]}"
        favorito = nombre_eq1 if pct1 >= pct2 else nombre_eq2
        pct_fav = max(pct1, pct2)
        color_fav = "#1f6feb" if pct1 >= pct2 else "#da3633"
 
        # ── Resultado principal ──
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid {color_fav};border-radius:14px;
                    padding:24px 28px;margin-bottom:16px;">
            <div style="font-family:'Oswald',sans-serif;font-size:0.9rem;
                        color:#8b949e;letter-spacing:0.1em;margin-bottom:6px;">FAVORITO</div>
            <div style="font-family:'Oswald',sans-serif;font-size:2rem;
                        color:#ffffff;font-weight:700;">{favorito}</div>
            <div style="font-size:1.1rem;color:{color_fav};font-weight:600;margin-top:4px;">
                {pct_fav}% de probabilidad de victoria
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # ── Barra visual ──
        fig, ax = plt.subplots(figsize=(10, 1.6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.barh(0, pct1, color="#1f6feb", height=0.5)
        ax.barh(0, pct2, left=pct1, color="#da3633", height=0.5)
        ax.text(pct1 / 2, 0, f"{nombre_eq1}\n{pct1}%",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(pct1 + pct2 / 2, 0, f"{nombre_eq2}\n{pct2}%",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
        st.divider()

        # ── Desglose por factor ──
        st.markdown("#### 🔍 Desglose por factor")
 
        _, n_exactos = _h2h_pareja_exacta(df_completo, eq1, eq2)

        for feature, peso in PESOS.items():

            if feature not in desglose:
                continue

            v1, v2 = desglose[feature]

            ventaja1 = v1 >= v2
            color1 = "#1f6feb" if ventaja1 else "#8b949e"
            color2 = "#da3633" if not ventaja1 else "#8b949e"
            icono = "🔵" if ventaja1 else "🔴"

            bar_eq1 = v1 / (v1 + v2) * 100 if (v1 + v2) > 0 else 50
            bar_eq2 = 100 - bar_eq1

            extra_h2h = ""
            if feature == "H2H pareja exacta":
                if n_exactos > 0:
                    wins_eq1 = int(round(v1 * n_exactos))
                    wins_eq2 = n_exactos - wins_eq1

                    extra_h2h = (
                        f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;margin-top:6px;'>"
                        f"<span style='color:#1f6feb;font-weight:600;'>{nombre_eq1}: {wins_eq1} victorias</span>"
                        f"<span style='color:#da3633;font-weight:600;'>{nombre_eq2}: {wins_eq2} victorias</span>"
                        f"</div>"
                        f"<div style='color:#8b949e;font-size:0.75rem;margin-top:2px;'>📊 {n_exactos} partidos exactos</div>"
                    )

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="color:#e6edf3;font-weight:600;">{icono} {feature}</span>
                </div>
                <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:8px;">
                    <div style="width:{bar_eq1:.1f}%;background:#1f6feb;"></div>
                    <div style="width:{bar_eq2:.1f}%;background:#da3633;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.88rem;">
                    <span style="color:{color1};font-weight:600;">{nombre_eq1}: {v1*100:.1f}%</span>
                    <span style="color:{color2};font-weight:600;">{nombre_eq2}: {v2*100:.1f}%</span>
                </div>
                {extra_h2h}
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── 🔥 HISTORIAL DE ENFRENTAMIENTOS PREVIOS (ORDEN POR FECHA REAL) ──
        _, n_exactos = _h2h_pareja_exacta(df_hist, eq1, eq2)
        
        if n_exactos > 0:
            st.markdown(f"#### 🎾 Historial Directo ({n_exactos} partidos)")
            
            p_ids = set()
            for e1, e2 in [(1, 2), (2, 1)]:
                p_j1a = set(df_hist[(df_hist["nombre"] == eq1[0]) & (df_hist["equipo"] == e1)]["id_partido"])
                p_j1b = set(df_hist[(df_hist["nombre"] == eq1[1]) & (df_hist["equipo"] == e1)]["id_partido"])
                p_j2a = set(df_hist[(df_hist["nombre"] == eq2[0]) & (df_hist["equipo"] == e2)]["id_partido"])
                p_j2b = set(df_hist[(df_hist["nombre"] == eq2[1]) & (df_hist["equipo"] == e2)]["id_partido"])
                p_ids.update(p_j1a & p_j1b & p_j2a & p_j2b)

            # 🛠️ ORDENACIÓN POR FECHA (Convertimos para asegurar orden cronológico)
            df_partidos_h2h = df_hist[df_hist["id_partido"].isin(p_ids)].copy()
            df_partidos_h2h["fecha_dt"] = pd.to_datetime(df_partidos_h2h["fecha"])
            
            # Quitamos duplicados de id_partido y ordenamos por fecha real descendente
            df_partidos_h2h = df_partidos_h2h.drop_duplicates("id_partido").sort_values("fecha_dt", ascending=False)

            for _, partido in df_partidos_h2h.iterrows():
                # Formatear fecha para mostrar
                fecha_display = partido["fecha_dt"].strftime("%d/%m/%Y")
                temp_label = f"T{partido['temporada']}" if "temporada" in partido else ""

                # Marcador
                j1, j2 = partido["juegos_equipo1"], partido["juegos_equipo2"]
                es_eq1_azul = (partido["nombre"] in eq1)
                res_azul = j1 if es_eq1_azul else j2
                res_rojo = j2 if es_eq1_azul else j1
                
                # Nombres (filtrados del dataframe original para ese ID)
                nombres_partido = df_hist[df_hist["id_partido"] == partido["id_partido"]]["nombre"].tolist()
                txt_azul = " & ".join([n for n in nombres_partido if n in eq1])
                txt_rojo = " & ".join([n for n in nombres_partido if n not in eq1])

                # UN SOLO st.markdown para todo el bloque para evitar que se rompa el diseño
                st.markdown(f"""
                <div style="background:#0d1117; border:1px solid #30363d; border-radius:10px; padding:12px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px; align-items:center;">
                        <span style="font-size:0.8rem; color:#8b949e; font-weight:500;">📅 {fecha_display} <span style="margin-left:8px; color:#58a6ff; background:#58a6ff11; padding:2px 6px; border-radius:4px;">{temp_label}</span></span>
                        <span style="font-size:0.65rem; color:#238636; background:#23863622; padding:2px 8px; border-radius:4px; border:1px solid #23863644;">FINALIZADO</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div style="flex:1; text-align:right; color:#1f6feb; font-weight:600; font-size:0.9rem;">{txt_azul}</div>
                        <div style="background:#161b22; padding:4px 12px; border-radius:6px; border:1px solid #30363d; min-width:70px; text-align:center;">
                            <span style="font-weight:800; font-size:1.2rem; color:#1f6feb;">{res_azul}</span>
                            <span style="color:#8b949e; margin:0 4px; font-weight:400;">-</span>
                            <span style="font-weight:800; font-size:1.2rem; color:#da3633;">{res_rojo}</span>
                        </div>
                        <div style="flex:1; text-align:left; color:#da3633; font-weight:600; font-size:0.9rem;">{txt_rojo}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
        # ── Accuracy del modelo ──
        #acc_color = "#238636" if acc >= 0.60 else "#d29922" if acc >= 0.50 else "#da3633"
        #st.markdown(f"""
        #<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;">
        #    <div style="color:#8b949e;font-size:0.85rem;margin-bottom:4px;">📊 Validación histórica del modelo</div>
        #    <div style="color:{acc_color};font-size:1.3rem;font-weight:700;">{acc*100:.1f}% de acierto</div>
        #    <div style="color:#8b949e;font-size:0.8rem;margin-top:4px;">
        #        sobre {n_validados} partidos históricos · excluye las primeras jornadas (sin datos previos suficientes)
        #    </div>
        #</div>
        #""", unsafe_allow_html=True)
 
    else:
        st.markdown("#### ⚖️ Pesos del modelo")
        for feature, peso in PESOS.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:#161b22;border:1px solid #30363d;border-radius:8px;
                        padding:10px 16px;margin-bottom:6px;">
                <span style="color:#e6edf3;">{feature}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN: CALCULADORA
# ═══════════════════════════════════════════════════════════════════════════
 
elif seccion == "🧮 Calculadora":
    st.markdown("## 🧮 Calculadora")
    
    tab1, tab2, tab3 = st.tabs(["📅 Simular Jornada", "🎯 Adelantar Jugador", "👑 Objetivo Ranking"])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: SIMULAR JORNADA
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Añade partidos hipotéticos y ve cómo cambia la clasificación")
        
        # Número de partidos a simular
        num_partidos = st.number_input(
            "¿Cuántos partidos quieres simular?",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )
        
        nombres_disponibles = nombres
        partidos_simulados = []
        
        st.divider()
        
        # Generar selectores por cada partido
        for i in range(num_partidos):
            st.markdown(f"#### 🎾 Partido {i+1}")
            
            col1, col2, col3, col4, col5, col6 = st.columns([3, 3, 3, 3, 2, 2])
            
            with col1:
                j1 = st.selectbox(
                    "Jugador 1 (Equipo 1)",
                    nombres_disponibles,
                    key=f"sim_j1_{i}"
                )
            
            with col2:
                opciones_j2 = [n for n in nombres_disponibles if n != j1]
                j2 = st.selectbox(
                    "Jugador 2 (Equipo 1)",
                    opciones_j2,
                    key=f"sim_j2_{i}"
                )
            
            with col3:
                opciones_j3 = [n for n in nombres_disponibles if n not in [j1, j2]]
                j3 = st.selectbox(
                    "Jugador 3 (Equipo 2)",
                    opciones_j3,
                    key=f"sim_j3_{i}"
                )
            
            with col4:
                opciones_j4 = [n for n in nombres_disponibles if n not in [j1, j2, j3]]
                j4 = st.selectbox(
                    "Jugador 4 (Equipo 2)",
                    opciones_j4,
                    key=f"sim_j4_{i}"
                )
            
            with col5:
                ganador = st.selectbox(
                    "Ganador",
                    ["Equipo 1", "Equipo 2"],
                    key=f"sim_ganador_{i}"
                )
            
            with col6:
                marcador = st.selectbox(
                    "Marcador",
                    ["2-0", "2-1"],
                    key=f"sim_marcador_{i}"
                )
            
            # Guardar partido
            eq1 = [j1, j2]
            eq2 = [j3, j4]
            
            if marcador == "2-0":
                juegos_ganador = 2
                juegos_perdedor = 0
            else:
                juegos_ganador = 2
                juegos_perdedor = 1
            
            if ganador == "Equipo 1":
                partidos_simulados.append({
                    "eq1": eq1,
                    "eq2": eq2,
                    "juegos_eq1": juegos_ganador,
                    "juegos_eq2": juegos_perdedor,
                    "ganador": 1
                })
            else:
                partidos_simulados.append({
                    "eq1": eq1,
                    "eq2": eq2,
                    "juegos_eq1": juegos_perdedor,
                    "juegos_eq2": juegos_ganador,
                    "ganador": 2
                })
            
            st.markdown("---")
        
        # Botón para calcular
        if st.button("🔮 Calcular clasificación simulada", use_container_width=True, type="primary"):
            
            # Crear DataFrame simulado
            df_simulado = df.copy()
            
            # Añadir partidos simulados
            id_partido_base = df_simulado["id_partido"].astype(str).str.split("_").str[0].astype(int).max() + 1000
            jornada_sim = df_simulado["id_jornada"].max() + 1
            
            filas_nuevas = []
            
            for idx, partido in enumerate(partidos_simulados):
                pid = f"{id_partido_base + idx}_SIM"
                
                # Equipo 1
                for jugador in partido["eq1"]:
                    filas_nuevas.append({
                        "id_partido": pid,
                        "id_jornada": jornada_sim,
                        "nombre": jugador,
                        "equipo": 1,
                        "juegos_equipo1": partido["juegos_eq1"],
                        "juegos_equipo2": partido["juegos_eq2"],
                        "equipo_ganador": partido["ganador"],
                        "victoria": 1 if partido["ganador"] == 1 else 0,
                        "juegos_ganados": partido["juegos_eq1"],
                        "juegos_perdidos": partido["juegos_eq2"],
                        "temporada": "SIMULADO"
                    })
                
                # Equipo 2
                for jugador in partido["eq2"]:
                    filas_nuevas.append({
                        "id_partido": pid,
                        "id_jornada": jornada_sim,
                        "nombre": jugador,
                        "equipo": 2,
                        "juegos_equipo1": partido["juegos_eq1"],
                        "juegos_equipo2": partido["juegos_eq2"],
                        "equipo_ganador": partido["ganador"],
                        "victoria": 1 if partido["ganador"] == 2 else 0,
                        "juegos_ganados": partido["juegos_eq2"],
                        "juegos_perdidos": partido["juegos_eq1"],
                        "temporada": "SIMULADO"
                    })
            
            df_simulado = pd.concat([df_simulado, pd.DataFrame(filas_nuevas)], ignore_index=True)
            
            # Calcular clasificaciones
            clasificacion_real = calcular_clasificacion(df)
            clasificacion_simulada = calcular_clasificacion(df_simulado)
            
            # Merge para comparar
            comparacion = clasificacion_real[["nombre", "victorias", "diferencia_juegos", "porcentaje_victorias"]].merge(
                clasificacion_simulada[["nombre", "victorias", "diferencia_juegos", "porcentaje_victorias"]],
                on="nombre",
                suffixes=("_real", "_sim")
            )
            
            # Posiciones
            clasificacion_real_sorted = clasificacion_real.reset_index(drop=True)
            clasificacion_real_sorted["pos_real"] = clasificacion_real_sorted.index + 1
            
            clasificacion_simulada_sorted = clasificacion_simulada.reset_index(drop=True)
            clasificacion_simulada_sorted["pos_sim"] = clasificacion_simulada_sorted.index + 1
            
            comparacion = comparacion.merge(
                clasificacion_real_sorted[["nombre", "pos_real"]],
                on="nombre"
            ).merge(
                clasificacion_simulada_sorted[["nombre", "pos_sim"]],
                on="nombre"
            )
            
            comparacion["cambio_pos"] = comparacion["pos_real"] - comparacion["pos_sim"]
            
            st.divider()
            st.markdown("### 📊 Resultado de la simulación")
            
            # Mostrar tabla comparativa
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 🏆 Clasificación REAL")
                tabla_real = clasificacion_real[["nombre", "victorias", "derrotas", "diferencia_juegos", "porcentaje_victorias"]].copy()
                tabla_real.columns = ["Jugador", "V", "D", "+/-", "% V"]
                tabla_real.index = range(1, len(tabla_real) + 1)
                st.dataframe(
                    tabla_real.style.background_gradient(subset=["% V"], cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True
                )
            
            with col_b:
                st.markdown("#### 🔮 Clasificación SIMULADA")
                tabla_sim = clasificacion_simulada[["nombre", "victorias", "derrotas", "diferencia_juegos", "porcentaje_victorias"]].copy()
                tabla_sim.columns = ["Jugador", "V", "D", "+/-", "% V"]
                tabla_sim.index = range(1, len(tabla_sim) + 1)
                st.dataframe(
                    tabla_sim.style.background_gradient(subset=["% V"], cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True
                )
            
            st.divider()
            st.markdown("### 📈 Cambios de posición")
            
            # Ordenar por mayor cambio
            comparacion_sorted = comparacion.sort_values("cambio_pos", ascending=False)
            
            for _, row in comparacion_sorted.iterrows():
                cambio = int(row["cambio_pos"])
                
                if cambio > 0:
                    emoji = "📈"
                    color = "#238636"
                    texto = f"sube {cambio} posición/es"
                elif cambio < 0:
                    emoji = "📉"
                    color = "#da3633"
                    texto = f"baja {abs(cambio)} posición/es"
                else:
                    emoji = "➡️"
                    color = "#8b949e"
                    texto = "mantiene posición"
                
                st.markdown(f"""
                <div style="background:#161b22;border-left:4px solid {color};border-radius:8px;
                            padding:10px 14px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:#ffffff;font-weight:600;font-size:1.1rem;">{emoji} {row['nombre']}</span>
                            <span style="color:#8b949e;margin-left:12px;">{texto}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:#8b949e;font-size:0.9rem;">Pos: </span>
                            <span style="color:#ffffff;font-weight:700;">#{int(row['pos_real'])} → #{int(row['pos_sim'])}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: ADELANTAR JUGADOR
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Calcula qué necesitas para adelantar a otro jugador")
        
        nombres_lista = nombres
        
        col1, col2 = st.columns(2)
        
        with col1:
            jugador_atras = st.selectbox(
                "¿Qué necesita...",
                nombres_lista,
                key="calc_j1"
            )
        
        with col2:
            opciones_delante = [n for n in nombres_lista if n != jugador_atras]
            jugador_delante = st.selectbox(
                "...para adelantar a?",
                opciones_delante,
                key="calc_j2"
            )
        
        if st.button("🔍 Calcular", use_container_width=True, type="primary"):
            
            # Obtener stats actuales
            stats_atras = clasificacion[clasificacion["nombre"] == jugador_atras].iloc[0]
            stats_delante = clasificacion[clasificacion["nombre"] == jugador_delante].iloc[0]
            
            pos_atras = clasificacion.index[clasificacion["nombre"] == jugador_atras][0]
            pos_delante = clasificacion.index[clasificacion["nombre"] == jugador_delante][0]
            
            # Verificar si ya está por delante
            if pos_atras <= pos_delante:
                st.success(f"🎉 **{jugador_atras}** ya está por delante de **{jugador_delante}** (posición #{pos_atras} vs #{pos_delante})")
            else:
                st.divider()
                
                # Stats actuales
                v_atras = int(stats_atras["victorias"])
                v_delante = int(stats_delante["victorias"])
                pj_atras = int(stats_atras["partidos_jugados"])
                pj_delante = int(stats_delante["partidos_jugados"])
                pct_atras = stats_atras["porcentaje_victorias"]
                pct_delante = stats_delante["porcentaje_victorias"]
                diff_juegos = stats_delante["diferencia_juegos"] - stats_atras["diferencia_juegos"]
                
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                            padding:16px;margin-bottom:16px;">
                    <div style="color:#8b949e;font-size:0.9rem;margin-bottom:8px;">Situación actual</div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
                        <div>
                            <div style="color:#da3633;font-weight:600;font-size:1.1rem;">{jugador_atras}</div>
                            <div style="color:#8b949e;font-size:0.85rem;">Posición #{pos_atras}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:#238636;font-weight:600;font-size:1.1rem;">{jugador_delante}</div>
                            <div style="color:#8b949e;font-size:0.85rem;">Posición #{pos_delante}</div>
                        </div>
                    </div>
                    <div style="color:#ffffff;font-size:0.9rem;">
                        <div>📊 {jugador_atras}: {v_atras}V de {pj_atras}PJ ({pct_atras:.1f}%)</div>
                        <div>📊 {jugador_delante}: {v_delante}V de {pj_delante}PJ ({pct_delante:.1f}%)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- DATOS BASE ---
                v_a, pj_a = int(stats_atras["victorias"]), int(stats_atras["partidos_jugados"])
                v_d, pj_d = int(stats_delante["victorias"]), int(stats_delante["partidos_jugados"])
                pct_d = stats_delante["porcentaje_victorias"]

                # 1. ESCENARIO ESTÁTICO (El de los "46 partidos")
                vics_estatico = 0
                for x in range(1, 101):
                    if ((v_a + x) / (pj_a + x)) * 100 > pct_d:
                        vics_estatico = x
                        break

                # 2. ESCENARIO CRUCE DIRECTO (Tú ganas, él pierde)
                vics_cruce = 0
                for x in range(1, 50):
                    # En cada partido: tú +1V y +1PJ | él +0V y +1PJ
                    if ((v_a + x) / (pj_a + x)) > (v_d / (pj_d + x)):
                        vics_cruce = x
                        break

                # 3. JUSTICIA POR VOLUMEN (Si él jugara los mismos que tú pero perdiendo)
                # ¿A cuánto bajaría su % si tuviera tus mismos PJ?
                pj_diff = max(0, pj_a - pj_d)
                pct_proyectado_rival = (v_d / max(pj_a, pj_d)) * 100
                vics_justicia = 0
                for x in range(1, 50):
                    if ((v_a + x) / (pj_a + x)) * 100 > pct_proyectado_rival:
                        vics_justicia = x
                        break

                # --- RENDERIZADO DE BLOQUES ---
                st.divider()
                
                # Bloque 1: Cruce Directo (El más realista)
                st.markdown(f"""
                <div style="background:#1c2128; border:1px solid #2188ff; border-radius:10px; padding:15px; margin-bottom:10px;">
                    <h4 style="color:#2188ff; margin:0;">⚔️ Escenario: Cruce Directo</h4>
                    <p style="margin:10px 0; font-size:1.1rem;">Necesitas ganar <b>{vics_cruce} partido(s)</b> contra él.</p>
                    <p style="color:#8b949e; font-size:0.85rem;">Si jugáis cara a cara, tu porcentaje sube y el suyo baja rápidamente. Es el camino más corto.</p>
                </div>
                """, unsafe_allow_html=True)

                # Bloque 2: Justicia por Volumen
                st.markdown(f"""
                <div style="background:#1c2128; border:1px solid #db6d28; border-radius:10px; padding:15px; margin-bottom:10px;">
                    <h4 style="color:#db6d28; margin:0;">⚖️ Escenario: Justicia por Volumen</h4>
                    <p style="margin:10px 0; font-size:1.1rem;">Necesitas <b>{vics_justicia} victoria(s)</b> adicionales.</p>
                    <p style="color:#8b949e; font-size:0.85rem;">Calculado asumiendo que {jugador_delante} igualara tus {pj_a} partidos jugados manteniendo sus victorias actuales.</p>
                </div>
                """, unsafe_allow_html=True)

                # Bloque 3: El Muro (Estático)
                txt_muro = f"{vics_estatico} partidos"
                st.markdown(f"""
                <div style="background:#1c2128; border:1px solid #808080; border-radius:10px; padding:15px;">
                    <h4 style="color:#8b949e; margin:0;">📉 Escenario: El Muro</h4>
                    <p style="margin:10px 0; font-size:1.1rem;">Necesitas <b>{txt_muro}</b> seguidos.</p>
                    <p style="color:#8b949e; font-size:0.85rem;">Si {jugador_delante} no vuelve a jugar nunca más, mantiene su % intacto.</p>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: OBJETIVO RANKING
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### ¿Qué necesitas para alcanzar una posición específica?")
        
        nombres_lista = nombres
        
        col_j, col_p = st.columns(2)
        with col_j:
            soy_yo = st.selectbox("Selecciona tu nombre", nombres_lista, key="obj_yo")
        with col_p:
            pos_deseada = st.slider("Posición objetivo", 1, len(clasificacion), 1, key="obj_pos")
        
        if st.button("🏆 Calcular Camino al Éxito", use_container_width=True, type="primary"):
            # 1. Identificar al rival que ocupa esa posición (o el inmediatamente superior)
            # Nota: clasificacion ya viene ordenada por porcentaje_victorias
            rival_objetivo = clasificacion.iloc[pos_deseada - 1]
            nombre_rival = rival_objetivo["nombre"]
            
            # 2. Obtener Stats
            stats_yo = clasificacion[clasificacion["nombre"] == soy_yo].iloc[0]
            pos_actual = clasificacion.index[clasificacion["nombre"] == soy_yo][0]
            
            if pos_actual <= pos_deseada:
                st.balloons()
                st.success(f"⭐ ¡Ya estás en el objetivo! Actualmente eres el **#{pos_actual}**.")
            else:
                st.info(f"Para ser **#{pos_deseada}**, tienes que superar el **{rival_objetivo['porcentaje_victorias']}%** de **{nombre_rival}**.")
                
                # --- CÁLCULOS (Reutilizando tu lógica de la Tab 2) ---
                v_a, pj_a = int(stats_yo["victorias"]), int(stats_yo["partidos_jugados"])
                v_d, pj_d = int(rival_objetivo["victorias"]), int(rival_objetivo["partidos_jugados"])
                pct_d = rival_objetivo["porcentaje_victorias"]

                # A. Escenario El Muro (Rival estático)
                vics_estatico = 0
                for x in range(1, 150):
                    if ((v_a + x) / (pj_a + x)) * 100 > pct_d:
                        vics_estatico = x
                        break
                
                # B. Escenario Cruce Directo (Jugando contra el de esa posición)
                vics_cruce = 0
                for x in range(1, 100):
                    if ((v_a + x) / (pj_a + x)) > (v_d / (pj_d + x)):
                        vics_cruce = x
                        break

                # --- RENDERIZADO VISUAL ---
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    txt_m = f"{vics_estatico} partidos" if vics_estatico > 0 else "N/A"
                    st.markdown(f"""
                    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:15px; text-align:center;">
                        <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase;">Solo ante el peligro</div>
                        <div style="color:#ffffff; font-size:1.8rem; font-weight:800; margin:10px 0;">{txt_m}</div>
                        <div style="color:#8b949e; font-size:0.85rem;">Ganados consecutivamente si <b>{nombre_rival}</b> no juega más.</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_res2:
                    txt_c = f"{vics_cruce} partidos" if vics_cruce > 0 else "N/A"
                    st.markdown(f"""
                    <div style="background:#161b22; border:1px solid #2188ff; border-radius:10px; padding:15px; text-align:center;">
                        <div style="color:#2188ff; font-size:0.8rem; text-transform:uppercase;">Duelo Directo</div>
                        <div style="color:#ffffff; font-size:1.8rem; font-weight:800; margin:10px 0;">{txt_c}</div>
                        <div style="color:#8b949e; font-size:0.85rem;">Ganados cara a cara contra <b>{nombre_rival}</b> para quitarle el puesto.</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: BUSCADOR
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "🔍 Buscador":
    st.markdown("## 🔍 Buscador de Partidos")

    tab_pareja, tab_enfrentamiento = st.tabs([
        "🤝 Partidos de una pareja",
        "⚔️ Pareja contra pareja"
    ])

    # ── TAB 1: partidos de una pareja ──────────────────────────────────────
    with tab_pareja:
        st.markdown("### Todos los partidos de una pareja")

        col1, col2 = st.columns(2)
        pa = col1.selectbox("Jugador A", nombres, key="bus_pa")
        pb = col2.selectbox("Jugador B", [j for j in nombres if j != pa], key="bus_pb")

        # Buscar partidos donde pa y pb jugaron juntos (mismo equipo)
        s1 = df_completo[df_completo["nombre"] == pa][["id_partido", "equipo", "victoria",
                                                         "juegos_ganados", "juegos_perdidos",
                                                         "id_jornada", "temporada"]]
        s2 = df_completo[df_completo["nombre"] == pb][["id_partido", "equipo"]]
        juntos = s1.merge(s2, on=["id_partido", "equipo"])

        if juntos.empty:
            st.info(f"**{pa}** y **{pb}** no han jugado juntos todavía.")
        else:
            pids = juntos["id_partido"].unique()
            df_partidos = df_completo[df_completo["id_partido"].isin(pids)].copy()
            df_partidos["_fecha_dt"] = pd.to_datetime(df_partidos["fecha"], errors="coerce")
            df_partidos_unicos = df_partidos.drop_duplicates("id_partido").sort_values(
                "_fecha_dt", ascending=False
            )

            total = len(df_partidos_unicos)
            victorias = juntos["victoria"].sum()
            derrotas = total - victorias
            pct = round(victorias / total * 100, 1) if total > 0 else 0

            # Resumen
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎾 Partidos juntos", total)
            c2.metric("✅ Victorias", int(victorias))
            c3.metric("❌ Derrotas", int(derrotas))
            c4.metric("📈 % Victorias", f"{pct}%")

            st.divider()

            for _, partido in df_partidos_unicos.iterrows():
                pid = partido["id_partido"]
                fecha_display = partido["_fecha_dt"].strftime("%d/%m/%Y") if pd.notna(partido["_fecha_dt"]) else "N/A"
                temp_label = str(partido["temporada"]) if "temporada" in partido else ""

                jugadores_partido = df_completo[df_completo["id_partido"] == pid]
                equipo_pareja = jugadores_partido[jugadores_partido["nombre"] == pa]["equipo"].values[0]

                eq_azul = jugadores_partido[jugadores_partido["equipo"] == equipo_pareja]["nombre"].tolist()
                eq_rojo = jugadores_partido[jugadores_partido["equipo"] != equipo_pareja]["nombre"].tolist()

                fila = jugadores_partido.iloc[0]
                if equipo_pareja == 1:
                    g_azul, g_rojo = int(fila["juegos_equipo1"]), int(fila["juegos_equipo2"])
                    gano_azul = fila["equipo_ganador"] == 1
                else:
                    g_azul, g_rojo = int(fila["juegos_equipo2"]), int(fila["juegos_equipo1"])
                    gano_azul = fila["equipo_ganador"] == 2

                color_score_azul = "#238636" if gano_azul else "#da3633"
                color_score_rojo = "#da3633" if gano_azul else "#238636"
                resultado_txt = "VICTORIA" if gano_azul else "DERROTA"
                resultado_color = "#238636" if gano_azul else "#da3633"

                txt_azul = " & ".join(eq_azul)
                txt_rojo = " & ".join(eq_rojo)
                num_partido = str(pid).split("_")[0]

                st.markdown(f"""
                <div style="background:#0d1117; border:1px solid #30363d; border-radius:10px;
                            padding:12px 16px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="color:#8b949e; font-size:0.78rem;">
                            📅 {fecha_display}
                            <span style="margin-left:8px; color:#58a6ff; background:#58a6ff11;
                                         padding:2px 6px; border-radius:4px;">{temp_label}</span>
                            <span style="margin-left:6px; color:#8b949e;">· Partido #{num_partido}</span>
                        </span>
                        <span style="font-size:0.7rem; color:{resultado_color};
                                     background:{resultado_color}22; padding:2px 8px;
                                     border-radius:4px; border:1px solid {resultado_color}44;
                                     font-weight:700;">{resultado_txt}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div style="flex:1; text-align:right; color:#1f6feb;
                                    font-weight:600; font-size:0.9rem;">{txt_azul}</div>
                        <div style="background:#161b22; padding:4px 14px; border-radius:6px;
                                    border:1px solid #30363d; text-align:center; min-width:70px;">
                            <span style="font-weight:800; font-size:1.2rem;
                                         color:{color_score_azul};">{g_azul}</span>
                            <span style="color:#8b949e; margin:0 4px;">-</span>
                            <span style="font-weight:800; font-size:1.2rem;
                                         color:{color_score_rojo};">{g_rojo}</span>
                        </div>
                        <div style="flex:1; text-align:left; color:#da3633;
                                    font-weight:600; font-size:0.9rem;">{txt_rojo}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: pareja contra pareja ────────────────────────────────────────
    with tab_enfrentamiento:
        st.markdown("### Partidos entre dos parejas concretas")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔵 Pareja 1**")
            e1a = st.selectbox("Jugador A", nombres, key="bus_e1a")
            e1b = st.selectbox("Jugador B", [j for j in nombres if j != e1a], key="bus_e1b")
        with col2:
            st.markdown("**🔴 Pareja 2**")
            disp2 = [j for j in nombres if j not in [e1a, e1b]]
            e2a = st.selectbox("Jugador C", disp2, key="bus_e2a")
            e2b = st.selectbox("Jugador D", [j for j in disp2 if j != e2a], key="bus_e2b")

        eq1_bus = [e1a, e1b]
        eq2_bus = [e2a, e2b]

        # Buscar partidos exactos (cualquier combinación de equipos)
        p_ids = set()
        for e1, e2 in [(1, 2), (2, 1)]:
            s_e1a = set(df_completo[(df_completo["nombre"] == e1a) & (df_completo["equipo"] == e1)]["id_partido"])
            s_e1b = set(df_completo[(df_completo["nombre"] == e1b) & (df_completo["equipo"] == e1)]["id_partido"])
            s_e2a = set(df_completo[(df_completo["nombre"] == e2a) & (df_completo["equipo"] == e2)]["id_partido"])
            s_e2b = set(df_completo[(df_completo["nombre"] == e2b) & (df_completo["equipo"] == e2)]["id_partido"])
            p_ids.update(s_e1a & s_e1b & s_e2a & s_e2b)

        if not p_ids:
            st.info(f"**{e1a} & {e1b}** nunca se han enfrentado a **{e2a} & {e2b}**.")
        else:
            df_h2h = df_completo[df_completo["id_partido"].isin(p_ids)].copy()
            df_h2h["_fecha_dt"] = pd.to_datetime(df_h2h["fecha"], errors="coerce")
            df_h2h_unicos = df_h2h.drop_duplicates("id_partido").sort_values("_fecha_dt", ascending=False)

            # Contar victorias de cada pareja
            v_eq1, v_eq2 = 0, 0
            for _, partido in df_h2h_unicos.iterrows():
                pid = partido["id_partido"]
                equipo_e1 = df_completo[(df_completo["id_partido"] == pid) &
                                        (df_completo["nombre"] == e1a)]["equipo"].values[0]
                gano_e1 = partido["equipo_ganador"] == equipo_e1
                if gano_e1:
                    v_eq1 += 1
                else:
                    v_eq2 += 1

            total_h2h = len(df_h2h_unicos)

            # Resumen
            c1, c2, c3 = st.columns(3)
            c1.metric("🎾 Partidos", total_h2h)
            c2.metric(f"🔵 {e1a} & {e1b}", f"{v_eq1} victorias ({round(v_eq1/total_h2h*100)}%)")
            c3.metric(f"🔴 {e2a} & {e2b}", f"{v_eq2} victorias ({round(v_eq2/total_h2h*100)}%)")

            st.divider()

            for _, partido in df_h2h_unicos.iterrows():
                pid = partido["id_partido"]
                fecha_display = partido["_fecha_dt"].strftime("%d/%m/%Y") if pd.notna(partido["_fecha_dt"]) else "N/A"
                temp_label = str(partido["temporada"]) if "temporada" in partido else ""
                num_partido = str(pid).split("_")[0]

                equipo_e1 = df_completo[(df_completo["id_partido"] == pid) &
                                        (df_completo["nombre"] == e1a)]["equipo"].values[0]
                gano_e1 = partido["equipo_ganador"] == equipo_e1

                if equipo_e1 == 1:
                    g_azul, g_rojo = int(partido["juegos_equipo1"]), int(partido["juegos_equipo2"])
                else:
                    g_azul, g_rojo = int(partido["juegos_equipo2"]), int(partido["juegos_equipo1"])

                color_azul = "#238636" if gano_e1 else "#da3633"
                color_rojo = "#da3633" if gano_e1 else "#238636"
                resultado_txt = f"Ganan {e1a} & {e1b}" if gano_e1 else f"Ganan {e2a} & {e2b}"
                resultado_color = "#238636" if gano_e1 else "#da3633"

                st.markdown(f"""
                <div style="background:#0d1117; border:1px solid #30363d; border-radius:10px;
                            padding:12px 16px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="color:#8b949e; font-size:0.78rem;">
                            📅 {fecha_display}
                            <span style="margin-left:8px; color:#58a6ff; background:#58a6ff11;
                                         padding:2px 6px; border-radius:4px;">{temp_label}</span>
                            <span style="margin-left:6px; color:#8b949e;">· Partido #{num_partido}</span>
                        </span>
                        <span style="font-size:0.7rem; color:{resultado_color};
                                     background:{resultado_color}22; padding:2px 8px;
                                     border-radius:4px; border:1px solid {resultado_color}44;
                                     font-weight:600;">{resultado_txt}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div style="flex:1; text-align:right; color:#1f6feb;
                                    font-weight:600; font-size:0.9rem;">{e1a} & {e1b}</div>
                        <div style="background:#161b22; padding:4px 14px; border-radius:6px;
                                    border:1px solid #30363d; text-align:center; min-width:70px;">
                            <span style="font-weight:800; font-size:1.2rem; color:{color_azul};">{g_azul}</span>
                            <span style="color:#8b949e; margin:0 4px;">-</span>
                            <span style="font-weight:800; font-size:1.2rem; color:{color_rojo};">{g_rojo}</span>
                        </div>
                        <div style="flex:1; text-align:left; color:#da3633;
                                    font-weight:600; font-size:0.9rem;">{e2a} & {e2b}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: ADMIN
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "🔐 Admin":

    import datetime

    st.markdown("## 🔐 Panel Admin")

    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False

    if not st.session_state.admin_ok:

        password = st.text_input("Contraseña", type="password")

        if password:
            if password == st.secrets["admin_password"]:
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.warning("Acceso restringido")

        st.stop()

    st.success("Acceso concedido")

    if st.button("Cerrar sesión"):
        st.session_state.admin_ok = False
        st.rerun()

    st.divider()

    # ─────────────────────────────────────────
    # FORMULARIO TIPO SIMULADOR (MEJORADO)
    # ─────────────────────────────────────────

    st.markdown("### 🎾 Nuevo Partido")

    nombres_disponibles = nombres  # por claridad

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        jornada = st.number_input("Jornada", min_value=1, step=1)

        fecha = st.date_input(
            "Fecha",
            value=datetime.date.today()
        )

    with col_info2:
        sede = st.text_input("Sede", value="")
        comentario = st.text_input("Comentario del partido (opcional)")

    st.markdown("---")

    # JUGADORES
    col1, col2 = st.columns(2)

    with col1:
        j1 = st.selectbox("Jugador 1 (Equipo 1)", nombres_disponibles)
        j2 = st.selectbox("Jugador 2 (Equipo 1)", [j for j in nombres_disponibles if j != j1])

    with col2:
        j3 = st.selectbox("Jugador 3 (Equipo 2)", [j for j in nombres_disponibles if j not in [j1, j2]])
        j4 = st.selectbox("Jugador 4 (Equipo 2)", [j for j in nombres_disponibles if j not in [j1, j2, j3]])

    st.markdown("---")

    colr1, colr2 = st.columns(2)

    with colr1:
        score1 = st.number_input("Juegos Equipo 1", 0, 2)

    with colr2:
        score2 = st.number_input("Juegos Equipo 2", 0, 2)

    submit = st.button("💾 Guardar partido")


    # ─────────────────────────────────────────
    # FUNCIÓN GITHUB
    # ─────────────────────────────────────────

    def subir_a_github(path, contenido, mensaje):
        token = st.secrets["github_token"]
        repo = "AlvarG12/padel_guimaneta"

        url = f"https://api.github.com/repos/{repo}/contents/{path}"

        contenido_b64 = base64.b64encode(contenido.encode()).decode()

        r = requests.get(url, headers={"Authorization": f"token {token}"})
        sha = r.json().get("sha") if r.status_code == 200 else None

        data = {
            "message": mensaje,
            "content": contenido_b64,
            "branch": "main"
        }

        if sha:
            data["sha"] = sha

        requests.put(url, json=data, headers={"Authorization": f"token {token}"})


    # ─────────────────────────────────────────
    # GUARDAR PARTIDO
    # ─────────────────────────────────────────

    if submit:

        try:
            partidos = pd.read_csv("data/partidos_25_26.csv")
            pj = pd.read_csv("data/partido_jugadores_25_26.csv")
            jugadores = pd.read_csv("data/jugadores.csv")

            mapa = dict(zip(jugadores["nombre"], jugadores["id_jugador"]))

            new_id = partidos["id_partido"].max() + 1
            ganador = 1 if score1 > score2 else 2

            nuevo_partido = {
                "id_partido": new_id,
                "id_jornada": jornada,
                "fecha": str(fecha),
                "sede": sede,
                "juegos_equipo1": score1,
                "juegos_equipo2": score2,
                "equipo_ganador": ganador,
                "comentario": comentario
            }

            nuevos_pj = [
                {"id_partido": new_id, "id_jugador": mapa[j1], "equipo": 1},
                {"id_partido": new_id, "id_jugador": mapa[j2], "equipo": 1},
                {"id_partido": new_id, "id_jugador": mapa[j3], "equipo": 2},
                {"id_partido": new_id, "id_jugador": mapa[j4], "equipo": 2},
            ]

            partidos = pd.concat([partidos, pd.DataFrame([nuevo_partido])], ignore_index=True)
            pj = pd.concat([pj, pd.DataFrame(nuevos_pj)], ignore_index=True)

            partidos_csv = partidos.to_csv(index=False)
            pj_csv = pj.to_csv(index=False)

            subir_a_github(
                "data/partidos_25_26.csv",
                partidos_csv,
                f"Nuevo partido jornada {jornada}"
            )

            subir_a_github(
                "data/partido_jugadores_25_26.csv",
                pj_csv,
                f"Nuevo partido jornada {jornada}"
            )

            st.success("✅ Partido guardado y subido a GitHub")

            st.cache_data.clear()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")