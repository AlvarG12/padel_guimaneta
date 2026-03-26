import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import os
import unicodedata

def quitar_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

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

@st.cache_data
def cargar_datos(base_path="data/"):
    print("🔍 DEBUG: Cargando datos...")
    jugadores = pd.read_csv(os.path.join(base_path, "jugadores.csv"))
    print(f"✅ Jugadores cargados: {len(jugadores)} filas")

    def leer_temp(suffix, label):
        file_p = os.path.join(base_path, f"partidos_{suffix}.csv")
        file_pj = os.path.join(base_path, f"partido_jugadores_{suffix}.csv")

        print(f"🔍 Buscando: partidos_{suffix}.csv y partido_jugadores_{suffix}.csv")

        if os.path.exists(file_p) and os.path.exists(file_pj):
            p = pd.read_csv(file_p)
            pj = pd.read_csv(file_pj)
            p["temporada"] = label
            pj["temporada"] = label
            print(f"✅ Temporada {label}: {len(p)} partidos, {len(pj)} PJ")
            return p, pj
        else:
            print(f"❌ Archivos {suffix} NO encontrados")
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

@st.cache_data
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

@st.cache_data
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
    clas = clas.sort_values(
        by=["porcentaje_victorias", "diferencia_juegos", "juegos_ganados"],
        ascending=False
    ).reset_index(drop=True)
    clas.index = clas.index + 1
    return clas

@st.cache_data
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
        ).reset_index()
        clas_j["diferencia_juegos"] = clas_j["juegos_ganados"] - clas_j["juegos_perdidos"]
        clas_j["porcentaje_victorias"] = (clas_j["victorias"] / clas_j["partidos_jugados"] * 100).round(2)
        clas_j = clas_j.sort_values(
            by=["porcentaje_victorias", "diferencia_juegos", "juegos_ganados"], ascending=False
        )
        clas_j["rank"] = range(1, len(clas_j) + 1)
        clas_j["hasta_jornada"] = j
        clasificacion_jornadas.append(clas_j)
    return pd.concat(clasificacion_jornadas, ignore_index=True)

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def construir_features_ml(df_hist):
    """
    Para cada partido histórico, calcula las features de ambos equipos
    usando SOLO datos anteriores a ese partido (no data leakage).
    Devuelve X (features) e y (quien ganó: 1 = equipo1, 0 = equipo2).
    """
    partidos_ordenados = df_hist.drop_duplicates("id_partido").copy()
    partidos_ordenados["_id_num"] = (
        partidos_ordenados["id_partido"].astype(str).str.split("_").str[0].astype(int)
    )
    partidos_ordenados = partidos_ordenados.sort_values(["id_jornada", "_id_num"])
 
    filas = []
    ids_vistos = []
 
    for _, partido_row in partidos_ordenados.iterrows():
        pid = partido_row["id_partido"]
        jornada = partido_row["id_jornada"]
 
        jugadores_partido = df_hist[df_hist["id_partido"] == pid]
        eq1 = jugadores_partido[jugadores_partido["equipo"] == 1]["nombre"].tolist()
        eq2 = jugadores_partido[jugadores_partido["equipo"] == 2]["nombre"].tolist()
 
        if len(eq1) != 2 or len(eq2) != 2:
            continue
 
        # Datos ANTERIORES a este partido (excluimos el actual)
        df_antes = df_hist[df_hist["id_partido"].isin(ids_vistos)]
 
        def winrate(nombre):
            sub = df_antes[df_antes["nombre"] == nombre]
            if len(sub) == 0:
                return 0.5
            return sub["victoria"].mean()
 
        def forma_reciente(nombre, n=5):
            sub = df_antes[df_antes["nombre"] == nombre].copy()
            sub["_id_num"] = sub["id_partido"].astype(str).str.split("_").str[0].astype(int)
            sub = sub.sort_values(["id_jornada", "_id_num"]).tail(n)
            if len(sub) == 0:
                return 0.5
            return sub["victoria"].mean()
 
        def winrate_pareja(j1, j2):
            sub1 = df_antes[df_antes["nombre"] == j1][["id_partido", "equipo", "victoria"]]
            sub2 = df_antes[df_antes["nombre"] == j2][["id_partido", "equipo"]]
            juntos = sub1.merge(sub2, on=["id_partido", "equipo"])
            if len(juntos) == 0:
                return 0.5
            return juntos["victoria"].mean()
 
        def h2h(atacantes, defensores):
            """% victorias de atacantes contra defensores en enfrentamientos directos"""
            wins, total = 0, 0
            for a in atacantes:
                for d in defensores:
                    sub_a = df_antes[(df_antes["nombre"] == a) & (df_antes["equipo"] == 1)]
                    sub_d = df_antes[(df_antes["nombre"] == d) & (df_antes["equipo"] == 2)]
                    enf = sub_a.merge(sub_d, on="id_partido", suffixes=("_a", "_d"))
                    wins += enf["victoria_a"].sum()
                    total += len(enf)
 
                    sub_a2 = df_antes[(df_antes["nombre"] == a) & (df_antes["equipo"] == 2)]
                    sub_d2 = df_antes[(df_antes["nombre"] == d) & (df_antes["equipo"] == 1)]
                    enf2 = sub_a2.merge(sub_d2, on="id_partido", suffixes=("_a", "_d"))
                    wins += enf2["victoria_a"].sum()
                    total += len(enf2)
 
            return wins / total if total > 0 else 0.5
 
        def diff_juegos_norm(nombre):
            sub = df_antes[df_antes["nombre"] == nombre]
            if len(sub) == 0:
                return 0.0
            diff = (sub["juegos_ganados"].sum() - sub["juegos_perdidos"].sum())
            total = sub["juegos_ganados"].sum() + sub["juegos_perdidos"].sum()
            return diff / total if total > 0 else 0.0
 
        # ── Calcular features ──
        wr_eq1 = (winrate(eq1[0]) + winrate(eq1[1])) / 2
        wr_eq2 = (winrate(eq2[0]) + winrate(eq2[1])) / 2
 
        forma_eq1 = (forma_reciente(eq1[0]) + forma_reciente(eq1[1])) / 2
        forma_eq2 = (forma_reciente(eq2[0]) + forma_reciente(eq2[1])) / 2
 
        pareja_eq1 = winrate_pareja(eq1[0], eq1[1])
        pareja_eq2 = winrate_pareja(eq2[0], eq2[1])
 
        h2h_eq1 = h2h(eq1, eq2)
        h2h_eq2 = h2h(eq2, eq1)
 
        diff_eq1 = (diff_juegos_norm(eq1[0]) + diff_juegos_norm(eq1[1])) / 2
        diff_eq2 = (diff_juegos_norm(eq2[0]) + diff_juegos_norm(eq2[1])) / 2
 
        fila = {
            # Diferencias entre equipos (el modelo aprende de estas diferencias)
            "wr_diff":      wr_eq1 - wr_eq2,
            "forma_diff":   forma_eq1 - forma_eq2,
            "pareja_diff":  pareja_eq1 - pareja_eq2,
            "h2h_diff":     h2h_eq1 - h2h_eq2,
            "diff_juegos":  diff_eq1 - diff_eq2,
            # Target
            "y": 1 if partido_row["equipo_ganador"] == 1 else 0
        }
        filas.append(fila)
        ids_vistos.append(pid)
 
    df_features = pd.DataFrame(filas)
    return df_features
 
 
@st.cache_data
def entrenar_modelo(df_features):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    import numpy as np
 
    feature_cols = ["wr_diff", "forma_diff", "pareja_diff", "h2h_diff", "diff_juegos"]
    X = df_features[feature_cols].values
    y = df_features["y"].values
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    ])
 
    # Cross-validation para estimar accuracy real
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    pipeline.fit(X, y)
 
    coefs = pipeline.named_steps["clf"].coef_[0]
 
    return pipeline, scores.mean(), scores.std(), feature_cols, coefs
 
 
def predecir_partido(pipeline, df_hist, eq1, eq2, feature_cols):
    """Calcula features con TODO el histórico disponible y predice."""
    import numpy as np
 
    def winrate(nombre):
        sub = df_hist[df_hist["nombre"] == nombre]
        return sub["victoria"].mean() if len(sub) > 0 else 0.5
 
    def forma_reciente(nombre, n=5):
        sub = df_hist[df_hist["nombre"] == nombre].copy()
        sub["_id_num"] = sub["id_partido"].astype(str).str.split("_").str[0].astype(int)
        sub = sub.sort_values(["id_jornada", "_id_num"]).tail(n)
        return sub["victoria"].mean() if len(sub) > 0 else 0.5
 
    def winrate_pareja(j1, j2):
        sub1 = df_hist[df_hist["nombre"] == j1][["id_partido", "equipo", "victoria"]]
        sub2 = df_hist[df_hist["nombre"] == j2][["id_partido", "equipo"]]
        juntos = sub1.merge(sub2, on=["id_partido", "equipo"])
        return juntos["victoria"].mean() if len(juntos) > 0 else 0.5
 
    def h2h(atacantes, defensores):
        wins, total = 0, 0
        for a in atacantes:
            for d in defensores:
                sub_a = df_hist[(df_hist["nombre"] == a) & (df_hist["equipo"] == 1)]
                sub_d = df_hist[(df_hist["nombre"] == d) & (df_hist["equipo"] == 2)]
                enf = sub_a.merge(sub_d, on="id_partido", suffixes=("_a", "_d"))
                wins += enf["victoria_a"].sum()
                total += len(enf)
                sub_a2 = df_hist[(df_hist["nombre"] == a) & (df_hist["equipo"] == 2)]
                sub_d2 = df_hist[(df_hist["nombre"] == d) & (df_hist["equipo"] == 1)]
                enf2 = sub_a2.merge(sub_d2, on="id_partido", suffixes=("_a", "_d"))
                wins += enf2["victoria_a"].sum()
                total += len(enf2)
        return wins / total if total > 0 else 0.5
 
    def diff_juegos_norm(nombre):
        sub = df_hist[df_hist["nombre"] == nombre]
        if len(sub) == 0:
            return 0.0
        diff = sub["juegos_ganados"].sum() - sub["juegos_perdidos"].sum()
        total = sub["juegos_ganados"].sum() + sub["juegos_perdidos"].sum()
        return diff / total if total > 0 else 0.0
 
    wr_eq1 = (winrate(eq1[0]) + winrate(eq1[1])) / 2
    wr_eq2 = (winrate(eq2[0]) + winrate(eq2[1])) / 2
    forma_eq1 = (forma_reciente(eq1[0]) + forma_reciente(eq1[1])) / 2
    forma_eq2 = (forma_reciente(eq2[0]) + forma_reciente(eq2[1])) / 2
    pareja_eq1 = winrate_pareja(eq1[0], eq1[1])
    pareja_eq2 = winrate_pareja(eq2[0], eq2[1])
    h2h_eq1 = h2h(eq1, eq2)
    h2h_eq2 = h2h(eq2, eq1)
    diff_eq1 = (diff_juegos_norm(eq1[0]) + diff_juegos_norm(eq1[1])) / 2
    diff_eq2 = (diff_juegos_norm(eq2[0]) + diff_juegos_norm(eq2[1])) / 2
 
    X_pred = np.array([[
        wr_eq1 - wr_eq2,
        forma_eq1 - forma_eq2,
        pareja_eq1 - pareja_eq2,
        h2h_eq1 - h2h_eq2,
        diff_eq1 - diff_eq2,
    ]])
 
    prob = pipeline.predict_proba(X_pred)[0]  # [prob_eq2_gana, prob_eq1_gana]
 
    desglose = {
        "Winrate histórico": (wr_eq1, wr_eq2),
        "Forma reciente (5 partidos)": (forma_eq1, forma_eq2),
        "Rendimiento como pareja": (pareja_eq1, pareja_eq2),
        "Head-to-head directo": (h2h_eq1, h2h_eq2),
        "Diferencia de juegos": (
            (diff_eq1 + 1) / 2,  # normalizar a [0,1] para mostrar
            (diff_eq2 + 1) / 2
        ),
    }
 
    return prob[1], prob[0], desglose  # prob_eq1, prob_eq2, desglose


PESOS = {
    "H2H pareja exacta":         0.30,
    "H2H individual":            0.10,
    "Rendimiento como pareja":   0.22,
    "Forma reciente (5 partidos)":0.20,
    "Winrate histórico":         0.13,
    "Diferencia de juegos":      0.05,
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
 
 
def _h2h(df, eq1, eq2):
    """
    - >= 2 partidos exactos  → solo pareja exacta
    -  = 1 partido exacto    → 60% exacto + 40% individual
    -  = 0 partidos exactos  → solo individual
    """
    exacto, n = _h2h_pareja_exacta(df, eq1, eq2)
    individual = _h2h_individual(df, eq1, eq2)
 
    if n >= 2:
        return exacto
    elif n == 1:
        return 0.6 * exacto + 0.4 * individual
    else:
        return individual
 
 
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
    elif n_partidos <= 3:
        return peso_base * 0.7
    else:
        return peso_base
 
def calcular_score(df_ref, eq1, eq2):
    vals = {}

    exacto_eq1, n = _h2h_pareja_exacta(df_ref, eq1, eq2)
    exacto_eq2, _ = _h2h_pareja_exacta(df_ref, eq2, eq1)

    # 🔥 Peso dinámico H2H pareja exacta
    peso_base_h2h = PESOS.get("H2H pareja exacta", 0.0)
    peso_h2h_real = _peso_h2h_dinamico(n, peso_base_h2h)

    # Solo añadimos si hay datos suficientes
    if peso_h2h_real > 0:
        vals["H2H pareja exacta"] = (exacto_eq1, exacto_eq2)

    # RESTO IGUAL
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

        # 👇 aquí está la magia
        if feature == "H2H pareja exacta":
            peso = peso_h2h_real
        else:
            peso = PESOS.get(feature, 0.0)

        total = v1 + v2
        p1, p2 = (v1 / total, v2 / total) if total > 0 else (0.5, 0.5)

        score_eq1 += peso * p1
        score_eq2 += peso * p2

    total_score = score_eq1 + score_eq2

    # Protección extra (por si todos los pesos son 0)
    if total_score == 0:
        return 0.5, 0.5, vals

    score_eq1 /= total_score
    score_eq2 /= total_score

    return score_eq1, score_eq2, vals
 
 
@st.cache_data
def validacion_historica(df_hist):
    """
    Aplica el modelo a cada partido histórico usando solo datos anteriores.
    Devuelve (accuracy, n_partidos_validados).
    """
    partidos_ord = df_hist.drop_duplicates("id_partido").copy()
    partidos_ord["_n"] = partidos_ord["id_partido"].astype(str).str.split("_").str[0].astype(int)
    partidos_ord = partidos_ord.sort_values(["id_jornada", "_n"])
 
    correctos, total, ids_vistos = 0, 0, []
 
    for _, row in partidos_ord.iterrows():
        pid = row["id_partido"]
        jugadores_p = df_hist[df_hist["id_partido"] == pid]
        eq1 = jugadores_p[jugadores_p["equipo"] == 1]["nombre"].tolist()
        eq2 = jugadores_p[jugadores_p["equipo"] == 2]["nombre"].tolist()
 
        if len(eq1) != 2 or len(eq2) != 2:
            ids_vistos.append(pid)
            continue
 
        df_antes = df_hist[df_hist["id_partido"].isin(ids_vistos)]
 
        if len(df_antes) < 8:
            ids_vistos.append(pid)
            continue
 
        s1, s2, _ = calcular_score(df_antes, eq1, eq2)
        prediccion = 1 if s1 >= s2 else 2
        if prediccion == row["equipo_ganador"]:
            correctos += 1
        total += 1
        ids_vistos.append(pid)
 
    return (correctos / total if total > 0 else 0.0), total

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
            "💻 Predictor"
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

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
if seccion == "🏆 Clasificación":
    st.markdown("## 🏆 Clasificación General")

    # Métricas top
    lider = clasificacion.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥇 Líder", lider["nombre"])
    col2.metric("📊 Partidos jugados", df["id_partido"].nunique())
    col3.metric("🗓️ Jornadas", df["id_jornada"].nunique())
    col4.metric("👥 Jugadores", len(clasificacion))

    st.divider()

    # Tabla de clasificación estilizada
    tabla = clasificacion[[
        "nombre", "partidos_jugados", "victorias", "derrotas",
        "diferencia_juegos", "juegos_ganados", "juegos_perdidos",
        "porcentaje_victorias", "jornadas"
    ]].copy()

    tabla.columns = ["Jugador", "PJ", "V", "D", "+/-", "JG", "JP", "% V", "Jornadas"]

    # 🔥 FUNCIONES DE COLOR SUAVE TIPO GRADIENTE
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
    st.markdown("## 📈 Evolución del Ranking por Jornada")

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colores = plt.cm.tab10.colors
    nombres = ranking_jornada["nombre"].unique()

    for i, nombre in enumerate(nombres):
        datos = ranking_jornada[ranking_jornada["nombre"] == nombre].sort_values("hasta_jornada")
        ax.plot(datos["hasta_jornada"], datos["rank"],
                marker="o", linewidth=2.5, markersize=6,
                color=colores[i % len(colores)], label=nombre)

    ax.invert_yaxis()
    ax.set_xlabel("Jornada", color="#8b949e")
    ax.set_ylabel("Posición", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.set_xticks(sorted(ranking_jornada["hasta_jornada"].unique()))
    ax.set_yticks(range(1, len(nombres) + 1))
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
    jornada_sel = st.selectbox("Selecciona jornada", jornadas)

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
    ].drop_duplicates(subset=["id_partido"])

    # Agrupar por partido
    for partido_id, grupo in partidos_jornada.groupby("id_partido"):

        # Obtener equipos
        jugadores = grupo["nombre"].tolist()

        # Dividir en 2 equipos según equipo (0/1)
        equipo1 = grupo[grupo["equipo"] == 1]["nombre"].tolist()
        equipo2 = grupo[grupo["equipo"] == 2]["nombre"].tolist()

        if not equipo1 or not equipo2:
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
            f"**{partido_id}. {eq1} vs {eq2}: {g1}-{g2}**  \n"
            f"_{texto_ganador}_"
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: PERFIL JUGADOR
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "👤 Perfil Jugador":
    st.markdown("## 👤 Perfil de Jugador")
    nombre_sel = st.selectbox(
        "Selecciona jugador",
        sorted(clasificacion["nombre"].tolist(), key=lambda x: quitar_acentos(x).lower())
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
        datos_jugador = ranking_jornada[ranking_jornada["nombre"] == nombre_sel].sort_values("hasta_jornada")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        ax.plot(datos_jugador["hasta_jornada"], datos_jugador["rank"],
                marker="o", linewidth=2.5, color="#238636", markersize=7)
        ax.fill_between(datos_jugador["hasta_jornada"], datos_jugador["rank"],
                        alpha=0.15, color="#238636")
        ax.invert_yaxis()
        ax.set_xlabel("Jornada", color="#8b949e")
        ax.set_ylabel("Posición", color="#8b949e")
        ax.set_xticks(sorted(datos_jugador["hasta_jornada"].unique()))
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
        jugadores_lista = sorted(df["nombre"].unique())
        col1, col2 = st.columns(2)
        j1 = col1.selectbox("Jugador 1", jugadores_lista, key="enf_j1")
        j2 = col2.selectbox("Jugador 2", [j for j in jugadores_lista if j != j1], key="enf_j2")
 
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
                st.divider()
                c1, c2, c3 = st.columns([2, 1, 2])
                c1.markdown(f"### {p1_name}")
                c1.metric("Victorias", v1, f"{v1/total*100:.1f}%")
                c1.metric("Juegos ganados", jg1)
                c2.markdown("### VS", unsafe_allow_html=False)
                c2.metric("Partidos", total)
                c3.markdown(f"### {p2_name}")
                c3.metric("Victorias", v2, f"{v2/total*100:.1f}%")
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
            jugadores_unicos = sorted(df["nombre"].unique())
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

    tab1, tab2, tab3 = st.tabs(["📊 V/D por jugador", "📈 % Victorias por jornada", "🎯 Diferencia juegos"])

    with tab1:
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

    with tab2:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        colores = plt.cm.tab10.colors
        for i, nombre in enumerate(sorted(df["nombre"].unique())):
            datos = ranking_jornada[ranking_jornada["nombre"] == nombre].sort_values("hasta_jornada")
            ax.plot(datos["hasta_jornada"], datos["porcentaje_victorias"],
                    marker="o", linewidth=2, color=colores[i % len(colores)], label=nombre)
        ax.set_xlabel("Jornada", color="#8b949e")
        ax.set_ylabel("% Victorias", color="#8b949e")
        ax.set_xticks(sorted(ranking_jornada["hasta_jornada"].unique()))
        ax.tick_params(colors="#8b949e")
        ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
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


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "💻 Predictor":
    st.markdown("## 💻 Predictor de Partido")
    st.caption("Scoring ponderado · H2H usa pareja exacta con fallback individual · Validado con datos históricos")
 
    df_hist = df_completo.copy()
 
    with st.spinner("⚙️ Validando modelo con datos históricos..."):
        acc, n_validados = validacion_historica(df_hist)
 
    nombres_todos = sorted(df_hist["nombre"].unique())
 
    st.divider()
 
    col_eq1, col_vs, col_eq2 = st.columns([5, 1, 5])
 
    with col_eq1:
        st.markdown("#### 🔵 Equipo 1")
        j1a = st.selectbox("Jugador A", nombres_todos, key="j1a")
        j1b = st.selectbox("Jugador B", [j for j in nombres_todos if j != j1a], key="j1b")
 
    with col_vs:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### VS")
 
    with col_eq2:
        st.markdown("#### 🔴 Equipo 2")
        disp2 = [j for j in nombres_todos if j not in [j1a, j1b]]
        j2a = st.selectbox("Jugador C", disp2, key="j2a")
        j2b = st.selectbox("Jugador D", [j for j in disp2 if j != j2a], key="j2b")
 
    eq1 = [j1a, j1b]
    eq2 = [j2a, j2b]
 
    st.divider()
 
    if st.button("🎯 Calcular predicción", use_container_width=True, type="primary"):
 
        prob_eq1, prob_eq2, desglose = calcular_score(df_hist, eq1, eq2)
 
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
 
        _, n_exactos = _h2h_pareja_exacta(df_hist, eq1, eq2)

        for feature, peso in PESOS.items():

            # 🔥 SI NO EXISTE → SKIP
            if feature not in desglose:
                continue

            v1, v2 = desglose[feature]

            ventaja1 = v1 >= v2
            color1 = "#1f6feb" if ventaja1 else "#8b949e"
            color2 = "#da3633" if not ventaja1 else "#8b949e"
            icono = "🔵" if ventaja1 else "🔴"

            bar_eq1 = v1 / (v1 + v2) * 100 if (v1 + v2) > 0 else 50
            bar_eq2 = 100 - bar_eq1

            # 🔥 INFO EXTRA H2H EXACTO
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
                <span style="color:#238636;font-weight:700;font-size:1rem;">{int(peso*100)}%</span>
            </div>
            """, unsafe_allow_html=True)