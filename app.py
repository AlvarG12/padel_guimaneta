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
    for nombre in df["nombre"].unique():
        df_j = df[df["nombre"] == nombre].sort_values(by=["id_jornada", "id_partido"])
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

    for nombre_jugador in df['nombre'].unique():
        df_jugador = df[df['nombre'] == nombre_jugador].sort_values(by=['id_jornada', 'id_partido'])

        # 🔥 RACHAS DE VICTORIAS
        current_racha = 0
        current_partidos = []

        for _, row in df_jugador.iterrows():
            if row['victoria']:
                current_racha += 1
                partido_limpio = str(row['id_partido']).split("_")[0]
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
            if not row['victoria']:
                current_racha += 1
                partido_limpio = str(row['id_partido']).split("_")[0]
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
            "📊 Detalle",
            "👤 Perfil Jugador",
            "⚔️ Enfrentamientos",
            "🤝 Parejas",
            "🔥 Rachas",
            "📊 Gráficas"
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
elif seccion == "📊 Detalle":
    st.markdown("## 📊 Detalle de Jornada")

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