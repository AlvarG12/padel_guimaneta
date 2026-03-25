import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Configuración de la página
st.set_page_config(page_title="Padel Guimaneta 🏆", layout="wide")

# --- TÍTULO Y SELECTOR ---
st.title("🏆 PADEL GUIMANETA")

# Selector de temporada en la barra lateral
temporada = st.sidebar.selectbox(
    "Selecciona la Temporada",
    ["24/25", "25/26"]
)

# Convertir selección a formato de archivo
suffix = temporada.replace("/", "_")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data(suffix):
    # Ajusta los nombres de tus archivos según tu carpeta /data
    jugadores = pd.read_csv("data/jugadores.csv")
    partidos = pd.read_csv(f"data/partidos_{suffix}.csv")
    pj = pd.read_csv(f"data/partido_jugadores_{suffix}.csv")
    
    # Lógica de cálculo de ganador
    partidos["equipo_ganador"] = partidos.apply(
        lambda x: 1 if x["juegos_equipo1"] > x["juegos_equipo2"] else 2, axis=1
    )
    
    # Merge principal
    df = pj.merge(jugadores, on="id_jugador").merge(partidos, on="id_partido")
    df["victoria"] = df["equipo"] == df["equipo_ganador"]
    
    # Juegos ganados/perdidos
    df["juegos_ganados"] = df.apply(lambda x: x["juegos_equipo1"] if x["equipo"]==1 else x["juegos_equipo2"], axis=1)
    df["juegos_perdidos"] = df.apply(lambda x: x["juegos_equipo2"] if x["equipo"]==1 else x["juegos_equipo1"], axis=1)
    
    return df, jugadores, partidos, pj

try:
    df, jugadores, partidos, pj = load_data(suffix)
except Exception as e:
    st.error(f"Error al cargar los datos de la temporada {temporada}. Revisa que los archivos existan en la carpeta /data")
    st.stop()

# --- PESTAÑAS DE LA APP ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Clasificación", "🎾 Partidos", "🔥 Rachas", "📈 Estadísticas"])

with tab1:
    st.header(f"Clasificación General - Temporada {temporada}")
    
    clas_total = df.groupby("nombre").agg(
        PJ=("id_partido", "count"),
        V=("victoria", "sum"),
        JG=("juegos_ganados", "sum"),
        JP=("juegos_perdidos", "sum"),
        Jornadas=("id_jornada", "nunique")
    ).reset_index()
    
    clas_total["DIF"] = clas_total["JG"] - clas_total["JP"]
    clas_total["D"] = clas_total["PJ"] - clas_total["V"]
    clas_total["% Victoria"] = (clas_total["V"] / clas_total["PJ"] * 100).round(2)
    
    clas_total = clas_total.sort_values(by=["% Victoria", "DIF"], ascending=False)
    st.dataframe(clas_total, use_container_width=True)

    # Gráfico de barras de diferencia de juegos
    fig, ax = plt.subplots()
    sns.barplot(x="nombre", y="DIF", data=clas_total, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab2:
    st.header("Resumen de Jornadas")
    jornada_sel = st.selectbox("Selecciona Jornada", sorted(df["id_jornada"].unique()))
    
    partidos_j = partidos[partidos["id_jornada"] == jornada_sel]
    id_to_nombre = dict(zip(jugadores["id_jugador"], jugadores["nombre"]))
    
    for _, p in partidos_j.iterrows():
        p_jugadores = pj[pj["id_partido"] == p["id_partido"]]
        eq1 = p_jugadores[p_jugadores["equipo"] == 1]["id_jugador"].map(id_to_nombre).tolist()
        eq2 = p_jugadores[p_jugadores["equipo"] == 2]["id_jugador"].map(id_to_nombre).tolist()
        
        col1, col2, col3 = st.columns([2,1,2])
        col1.write(f"**{' & '.join(eq1)}**")
        col2.write(f"**{int(p['juegos_equipo1'])} - {int(p['juegos_equipo2'])}**")
        col3.write(f"**{' & '.join(eq2)}**")
        if p["comentario"]: st.caption(f"💬 {p['comentario']}")
        st.divider()

with tab3:
    st.header("Rachas Actuales")
    # Lógica de rachas simplificada para visualización
    rachas_list = []
    for player in df['nombre'].unique():
        p_matches = df[df['nombre'] == player].sort_values(by=['fecha', 'id_partido'])
        last_v = p_matches.iloc[-1]['victoria']
        count = 0
        for v in reversed(p_matches['victoria'].tolist()):
            if v == last_v: count += 1
            else: break
        rachas_list.append({"Jugador": player, "Racha": "Ganadora" if last_v else "Perdedora", "Núm": count})
    
    st.table(pd.DataFrame(rachas_list).sort_values(by="Núm", ascending=False))

with tab4:
    st.header("Análisis de Jugadores")
    jugador_stats = st.selectbox("Ver detalles de:", jugadores["nombre"].unique())
    
    # Evolución del ranking (Muestra el % de victoria a lo largo del tiempo)
    p_evol = df[df['nombre'] == jugador_stats].copy()
    p_evol['cum_victorias'] = p_evol['victoria'].cumsum()
    p_evol['partido_num'] = range(1, len(p_evol) + 1)
    p_evol['pct_evol'] = (p_evol['cum_victorias'] / p_evol['partido_num'] * 100)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(p_evol['partido_num'], p_evol['pct_evol'], marker='o')
    ax2.set_ylabel("% Victoria Acumulado")
    ax2.set_xlabel("Partidos Jugados")
    st.pyplot(fig2)