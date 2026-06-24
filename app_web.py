import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np

# 1. Configuración de pantalla móvil
st.set_page_config(page_title="🤖 RestoDSS Premium", page_icon="🍴", layout="centered")

# Estilos de diseño móvil avanzados
st.markdown("""
    <style>
    .main-title { font-size: 30px !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: 0px; }
    .subtitle { font-size: 15px; color: #A3A3A3; text-align: center; margin-bottom: 25px; }
    .card { background-color: #1E222B; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-bottom: 12px; }
    .card-ia { border-left: 5px solid #00D1B2; }
    .card-title { font-size: 18px; font-weight: bold; color: #FFFFFF; margin-bottom: 3px; }
    .card-meta { font-size: 14px; color: #E0E0E0; }
    .highlight { color: #00D1B2; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. Carga, Entrenamiento Matemático y Predicciones en Caché (Mismo algoritmo que tu model.py)
@st.cache_resource
def entrenar_sistema_ia():
    df = pd.read_csv('restaurant_data.csv')
    df.columns = df.columns.str.strip().str.lower()
    
    # Construcción de la matriz y cálculo de SVD
    matriz_usuario_resto = df.pivot_table(index='customer_id', columns='restaurant_id', values='rating')
    medias_usuarios = matriz_usuario_resto.mean(axis=1)
    matriz_normalizada = matriz_usuario_resto.sub(medias_usuarios, axis=0).fillna(0)
    
    n_componentes = min(5, matriz_normalizada.shape[0] - 1, matriz_normalizada.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    
    matriz_reducida = svd.fit_transform(matriz_normalizada)
    matriz_predicha = np.dot(matriz_reducida, svd.components_)
    
    df_pred_final = pd.DataFrame(matriz_predicha, index=matriz_normalizada.index, columns=matriz_normalizada.columns)
    df_pred_final = df_pred_final.add(medias_usuarios, axis=0)
    
    conteos_reviews = df['restaurant_id'].value_counts().to_dict()
    return df, df_pred_final, conteos_reviews

df, df_predicciones_final, conteos_reviews = entrenar_sistema_ia()

# Limpieza y orden de cocinas para los desplegables táctiles
df['cuisine_clean'] = df['cuisine'].astype(str).str.strip().str.lower()
cocinas_disponibles = sorted(df['cuisine_clean'].str.capitalize().unique())

# --- INTERFAZ DE USUARIO ---
st.markdown('<p class="main-title">🤖 AI-RestoDSS</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predicciones gastronómicas personalizadas con Scikit-Learn SVD</p>', unsafe_allow_html=True)

opcion_menu = st.radio("¿Qué buscas hoy?", ["Recomendaciones por Cocina", "Mi Perfil Personalizado (IA)"])
st.write("---")

if opcion_menu == "Recomendaciones por Cocina":
    st.subheader("📋 Top Recomendaciones Generales")
    cocina_sel = st.selectbox("Selecciona un tipo de cocina:", cocinas_disponibles)
    
    filtrados = df[df['cuisine_clean'] == cocina_sel.lower()]
    top_5 = filtrados.drop_duplicates(subset=['restaurant_id']).sort_values(by='rating', ascending=False).head(5)
    
    for _, fila in top_5.iterrows():
        st.markdown(f"""
            <div class="card">
                <div class="card-title">🍴 {fila['restaurant_name']}</div>
                <div class="card-meta">⭐ Puntuación: {fila['rating']} | 💬 Reseñas en sistema: {conteos_reviews.get(fila['restaurant_id'], 0)}</div>
            </div>
        """, unsafe_allow_html=True)

elif opcion_menu == "Mi Perfil Personalizado (IA)":
    st.subheader("🔮 Recomendador Inteligente")
    
    id_usuario = st.number_input("Ingresa tu ID de Cliente:", min_value=1, value=546, step=1)
    cocina_fav = st.selectbox("¿Qué tipo de comida te apetece?", cocinas_disponibles)
    
    if st.button("🔮 Generar Recomendación con IA", use_container_width=True):
        restaurantes_cocina = df[df['cuisine_clean'] == cocina_fav.lower()]
        lista_restaurantes = restaurantes_cocina.drop_duplicates(subset=['restaurant_id'])
        
        if lista_restaurantes.empty:
            st.warning(f"⚠️ No hay restaurantes registrados de tipo: {cocina_fav}")
        else:
            predicciones_usuario = []
            for _, fila in lista_restaurantes.iterrows():
                rid = fila['restaurant_id']
                
                if id_usuario in df_predicciones_final.index and rid in df_predicciones_final.columns:
                    nota_estimada = df_predicciones_final.loc[id_usuario, rid]
                else:
                    nota_estimada = fila['rating']
                    
                nota_estimada = max(1.0, min(5.0, nota_estimada))
                
                predicciones_usuario.append({
                    'name': fila['restaurant_name'],
                    'rating': nota_estimada,
                    'reviews': conteos_reviews.get(rid, 0)
                })
                
            df_res = pd.DataFrame(predicciones_usuario).sort_values(by='rating', ascending=False).head(5)
            
            st.success(f"🎯 Recomendaciones exclusivas para el Cliente {id_usuario}:")
            for _, fila in df_res.iterrows():
                st.markdown(f"""
                    <div class="card card-ia">
                        <div class="card-title">🎯 {fila['name']}</div>
                        <div class="card-meta">✨ Compatibilidad IA: <span class="highlight">{round(fila['rating'], 2)} ⭐</span> | 💬 Reseñas: {fila['reviews']}</div>
                    </div>
                """, unsafe_allow_html=True)