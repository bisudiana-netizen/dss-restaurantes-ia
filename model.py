import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import numpy as np

# --- Cargar los datos desde el archivo CSV ---
df = pd.read_csv('restaurant_data.csv')

# Normalizamos los nombres de las columnas por seguridad
df.columns = df.columns.str.strip().str.lower()

# --- Visualizar los primeros registros para verificar la carga correcta de datos ---
print("--- PRIMEROS REGISTROS DEL DATASET ---")
print(df.head())
print("==================================================\n")

# --- Dividir los datos en entrenamiento y prueba ---
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# --- Crear la matriz de usuario-restaurante ---
# Usamos 'customer_id' que es el nombre real en tu CSV
matriz_usuario_resto = df_train.pivot_table(index='customer_id', columns='restaurant_id', values='rating')

# Guardamos las medias de cada usuario para rellenar celdas vacías
medias_usuarios = matriz_usuario_resto.mean(axis=1)
matriz_normalizada = matriz_usuario_resto.sub(medias_usuarios, axis=0).fillna(0)

# --- Descomposición en valores singulares (SVD) ---
n_componentes = min(5, matriz_normalizada.shape[0] - 1, matriz_normalizada.shape[1] - 1)
svd = TruncatedSVD(n_components=n_componentes, random_state=42)

# --- Entrenar el modelo de SVD en los datos de entrenamiento ---
matriz_reducida = svd.fit_transform(matriz_normalizada)

# Reconstruimos la matriz completa sumando las medias originales
matriz_predicha_normalizada = np.dot(matriz_reducida, svd.components_)
df_predicciones_train = pd.DataFrame(matriz_predicha_normalizada, 
                                     index=matriz_normalizada.index, 
                                     columns=matriz_normalizada.columns)
df_predicciones_final = df_predicciones_train.add(medias_usuarios, axis=0)

# --- Predecir las calificaciones en los datos de prueba ---
calificaciones_reales = []
calificaciones_predichas = []

for _, fila in df_test.iterrows():
    uid = fila['customer_id']
    rid = fila['restaurant_id']
    real_rating = fila['rating']
    
    if uid in df_predicciones_final.index and rid in df_predicciones_final.columns:
        pred_rating = df_predicciones_final.loc[uid, rid]
    else:
        pred_rating = df_train['rating'].mean()
        
    calificaciones_reales.append(real_rating)
    calificaciones_predichas.append(pred_rating)

# --- Calcular el error cuadrático medio (RMSE) ---
rmse = np.sqrt(mean_squared_error(calificaciones_reales, calificaciones_predichas))
print(f"✅ Modelo entrenado. Error Cuadrático Medio (RMSE) en Prueba: {round(rmse, 4)} ⭐\n")

# --- Contar el número de reseñas por restaurante ---
# Contamos cuántas veces aparece cada restaurante en el CSV para simular las reseñas
conteos_reviews = df['restaurant_id'].value_counts().to_dict()


# --- FUNCION PARA RECOMENDACIONES SEGÚN EL TIPO DE COMIDA ---
def recomendar_por_tipo_comida(tipo_comida):
    """Muestra 5 recomendaciones basadas en el tipo de comida (columna 'cuisine')."""
    filtrados = df[df['cuisine'].str.lower() == tipo_comida.lower()]
    
    if filtrados.empty:
        print(f"❌ No se encontraron restaurantes del tipo: {tipo_comida}")
        return
        
    top_5 = (filtrados.drop_duplicates(subset=['restaurant_id'])
             .sort_values(by='rating', ascending=False)
             .head(5))
    
    print(f"--- TOP 5 RECOMENDACIONES DE COMIDA: {tipo_comida.upper()} ---")
    for _, fila in top_5.iterrows():
        rid = fila['restaurant_id']
        reviews = conteos_reviews.get(rid, 0)
        print(f"🍴 Nombre: {fila['restaurant_name']} | Puntuación: {fila['rating']} ⭐ | Reseñas: {reviews}")

# --- FUNCION PARA RECOMENDACIONES SEGÚN EL ID DEL USUARIO Y SU COMIDA FAVORITA ---
def recomendar_por_usuario_y_cocina(id_usuario, cocina_favorita):
    """Predice con TruncatedSVD los mejores restaurantes usando 'customer_id' y 'cuisine'."""
    restaurantes_cocina = df[df['cuisine'].str.lower() == cocina_favorita.lower()]
    lista_restaurantes = restaurantes_cocina.drop_duplicates(subset=['restaurant_id'])
    
    if lista_restaurantes.empty:
        print(f"❌ No hay restaurantes registrados bajo la cocina: {cocina_favorita}")
        return

    predicciones_usuario = []
    
    for _, fila in lista_restaurantes.iterrows():
        rid = fila['restaurant_id']
        
        if id_usuario in df_predicciones_final.index and rid in df_predicciones_final.columns:
            nota_estimada = df_predicciones_final.loc[id_usuario, rid]
        else:
            nota_estimada = fila['rating']
            
        nota_estimada = max(1.0, min(5.0, nota_estimada))
        reviews = conteos_reviews.get(rid, 0)
        
        predicciones_usuario.append({
            'restaurant_name': fila['restaurant_name'],
            'predicted_rating': nota_estimada,
            'reviews': reviews
        })
        
    df_res = pd.DataFrame(predicciones_usuario).sort_values(by='predicted_rating', ascending=False).head(5)
    
    print(f"\n--- TOP 5 PERSONALIZADO IA PARA CLIENTE {id_usuario} ({cocina_favorita.upper()}) ---")
    for _, fila in df_res.iterrows():
        print(f"🎯 {fila['restaurant_name']} | Predicción IA: {round(fila['predicted_rating'], 2)} ⭐ | Reseñas: {fila['reviews']}")


# --- PRUEBAS DE EJECUCIÓN (Ejemplos de uso) ---
if __name__ == "__main__":
    # Probamos la primera función con comida Griega (que abunda en tu archivo)
    recomendar_por_tipo_comida("Greek")
    
    # Probamos la IA con un cliente real de tu lista (ej: Cliente 546) y comida Italiana
    recomendar_por_usuario_y_cocina(id_usuario=546, cocina_favorita="Italian")