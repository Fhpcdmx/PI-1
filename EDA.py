import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carga de datos
movies = pd.read_csv('data/movies_dataset.csv' ,encoding='latin1')

# 2. Exploración de datos
print("Primeras filas del archivo:")
print(movies.head())
print("\nInformación del archivo:")
print(movies.info())
print("\nEstadísticas descriptivas:")
print(movies.describe())

# 3. Limpieza de datos
# Eliminar duplicados
movies.drop_duplicates(subset=['id'], inplace=True)

# Eliminar columnas irrelevantes para el modelo de recomendación
irrelevant_cols = ['video', 'status', 'spoken_languages', 'production_countries']
movies.drop(columns=irrelevant_cols, inplace=True)

# Convertir release_date a formato datetime
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

# Rellenar valores nulos en numéricos con la mediana
numerical_cols = ['popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
movies[numerical_cols] = movies[numerical_cols].fillna(movies[numerical_cols].median())

# 4. Análisis descriptivo
print("\nPelículas con mayor popularidad:")
print(movies[['title', 'popularity']].sort_values(by='popularity', ascending=False).head())

print("\nDistribución de la duración de las películas:")
plt.figure(figsize=(8, 4))
sns.histplot(movies['runtime'], bins=30, kde=True)
plt.title('Distribución de la duración de las películas')
plt.show()

# 5. Preprocesamiento para la recomendación
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Crear una columna combinada de características para calcular la similitud
movies['features'] = movies['original_title'] + ' ' + movies['original_language']

# Vectorización con TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['features'].fillna(''))

# Similitud del coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("\nMatriz de similitud del coseno:")
print(cosine_sim[:5, :5])
