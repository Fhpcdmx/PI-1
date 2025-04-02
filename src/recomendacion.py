import pandas as pd
from fastapi import FastAPI, HTTPException
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

peliculas_df = pd.read_csv('data/movies_dataset.csv', encoding='latin1')


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(peliculas_df['content'].fillna(''))

# Entrenar modelo KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_matrix)

# Crear índice para búsqueda rápida
indices = pd.Series(peliculas_df.index, index=peliculas_df['title']).drop_duplicates()

app = FastAPI()

@app.get("/")
def home():
    """
    Ruta raíz de la API.
    Retorna un mensaje de bienvenida en formato JSON.
    """
    return {"message": "API de Recomendación de Películas"}

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    """
    Obtiene la cantidad de películas estrenadas en un mes específico.
    
    Parámetros:
        mes (str): Nombre del mes en español (Ej: 'Enero', 'Febrero').
    
    Retorno:
        dict: Mensaje con la cantidad de películas estrenadas en el mes indicado.
    """
    meses_espanol = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                     'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    if mes not in meses_espanol:
        raise HTTPException(status_code=400, detail="Mes no válido")

    mes_numero = meses_espanol.index(mes) + 1
    peliculas_df['release_date'] = pd.to_datetime(peliculas_df['release_date'], errors='coerce')
    peliculas_mes = peliculas_df[peliculas_df['release_date'].dt.month == mes_numero]
    
    return {"mensaje": f"{peliculas_mes.shape[0]} películas fueron estrenadas en {mes}"}

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    """
    Obtiene la cantidad de películas estrenadas en un día de la semana específico.

    Parámetros:
        dia (str): Nombre del día en español (Ej: 'lunes', 'martes').

    Retorno:
        dict: Mensaje con la cantidad de películas estrenadas en el día indicado.
    """
    dias_esp = {
        'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles',
        'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sabado', 'Sunday': 'domingo'
    }

    if dia.lower() not in dias_esp.values():
        raise HTTPException(status_code=400, detail="Día no válido")

    peliculas_df['release_date'] = pd.to_datetime(peliculas_df['release_date'], errors='coerce')
    peliculas_df['dia_semana'] = peliculas_df['release_date'].dt.day_name().map(dias_esp).str.lower()
    peliculas_dia = peliculas_df[peliculas_df['dia_semana'] == dia.lower()]
    
    return {"mensaje": f"{peliculas_dia.shape[0]} películas fueron estrenadas un {dia}"}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    """
    Obtiene el score y los datos generales de una pelicula a partir de el titulo.

    Parámetros:
        dia (str): titulo de la pelicula.

    Retorno:
        dict: titulo de la pelicula, año de esteno y score.
    """
    pelicula = peliculas_df[peliculas_df['title'].str.contains(titulo, case=False, na=False)]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    pelicula_info = pelicula.iloc[0]
    return {
        "titulo": pelicula_info['title'],
        "año_estreno": int(pelicula_info['release_year']),
        "score": float(pelicula_info['popularity'])
    }

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    """
    Obtiene la cantidad de votos y el promedio de votos de una película si tiene al menos 2000 valoraciones.

    Parámetros:
        titulo (str): Título de la película.

    Retorno:
        dict: Título de la película, cantidad de votos y promedio de votos.
    """
    pelicula = peliculas_df[peliculas_df['original_title'].str.contains(titulo, case=False, na=False)]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    pelicula_info = pelicula.iloc[0]
    
    if pelicula_info['vote_count'] < 2000:
        raise HTTPException(status_code=400, detail="La película no tiene al menos 2000 valoraciones")

    return {
        "titulo": pelicula_info['original_title'],
        "cantidad_votos": int(pelicula_info['vote_count']),
        "promedio_votos": float(pelicula_info['vote_average'])
    }

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    """
    Obtiene la cantidad de películas en las que ha participado un actor y el promedio de retorno.

    Parámetros:
        nombre_actor (str): Nombre del actor.

    Retorno:
        dict: Nombre del actor, cantidad de películas y promedio de retorno.
    """
    actores_df = pd.read_csv('data/cast.csv')
    actor_peliculas = actores_df[actores_df['actor_name'].str.contains(nombre_actor, case=False, na=False)]
    
    if actor_peliculas.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")

    peliculas_actor = pd.merge(actor_peliculas, peliculas_df, on='id', how='left')
    cantidad_peliculas = peliculas_actor.shape[0]
    promedio_retorno = peliculas_actor['revenue'].mean()

    return {
        "actor": nombre_actor,
        "cantidad_peliculas": cantidad_peliculas,
        "promedio_retorno": round(promedio_retorno, 2)
    }

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    """
    Obtiene las películas dirigidas por un director, incluyendo título, fecha de lanzamiento, ingresos,
    compañías de producción, países de producción y promedio de votos.

    Parámetros:
        nombre_director (str): Nombre del director.

    Retorno:
        dict: Nombre del director y una lista de películas con su información.
    """
    directores_df = pd.read_csv('data/crew.csv')
    director_peliculas = directores_df[
        (directores_df['crew_name'].str.contains(nombre_director, case=False, na=False)) &
        (directores_df['job'] == 'Director')
    ]
    
    if director_peliculas.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")

    peliculas_director = pd.merge(director_peliculas, peliculas_df, on='id', how='left')
    peliculas_info = peliculas_director[['original_title', 'release_date', 'revenue', 'production_companies', 'production_countries', 'vote_average']].to_dict(orient='records')

    return {
        "director": nombre_director,
        "peliculas": peliculas_info
    }

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    """
    Genera recomendaciones de películas similares a la película indicada basado en el genero titulo y popularidad.

    Parámetros:
        titulo (str): Título de la película para generar recomendaciones.

    Retorno:
        dict: Lista de títulos de películas recomendadas.
    """
    if titulo not in indices:
        raise HTTPException(status_code=404, detail="La película no se encuentra en la base de datos.")

    idx = indices[titulo]
    distances, indices_nn = knn.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    movie_indices = indices_nn[0][1:6]
    
    return {"recomendaciones": peliculas_df['title'].iloc[movie_indices].tolist()}

# Para correr el servidor con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
