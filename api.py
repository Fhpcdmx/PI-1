import pandas as pd
from flask import Flask, jsonify
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

peliculas_df = pd.read_csv('data/movies_dataset.csv',encoding='latin1')
# Vectorizar el contenido
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(peliculas_df['content'])

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_matrix)

# Crear índice para búsqueda rápida
indices = pd.Series(peliculas_df.index, index=peliculas_df['title']).drop_duplicates()

app = Flask(__name__)

# Función para convertir un número de mes a su nombre en español
def mes_a_espanol(mes_numero):
    meses = [
        'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
    ]
    return meses[mes_numero - 1]

@app.get("/")  
def home():
    return {"message": "API de Recomendación de Películas"}

@app.route('/cantidad_filmaciones_mes/<mes>', methods=['GET'])
def cantidad_filmaciones_mes(mes):
    # Convertir mes a número
    meses_espanol = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                     'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    if mes not in meses_espanol:
        return jsonify({"mensaje": "Mes no válido"}), 400

    mes_numero = meses_espanol.index(mes) + 1
    # Convertir 'release_date' a datetime y filtrar por el mes
    peliculas_df['release_date'] = pd.to_datetime(peliculas_df['release_date'])
    peliculas_mes = peliculas_df[peliculas_df['release_date'].dt.month == mes_numero]
    
    cantidad = peliculas_mes.shape[0]
    return jsonify({
        "mensaje": f"{cantidad} cantidad de peliculas fueron estrenadas en el mes de {mes}"
    })


@app.route('/cantidad_filmaciones_dia/<dia>', methods=['GET'])
def cantidad_filmaciones_dia(dia):
    dias_espanol = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']
    dias_esp = {
    'Monday': 'lunes',
    'Tuesday': 'martes',
    'Wednesday': 'miércoles',
    'Thursday': 'jueves',
    'Friday': 'viernes',
    'Saturday': 'sabado',
    'Sunday': 'domingo'
    }
    if dia not in dias_espanol:
        return jsonify({"mensaje": "Dia no valido"}), 400

    # Convertir 'release_date' a datetime y obtener el día de la semana
    peliculas_df['release_date'] = pd.to_datetime(peliculas_df['release_date'])
    peliculas_df['dia_semana_ingles'] = peliculas_df['release_date'].dt.day_name()
    # Convertir 'dia_semana_ingles' a dias de la semana en español
    peliculas_df['dia_semana'] = peliculas_df['dia_semana_ingles'].map(dias_esp).str.lower()
     # Convertirtimos en minusculas y evaluamos con el parametro de GET
    peliculas_dia = peliculas_df[peliculas_df['dia_semana'].str.lower() == dia]
    
    cantidad = peliculas_dia.shape[0]
    return jsonify({
        "mensaje": f"{cantidad} cantidad de peliculas fueron estrenadas en los dias {dia}"
    })
@app.route('/score_titulo/<titulo>', methods=['GET'])
def score_titulo(titulo):
    pelicula = peliculas_df[peliculas_df['title'].str.contains(titulo, case=False, na=False)]
    
    if pelicula.empty:
        return jsonify({"mensaje": "Película no encontrada"}), 404

    pelicula_info = pelicula.iloc[0]
    respuesta= jsonify({
        "titulo": pelicula_info['title'],
        "año_estreno": int(pelicula_info['release_year']),
        "score": float(pelicula_info['popularity'])
    })

    respuesta.headers['Content-Type'] = 'application/json; charset=latin1'
    return respuesta

@app.route('/votos_titulo/<titulo>', methods=['GET'])
def votos_titulo(titulo):
    pelicula = peliculas_df[peliculas_df['original_title'].str.contains(titulo, case=False, na=False)]
    
    if pelicula.empty:
        return jsonify({"mensaje": "Película no encontrada"}), 404

    pelicula_info = pelicula.iloc[0]
    
    if pelicula_info['vote_count'] < 2000:
        return jsonify({"mensaje": "La película no cumple con el mínimo de 2000 valoraciones"}), 400

    return jsonify({
        "titulo": pelicula_info['original_title'],
        "cantidad_votos": int(pelicula_info['vote_count']),
        "promedio_votos": float(pelicula_info['vote_average'])    
    })


@app.route('/get_actor/<nombre_actor>', methods=['GET'])
def get_actor(nombre_actor):
    actores_df = pd.read_csv('data/cast.csv')
    actor_peliculas = actores_df[actores_df['actor_name'].str.contains(nombre_actor, case=False, na=False)]
    
    if actor_peliculas.empty:
        return jsonify({"mensaje": "Actor no encontrado"}), 404

    # Obtener información sobre las películas de ese actor
    peliculas_actor = pd.merge(actor_peliculas, peliculas_df, on='id', how='left')
    cantidad_peliculas = peliculas_actor.shape[0]
    promedio_retorno = peliculas_actor['revenue'].mean()

    return jsonify({
        "actor": nombre_actor,
        "cantidad_peliculas": cantidad_peliculas,
        "promedio_retorno": round(promedio_retorno,2)
    })
@app.route('/get_director/<nombre_director>', methods=['GET'])
def get_director(nombre_director):
    directores_df = pd.read_csv('data/crew.csv')
    director_peliculas = directores_df[directores_df['crew_name'].str.contains(nombre_director, case=False, na=False)& 
    (directores_df['job'] == 'Director')]
    
    if director_peliculas.empty:
        return jsonify({"mensaje": "Director no encontrado"}), 404

    peliculas_director = pd.merge(director_peliculas, peliculas_df, on='id', how='left')
    
    peliculas_info = peliculas_director[['original_title', 'release_date', 'revenue', 'production_companies', 'production_countries', 'vote_average']].to_dict(orient='records')

    return jsonify({
        "director": nombre_director,
        "peliculas": peliculas_info
    })

@app.route('/recomendacion/<titulo>', methods=['GET'])

def recomendacion(titulo: str):
    if titulo not in indices:
        return {"error": "La pelicula no se encuentra en la base de datos."}
    
    idx = indices[titulo]
    distances, indices_nn = knn.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    movie_indices = indices_nn[0][1:6]
    
    return {"recomendaciones": peliculas_df['title'].iloc[movie_indices].tolist()}


if __name__ == '__main__':
    app.run(debug=True)

