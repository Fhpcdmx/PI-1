import pandas as pd
import ast 

# Cargar el CSV
credits = pd.read_csv('data/credits.csv')
data_movie = pd.read_csv('data/movies_dataset.csv',encoding='latin1')

for col in ["genres"]:  
    data_movie[col] = data_movie[col].apply(ast.literal_eval)  # Convierte string en listas

# Convertir las columnas de string a listas de diccionarios
for col in ["cast", "crew"]:  
    credits[col] = credits[col].apply(ast.literal_eval)  # Convierte string en listas

# generar una fila para cada miembro del equipo
df_cast = credits.explode("cast")
df_crew = credits.explode("crew")
df_genres = data_movie.explode("genres")


# Extraer
df_cast["cast_id"] = df_cast["cast"].apply(lambda x: x["cast_id"] if isinstance(x, dict) else None)
df_cast["character"] = df_cast["cast"].apply(lambda x: x[str("character")] if isinstance(x, dict) else None)
df_cast["id_actor"] = df_cast["cast"].apply(lambda x: x["id"] if isinstance(x, dict) else None)
df_cast["actor_name"] = df_cast["cast"].apply(lambda x: x["name"] if isinstance(x, dict) else None)
df_cast["order"] = df_cast["cast"].apply(lambda x: x["order"] if isinstance(x, dict) else None)
df_genres["id_genres"] = df_genres["genres"].apply(lambda x: x["id"] if isinstance(x, dict) else None)
df_genres["name"] = df_genres["genres"].apply(lambda x: x[str("name")] if isinstance(x, dict) else None)
df_crew["crew_name"] = df_crew["crew"].apply(lambda x: x["name"] if isinstance(x, dict) else None)
df_crew["job"] = df_crew["crew"].apply(lambda x: x["job"] if isinstance(x, dict) else None)

df_cast = df_cast[['id','cast_id','character','id_actor', 'actor_name','order' ]]
df_crew = df_crew[['id', 'crew_name', 'job']]
data_movie['revenue'].fillna('').replace('',0, inplace=True)
data_movie['budget'].fillna('').replace('',0, inplace=True)
data_movie.dropna(subset='release_date',inplace=True)
data_movie['return'] = (data_movie['revenue'] / data_movie['budget']).fillna(0)
data_movie.drop(['video','imdb_id','adult','original_title','poster_path'],axis=1)
data_movie=pd.to_datetime(data_movie['release_date'], errors='coerce')
df_genres = df_genres[['id','id_genres','name' ]]





df_cast.dropna(how='any',inplace=True, subset='character')
df_cast.to_csv('cast.csv')
df_crew.to_csv('crew.csv')
df_genres.to_csv('genres.csv')
data_movie.to_csv('movies')