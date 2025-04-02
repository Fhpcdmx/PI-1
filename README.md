# Proyecto individual 1 Sistema de recomendacion de peliculaa

#Introducción:
El proyecto se realiza con la intención de brindar un api para consulta y recomendacion de peliculas, en el proyecto se realiza el ETL de los datos y el analisis exploratorio de los mismo para determinar
los parametros mas adecuados para el sistema de recomendación, este proyecto represento un reto al tener que manejar datos en formatos nuevos y realización de uniones entre bases de datos directo en python,

Utilizacion del api:

El api nos permite realizar las busqueda de información en una base de datos muy amplia de una manera rapida devolviendonos informacion en formato json para la utilización en paginas web.
el sistema cuenta con las siguietes funciones:

"/cantidad_filmaciones_mes/<Mes> ejemplo:https://pi-1-wyls.onrender.com/cantidad_filmaciones_mes/Enero "
"/cantidad_filmaciones_dia/<Mes> ejemplo:https://pi-1-wyls.onrender.com/cantidad_filmaciones_mes/Lunes "
"/score_titulo/<Titulo de la pelicula> ejemplo:https://pi-1-wyls.onrender.com/score_titulo/Toy story "
"/votos_titulo/<Titulo de la pelicula> ejemplo:https://pi-1-wyls.onrender.com/votos_titulo/Toy story "
"/get_actor/<Nombre del actor> ejemplo:https://pi-1-wyls.onrender.com/get_actor/Tom hanks "
"/get_director/<Nombre del director> ejemplo:https://pi-1-wyls.onrender.com/get_director/John Lasseter "
"/recomendacion/<Titulo de la pelicula> ejemplo:https://pi-1-wyls.onrender.com/recomendacion/Babe "

Cada una de  ellas nos brindara información relevante.
