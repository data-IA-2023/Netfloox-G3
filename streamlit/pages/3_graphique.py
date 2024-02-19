# ===================================
# importation
# ===================================

import os
import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine

# ===================================
# style
# ===================================

# [theme]
# primaryColor="#e50914"
# backgroundColor="#2f2c2c"
# secondaryBackgroundColor="#171616"
# textColor="#e50914"
# font="serif"

# ===================================
# traitement des boutons et variable de session
# ===================================

st.set_page_config(page_title="Film Data Analysis", page_icon="📈")

######

try  :
    conn = st.session_state.con_session
except :
    conn = False

# ===================================
# body 
# ===================================

chatbot_tiny_logo = st.session_state.image_open
st.image(chatbot_tiny_logo) 
# st.header("NETFLOOX")
st.title("📈 Film Data Analysis 📈")

st.write(conn)
if conn != False :

    # Function to load data from PostgreSQL database
    @st.cache_data
    def load_data(query):
        data = pd.read_sql_query(query, conn)
        return data

    option = st.sidebar.selectbox(
        'Choose a category:',
        ('Top 20 Movies by Average Rating',
        'Movies Released Over Years by Genre',
        'Number of movies  released in each Genres',
        'Number of Films by Genre and Year of Release',
        'heatmap'
        )
    )

    # Main content: Display visualization based on selected category

    if option == 'Top 20 Movies by Average Rating':
        st.title('Top 20 Movies by Average Rating')
        query = """
        SELECT tb."originalTitle", tr."numVotes", tr."averageRating"
        FROM "titleRatings" tr
        JOIN "titleBasics" tb ON tb."tconst" = tr."tconst"
        WHERE tb."titleType" = 'movie' AND "numVotes" > 1000
        ORDER BY "numVotes" DESC, "averageRating" DESC
        LIMIT 20
        """
        data = load_data(query)
        st.subheader('Top 20 Movies by Average Rating and Number of Votes (>1000)')
        fig, ax = plt.subplots()
        sns.barplot(x='averageRating', y='originalTitle', data=data, ax=ax, palette='coolwarm')
        plt.xlabel('Average Rating')
        plt.ylabel('')
        st.pyplot(fig)

    elif option == 'Movies Released Over Years by Genre':
        st.title('Movies Released Over Years by Genre')
        query = '''SELECT
        tb."originalTitle" AS Titre,
        tb."startYear" AS Année,
        tb."genres" AS Genre
    FROM
        "titleBasics" tb
    WHERE 
    tb."titleType" = 'movie' AND tb."startYear" IS NOT NULL AND tb."startYear"BETWEEN 2000 AND 2024
    GROUP BY Genre, Titre, Année
    ORDER BY Année  ASC; ''' 

        data = load_data(query)
        data['FirstGenre'] = data['genre'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else None)
        
        top_genres = data['FirstGenre'].value_counts().nlargest(5).index
        
        at_top_genres = data[data['FirstGenre'].isin(top_genres)]
        
        plt.figure(figsize=(20, 8))
        sns.countplot(x='année', hue='FirstGenre', data=at_top_genres, palette='viridis')
        plt.xticks(rotation=90)
        plt.title('Number of Movies Released Over Years by Genre')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.legend(title='Genre')
        
        st.pyplot(plt)

    elif option == 'Number of movies  released in each Genres':
            st.title('Number of movies  released in each Genres')
            query = '''SELECT
        tb."startYear" AS "Year",
        tb."originalTitle" AS "Titre",                        
        tb."genres" AS "Genres",
        AVG(tr."averageRating") AS "AvgRating",
        COUNT(tb."tconst") AS "MovieCount"
        FROM
            "titleBasics" AS tb
        JOIN
            "titleRatings" AS tr ON tb."tconst" = tr."tconst"
        WHERE
            tb."genres" IS NOT NULL
        GROUP BY
        tb."startYear", tb."genres", tb."originalTitle"
                    
    ; '''

            data1 = load_data(query)
            data1['Genres'] = data1['Genres'].str.split(',')
            df_exploded = data1.explode('Genres')

            # Aggregating the exploded data to count movies by genre
            genre_counts = df_exploded['Genres'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'MovieCount']
            df_mod = df_exploded.groupby(['Year', 'Genres']).agg({'AvgRating': 'mean'}).reset_index()
            pivot_table = df_mod.pivot(index='Year', columns='Genres', values='AvgRating')
            genre_counts = df_exploded['Genres'].value_counts()
            data=pd.DataFrame.from_dict([ genre_counts ])
            data.columns=list(genre_counts.keys())
            data.plot.bar(figsize=(12,6))
            plt.title("Number of movies  released in each Genres")
            plt.xlabel("Genres")
            plt.ylabel("No of movies released")
            st.pyplot(plt)

    elif option == 'Number of Films by Genre and Year of Release':
            st.title('Number of Films by Genre and Year of Release')
            query = '''SELECT
        tb."startYear" AS "Year",
        tb."originalTitle" AS "Titre",                        
        tb."genres" AS "Genres",
        AVG(tr."averageRating") AS "AvgRating",
        COUNT(tb."tconst") AS "MovieCount"
        FROM
            "titleBasics" AS tb
        JOIN
            "titleRatings" AS tr ON tb."tconst" = tr."tconst"
        WHERE
            tb."genres" IS NOT NULL
        GROUP BY
        tb."startYear", tb."genres", tb."originalTitle"
                    
    ; '''

            data2 = load_data(query)
            data2['Genres'] = data2['Genres'].str.split(',')
            df_exploded = data2.explode('Genres')
            film_counts = df_exploded.groupby(['Year', 'Genres']).size().reset_index(name='count')
            film_matrix = film_counts.pivot(index="Year", columns="Genres", values="count")
            plt.figure(figsize=(40, 22))
            sns.heatmap(film_matrix.head(100), cmap="YlGnBu", annot=True, fmt="g") 
            plt.title('Number of Films by Genre and Year of Release')
            plt.xlabel('Genre')
            plt.ylabel('Year')
            st.pyplot(plt)

    elif option == 'heatmap':
            st.title('heatmap')
            query = '''SELECT
        tb."startYear" AS "Year",
        tb."originalTitle" AS "Titre",                        
        tb."genres" AS "Genres",
        AVG(tr."averageRating") AS "AvgRating",
        COUNT(tb."tconst") AS "MovieCount"
        FROM
            "titleBasics" AS tb
        JOIN
            "titleRatings" AS tr ON tb."tconst" = tr."tconst"
        WHERE
            tb."genres" IS NOT NULL
        GROUP BY
        tb."startYear", tb."genres", tb."originalTitle"
                    
    ; '''

            data3 = load_data(query)
            data3['Genres'] = data3['Genres'].str.split(',')
            df_exploded = data3.explode('Genres')
            film_counts = df_exploded.groupby(['Year', 'Genres']).size().reset_index(name='count')
            film_matrix = film_counts.pivot(index="Year", columns="Genres", values="count")
            correlation =  film_matrix.corr()
            plt.figure(figsize=(20,20))
            sns.heatmap(correlation,annot=True,linewidths=0.01,vmax=1,square=True,cbar=True);
            st.pyplot(plt)


else :
    st.sidebar.error("Tu dois de connecter avant de pouvoir te déconnecter")