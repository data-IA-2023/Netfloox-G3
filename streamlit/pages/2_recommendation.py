# ===================================
# importation
# ===================================

import os
import streamlit as st
import pandas as pd
import psycopg2
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
from scipy.sparse import hstack
from fuzzywuzzy import process

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

st.set_page_config(page_title="Recommendation", page_icon="🎬​")

######

try :
    dfc = st.session_state.df_session
except :
    dfc = False

# ===================================
# body pour estimation
# ===================================

try :
    cur = st.session_state.cur_session
except :
    cur = False

chatbot_tiny_logo = st.session_state.image_open
st.image(chatbot_tiny_logo) 
# st.header("NETFLOOX")
st.title('🎬​ Movie Recommendation System 🎬​')

if cur != False :
    
    # Load data with st.cache_data for better caching
    @st.cache_data
    def load_data():
        cur.execute('''SELECT tb.tconst, tb."primaryTitle", tb."titleType", tb.genres, 
tb."startYear", tb."runtimeMinutes", 
array_to_string(cbt."actor",','), array_to_string(cbt."director",','), array_to_string(cbt."writer",','),
array_to_string(cbt."producer",','), array_to_string(cbt."cinematographer",','), array_to_string(cbt."composer",','),
array_to_string(cbt."editor",','), array_to_string(cbt."production_designer",','), array_to_string(cbt."self",','),
array_to_string(cbt."archive_footage",','), array_to_string(cbt."archive_sound",','),
tr."averageRating", tr."numVotes", count(ta.title)
FROM "titleBasics" tb 
FULL OUTER JOIN "titleAkas" ta ON tb.tconst = ta."titleId" 
FULL OUTER JOIN cast_by_title cbt ON tb.tconst = cbt.tconst 
FULL OUTER JOIN "titleRatings" tr ON tb.tconst = tr.tconst 
WHERE tb."isAdult" = '0'  AND tb."titleType" = 'movie'
GROUP BY tb.tconst, cbt."actor", cbt."director", cbt."writer", cbt."producer", cbt."cinematographer", cbt."composer", cbt."editor", cbt."production_designer", cbt."self", cbt."archive_footage", cbt."archive_sound", tr."averageRating", tr."numVotes"
ORDER BY tr."numVotes" DESC NULLS LAST, tr."averageRating" DESC NULLS LAST
LIMIT 500000''')
        movies = cur.fetchall()
        df = pd.DataFrame(movies, columns=["tconst", "title", "titleType", "genres", "startYear", "runtimeMinutes", "actor", "director", "writer", "producer", "cinematographer", "composer", "editor", "production_designer", "self", "archive_footage", "archive_sound", "averageRating", "numVotes", "countTitleByRegion"])
        
        return df
    
    def df_fillna (df) :
        df['genres'] = df['genres'].fillna('')
        df['genres'] =  df['genres'].str.replace(',', ' ')
        df['actor'] = df['actor'].astype(str) 
        df['director'] =df['director'].astype(str)
        df['writer'] =df['writer'].astype(str)
        df['genres'] = df['genres'].fillna('')
        df['countTitleByRegion']= df['countTitleByRegion'].fillna('')
        df['averageRating'] =df['averageRating'].astype(str)
       
        # For simplicity,  fill missing values for categorical data with a placeholder and numerical with median
        df.fillna({'title': 'Unknown', 'director': 'Unknown', 'actor': 'Unknown', 
                'genres': 'Unknown', 'writer': 'Unknown'}, inplace=True)
        
        # Imputing missing numerical values with median
        num_imputer = SimpleImputer(strategy='median')
        df[['startYear', 'runtimeMinutes' , 'countTitleByRegion']] = num_imputer.fit_transform(df[['startYear', 'runtimeMinutes','countTitleByRegion']])
        
        return df
     
    df = load_data()
    if dfc != False :
        df = df_fillna (dfc)
    else :   
        dfc = load_data()
        df = df_fillna (dfc)
    
    df['combined_features'] = df['title']+ ' ' + df['director']+ ' ' + df['actor']   + ' ' + df['genres'] + ' ' + df['writer'] 
    scaler = MinMaxScaler()
    numerical_features_scaled = scaler.fit_transform(df[['startYear', 'runtimeMinutes', 'countTitleByRegion']])
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    combined_features_tfid1 = vectorizer.fit_transform(df['combined_features'])
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(combined_features_tfid1)
    combined_features_tfidf = hstack([combined_features_tfid1, numerical_features_scaled])



    def get_recommendations(film_name,df):
        film_index = df[df['title'] == film_name].index[0] # Assuming film_name exactly matches a title in df
        film_features = vectorizer.transform([df.iloc[film_index]['combined_features']])
        distances, indices = model_knn.kneighbors(film_features, n_neighbors=50) # Fetching20neighbors to exclude the film itself
        recommendations = []
        
        for i in range(0, len(distances.flatten())):

            if i == 0:
                print(f'Recommendations for "{film_name}":\n')
            else:
                # Append recommendation details to the list
                index = indices.flatten()[i]  # Get the original DataFrame index for the recommended item
                recommendations.append({
                    'index': df.index[index],
                    'title': df.iloc[indices.flatten()[i]]['title'],
                    'Distance': distances.flatten()[i],
                    'rating': df['averageRating'][i] ,
                    'actor' : df['actor'][i] ,
                    'dir': df['director'][i] 
                })
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df

  

    film_to_index = pd.Series(df.index, index=df['title']).to_dict()
    # Recommendation cosine_sim
    @st.cache_data
    def get_cosine_sim_recommendations(film_name):
        combined_features_tfidf_csr = combined_features_tfidf.tocsr()
            # Get the index of the film from its name
        film_index = df[df['title'] == film_name].index[0]
            # Compute cosine similarity between the film's features and the features of all films
        cosine_sim = cosine_similarity(combined_features_tfidf_csr, combined_features_tfidf_csr[film_index])
            
            # Get pairwise similarity scores for all films with that film
        sim_scores = list(enumerate(cosine_sim.flatten()))
            
            # Sort the films based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the scores of the 50 most similar films, skip the first one since it is the query film itself
        sim_scores = sim_scores[1:51]
        
        # Get the film indices
        film_indices = [i[0] for i in sim_scores]
        
        # Get the similarity scores (excluding the film itself)
        similarity_scores = [i[1] for i in sim_scores]
            
        recommendations = df.iloc[film_indices][['title', 'actor']]
        
        # Add the cosine similarity scores to the dataframe
        recommendations['cosine_similarity'] = similarity_scores
        
        # Reset the index of the dataframe
        recommendations = recommendations.reset_index()
        
        return recommendations

    def create_graph(recommendations_df, input_film_name):
        G = nx.Graph()
        G.add_node(input_film_name, size=20, color= "black" , title=input_film_name)
        
        for index, row in recommendations_df.iterrows():
            G.add_node(row['title'], size=15,color= "red", title=row['title'])
            G.add_edge(input_film_name, row['title'])
        return G


    def show_graph(G):
        nt = Network('500px', '100%', notebook=True)
        nt.from_nx(G)
        nt.show('nx.html')
        return 'nx.html'
    

    st.write(df)    
    film_titles = dfc['title'].tolist()
    

    # Function to suggest film names based on input
    def suggest_film_names(input_text, film_titles):
      suggestions = [title for title in film_titles if input_text.lower() in title.lower()]
      return suggestions

    input_text = st.text_input("Enter the name of the film:")
    def main():
        # Main page
        suggested_titles = suggest_film_names(input_text, film_titles)

        # Update the selectbox with suggested titles
        selected_title = st.selectbox("Did you mean:", suggested_titles)
        if st.button('Recommendations'):
            if selected_title:
                # Apply filtering only if a specific genre is selected; otherwise, use the full dataset
                suggested_film_name=selected_title
                if suggested_film_name:
                    st.write(f"Suggested Name: {suggested_film_name}")
                    rec_knn = get_recommendations(suggested_film_name,df)
                    rec_knn = pd.DataFrame(rec_knn)
                    rec_movies = get_cosine_sim_recommendations(suggested_film_name )

                    if not rec_knn.empty and rec_movies is not None and not rec_movies.empty:
                            st.write("KNN Recommendations:")
                            st.dataframe(rec_knn)
                            G_knn = create_graph(rec_knn, suggested_film_name)
                            knn_graph_path = show_graph(G_knn)
                            HtmlFile = open(knn_graph_path, 'r', encoding='utf-8')
                            source_code = HtmlFile.read() 
                            components.html(source_code, width=500, height=500)

                    
                            st.write("Cosine Similarity :")
                            st.dataframe(rec_movies)

                            G_cosine = create_graph(rec_movies, suggested_film_name)
                            cosine_graph_path = show_graph(G_cosine)
                            HtmlFile = open(cosine_graph_path, 'r', encoding='utf-8')
                            source_code = HtmlFile.read()
                            components.html(source_code, width=500, height=500)

                            matched_df = pd.merge(rec_knn, rec_movies, how='inner', on=('title','index')).head(5)
                        
                            if not matched_df.empty:
                                st.write("Matched Recommendations:")
                                st.dataframe(matched_df)
                                G_cosine = create_graph(matched_df , suggested_film_name)
                                cosine_graph_path = show_graph(G_cosine)
                                HtmlFile = open(cosine_graph_path, 'r', encoding='utf-8')
                                source_code = HtmlFile.read()
                                components.html(source_code, width=500, height=500)

                            else:
                                st.write("No matched recommendations found.")
                    else:
                        st.write("No recommendations found.")
               
            else:
                st.write("Please enter a film name for recommendations.")
    
    if __name__ == "__main__":
        main()



else :
    st.sidebar.error("Tu dois te connecter avant de pouvoir faire une prédiction de popularité")


