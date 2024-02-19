# ===================================
# importation
# ===================================

import os
import streamlit as st
import pandas as pd
from PIL import Image
import algo_prediction as algo_pred

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

st.set_page_config(page_title="Prédiction", page_icon="🌟")

######

# traitement du bouton de formulaire
def click_formulaire():
    st.session_state.formulaire = True
    st.session_state.recommendation = False

if 'formulaire' not in st.session_state:
    st.session_state.formulaire = False

if 'formulaire_complet' not in st.session_state:
    st.session_state.formulaire_complet = False

######

try :
    df = st.session_state.df_session
except :
    df = False

# ===================================
# slidbar
# ===================================

try :
    cur = st.session_state.cur_session
except :
    cur = False

if cur != False :

    if df != False :
        dfc = algo_pred.preparation_df (df)
    else :
        df = algo_pred.postgres_execute_sql(cur)
        dfc = algo_pred.preparation_df (df)

    st.sidebar.header("Formulaire pour l'estimation de la popularité d'un nouveau film :")

    with st.sidebar:
        with st.form("Formulaire pour l'estimation de la popularité d'un nouveau film :"):
            new_genres = st.text_input("choisis un ou plusieurs genres séparé par des espaces :", value="Animation")
            st.write("actor, self, director, producer, writer, cinematographer, composer, editor, production_designer, archive_footage, archive_sound")
            new_cast = st.text_input("Entre le nom du casting dans l'ordre ci-dessus", value="Christian Clavier, Guillaume Briat, Daniel Mesguich, Lévanah Solomon, Alex Lutz, Alexandre Astier, Elie Semoun, Gérard Hernandez, François Morel, Lionnel Astier, Florence Foresti, Serge Papagalli, Alexandre Astier, Louis Clichy,  Alexandre Astier, Louis Clichy, Joël Savdie, Mariette Kelley, Philippe Rombi, David Dulac")
            new_title = st.text_input("Entre le nom du film", value="Astérix : Le Secret de la potion magique")
            new_countRegion = st.slider("Entre le nombre de région où le film va sortir", min_value=1.0, max_value=300.0, step =1.0, value=20.0)
            new_startYear = st.slider("En quelle année le film va sortir", min_value=1850.0, max_value=2100.0, step =1.0, value=2018.0)
            new_runtimeMinutes = st.slider("Quelle est la durée du film en minute", min_value=0.1, max_value=600.0, step =0.1, value=85.0)
            st.form_submit_button("formulaire", on_click=click_formulaire)

    # valeur par default :
    # "Animation", 
    # "Christian Clavier, Guillaume Briat, Daniel Mesguich, Lévanah Solomon, Alex Lutz, Alexandre Astier, 
    # Elie Semoun, Gérard Hernandez, François Morel, Lionnel Astier, Florence Foresti, Serge Papagalli, Alexandre Astier, 
    # Louis Clichy,  Alexandre Astier, Louis Clichy, Joël Savdie, Mariette Kelley, Philippe Rombi,
    # David Dulac", 
    # "Astérix : Le Secret de la potion magique", 
    # 1,
    # 2018,
    # 87

else :
    st.sidebar.error("Tu dois te connecter avant de pouvoir faire une prédiction de popularité")

# ===================================
# vérification de la validité du formulaire
# ===================================
    
if st.session_state.formulaire :

    if not new_genres :
        st.sidebar.error("Tu dois renseigner le ou les genres du film")
    if not new_cast :
        st.sidebar.error("Tu dois renseigner la liste du casting du film")
    if not new_title :
        st.sidebar.error("Tu dois renseigner un titre pour le film")
    if not new_countRegion :
        st.sidebar.error("Tu dois renseigner un nombre de région où le film va sortir")
    if not new_startYear :
        st.sidebar.error("Tu dois renseigner une année pour le film")
    if not new_runtimeMinutes :
        st.sidebar.error("Tu dois renseigner une durée pour le film")

    if new_genres and new_cast and new_title and new_countRegion and new_startYear and new_runtimeMinutes :
        statu_form = "formulaire_complet"
        st.session_state.formulaire_complet = True

# ===================================
# body pour estimation
# ===================================

chatbot_tiny_logo = st.session_state.image_open
st.image(chatbot_tiny_logo) 
# st.header("NETFLOOX")
st.title("🌟 Prédiction 🌟")

if st.session_state.formulaire_complet :
    st.header("Résultat de l'estimation :")

    # st.write(new_genres, new_cast, new_title, new_countRegion, new_startYear, new_runtimeMinutes)
    X_new = pd.DataFrame([{"title":new_title, "titleType":"movie", "genres":new_genres, 
                          "startYear":new_startYear, "runtimeMinutes":new_runtimeMinutes, 
                          "numVotes":0.0, "countTitleByRegion":new_countRegion, "cast":new_cast}])
    
    X_train, X_test, y_train, y_test = algo_pred.x_y_split (dfc)
    model = algo_pred.load_model()
    rmse, result_score = algo_pred.result_model (model, X_train, X_test, y_train, y_test) 

    pred = algo_pred.test_model (model, X_new)

    result_prop = round(pred[0]*10, 2)

    if pred != None :
        st.write(f"le model prédit une popularité de {result_prop}%")
        st.write("fiabilité du résultat : ")
        st.write(f"rmse = {rmse}, r2 = {result_score}")
        
    else :
        st.error("Erreur dans la création de la base de donnée")
