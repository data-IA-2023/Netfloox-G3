# ===================================
# importation
# ===================================

import streamlit as st
import os
from PIL import Image
import fonction_connexion_deconnexion as connex_deconnex

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

st.set_page_config(
    page_title="Connection",
    page_icon="🌍",
)

######

if 'cur' not in st.session_state:
    st.session_state.cur_session = False

######

# traitement du bouton de connexion
def click_connexion():
    st.session_state.connexion = True
    st.session_state.deconnexion = False

if 'connexion' not in st.session_state:
    st.session_state.connexion = False

######

# traitement du bouton de connection
def click_deconnexion():
    st.session_state.deconnexion = True
    st.session_state.connexion = False

if 'deconnexion' not in st.session_state:
    st.session_state.deconnexion = False

# ===================================
# slidbar : déconnexion
# ===================================

st.sidebar.button("deconnexion", on_click=click_deconnexion)

if st.session_state.deconnexion :
    st.write("connexion ...")

    try  :
        cur = st.session_state.cur_session
        text_deconnex = connex_deconnex.deconnexion_postgres( cur )
        if text_deconnex == "deconnexion" :
            st.sidebar.success("deconnexion success")

        else :
            st.sidebar.error("erreur de déconnexion")
    
    except :
        st.sidebar.error("Tu dois de connecter avant de pouvoir te déconnecter")

# ===================================
# body : connection
# ===================================

repertoir_fichier = os.path.dirname(__file__)
# st.write(repertoir_fichier)
chatbot_tiny_logo = Image.open(f'{repertoir_fichier}\\floox.png')
st.session_state.image_open = chatbot_tiny_logo
st.image(chatbot_tiny_logo) 
# st.header("NETFLOOX")
st.title("🌍 Connexion 🌍")

with st.form("Formulaire pour la connexion à la base de donnée :"):
    # affichage les diférent parametre avec input
    hostname = st.text_input("nom du hostname", "localhost")
    database = st.text_input("nom de la database", "netfloox")
    username = st.text_input("nom de l'utilisateur", "postgres")
    password = st.text_input("mot de passe", type="password")
    st.form_submit_button("connexion", on_click=click_connexion)


if st.session_state.connexion :
    st.write("connexion ...")

    cur, con = connex_deconnex.connexion_postgres( hostname, database, username, password)
    st.session_state.cur_session = cur
    st.session_state.con_session = con

    if cur :
        st.success("connexion success")

    else :
        st.error("erreur de déconnexion")