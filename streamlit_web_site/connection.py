####
# importation
####

import streamlit as st
import os
import fonction_connection_deconnection as connect_deconnet

####
# style
####

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color : #171616;
#         color : #e50914;
#         font-family: 'Inter', sans-serif;
#     }
#     body {
#         background-color : #2f2c2c;
#         color : #e50914;
#         font-family: 'Inter', sans-serif;
#     }
# </style>
# """, unsafe_allow_html=True)

####
# traitement des boutons et variable de session
####

st.set_page_config(
    page_title="Connection",
    page_icon="🌍",
)

######

if 'cur' not in st.session_state:
    st.session_state.cur_session = False

######

# traitement du bouton de connection
def click_connection():
    st.session_state.connection = True
    st.session_state.deconnection = False

if 'connection' not in st.session_state:
    st.session_state.connection = False

######

# traitement du bouton de connection
def click_deconnection():
    st.session_state.deconnection = True
    st.session_state.connection = False

if 'deconnection' not in st.session_state:
    st.session_state.deconnection = False

####
# slidbar : déconnection
####

st.sidebar.button("deconnection", on_click=click_deconnection)

if st.session_state.deconnection :
    st.write("connection ...")

    try  :
        cur = st.session_state.cur_session
        text_deconnet = connect_deconnet.deconnection_postgres( cur )
        if text_deconnet == "deconnection" :
            st.sidebar.success("deconnection success")

        else :
            st.sidebar.error("erreur de déconnection")
    
    except :
        st.sidebar.error("Tu dois de connecter avant de pouvoir te déconnecter")

####
# body : connection
####

repertoir_fichier = os.path.dirname(__file__)
# st.write(repertoir_fichier)
st.image(f'{repertoir_fichier}/pages//floox.png')
st.header("NETFLOOX")
st.write("Connection à la base de donnée :")

with st.form("Formulaire pour la connection à la base de donnée :"):
    # affichage les diférent parametre avec input
    hostname = st.text_input("nom du hostname", "localhost")
    database = st.text_input("nom de la database", "netfloox")
    username = st.text_input("nom de l'utilisateur", "postgres")
    password = st.text_input("mot de passe")
    st.form_submit_button("connection", on_click=click_connection)


if st.session_state.connection :
    st.write("connection ...")

    cur = connect_deconnet.connection_postgres( hostname, database, username, password)
    st.session_state.cur_session = cur

    if cur :
        st.success("connection success")

    else :
        st.error("erreur de déconnection")