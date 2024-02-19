import pandas as pd
import requests
import gzip
import io
import os
from sqlalchemy import create_engine
import psycopg2
import streamlit as st

############################


# Fonction qui retourne le l'url vers la base de donnée 
def connectEnginePostgresql(hostname,username,db,port,password):
    connection_string = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{db}"
    return connection_string

# Fonction qui transforme des fichier .tsv en dataframe puis en table sql 
def Yahya(file_url,rowLimit,engine,nameTable):
        
        # Si l'utlisateur indique None ou laisse le champ vide
        if rowLimit == 'None' or rowLimit =='':
             rowLimit = None
        # Téléchargement du fichier
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with io.BytesIO(r.content) as buf, gzip.open(buf, 'rt', encoding='latin-1') as gz_file:
                 # Transformation en dataframe.
                df = pd.read_csv(gz_file, delimiter='\t', low_memory=False, nrows=rowLimit)

        # Remplacement des  '\\N' par NaN
        df = df.replace('\\N', pd.NA)

       # Transformation du dataframe en table sql 
        df.to_sql(nameTable, engine, if_exists='append', index=False, chunksize=1000)
        print("Data successfully saved to PostgreSQL database.")

# Fonction d'importation de donnée
def importDB() :
  
  # Connexion à la base de donnée (seulement pour du Postgresql)
  connection_string = connectEnginePostgresql(hostname,username,database,port,password)
  engine            = create_engine(connection_string)

  # boucle for pour chaque URL
  try:
        for i,j in zip (urlList,nameDB):
            Yahya(i,rowLimit,engine,j)
  except Exception as e:
        print("Error:", e)
  finally:
        engine.dispose()    
        print("Connection closed")

#############################


st.set_page_config(
    page_title="ImportDb",
    page_icon="🍿",
)


repertoir_fichier = os.path.dirname(__file__)
# st.write(repertoir_fichier)
st.image(f'{repertoir_fichier}\\floox.png')
st.header("NETFLOOX")
st.write("Importation à la base de donnée :")

# Url des fichiers
urlList  = [
           'https://datasets.imdbws.com/name.basics.tsv.gz',
           'https://datasets.imdbws.com/title.akas.tsv.gz',
           'https://datasets.imdbws.com/title.basics.tsv.gz',
           'https://datasets.imdbws.com/title.crew.tsv.gz',
           'https://datasets.imdbws.com/title.episode.tsv.gz',
           'https://datasets.imdbws.com/title.principals.tsv.gz',
           'https://datasets.imdbws.com/title.ratings.tsv.gz'
           ]

# Nom des tables 
nameDB   = ['namesBasics','titleAkas','titleBasics','titleCrew','titleEpisode','titlePrincipals','titleRatings']



# L'utilisateur doit inscrire ses données de connexion à sa base de donnée
with st.form("Formulaire pour l'importation à la base de donnée :"):
    # affichage les diférent parametre avec input
    hostname = st.text_input("nom du hostname", "localhost")
    database = st.text_input("nom de la database", "netfloox")
    username = st.text_input("nom de l'utilisateur", "postgres")
    port     = st.text_input("port", "5432")
    rowLimit = st.text_input("Limite d'importation de ligne", "None")
    password = st.text_input("mot de passe",type="password")
    st.form_submit_button("import", on_click=importDB)

