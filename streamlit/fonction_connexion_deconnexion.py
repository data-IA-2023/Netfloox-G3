##########
# Importe les modules
##########

import psycopg2
import pandas as pd


##########
# Connexion à la base de donnée et création du data frame
##########

def connexion_postgres (hostname, database, username, password) :
    
    try :
        cur.close()
    except :
        print(0)

    con = psycopg2.connect(f"host={hostname} dbname={database} user={username} password={password}")
    cur = con.cursor()

    return cur, con

def deconnexion_postgres (cur) :
    cur.close()

    return "deconnexion"