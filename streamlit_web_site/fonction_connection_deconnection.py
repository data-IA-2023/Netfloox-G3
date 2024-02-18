##########
# Importe les modules
##########
import psycopg2
import pandas as pd


##########
# Conection à la base de donnée et création du data frame
##########

def connection_postgres (hostname, database, username, password) :
    
    try :
        cur.close()
    except :
        print(0)

    con = psycopg2.connect(f"host={hostname} dbname={database} user={username} password={password}")
    cur = con.cursor()

    return cur

def deconnection_postgres (cur) :
    cur.close()

    return "deconnection"