# ===================================
# Importe les modules
# ===================================

# sql
import psycopg2
from sqlalchemy import create_engine
# base 
import pandas as pd
import numpy as np
# test et train
from sklearn.model_selection import train_test_split
# pipeline
from sklearn.pipeline import make_pipeline, Pipeline
# preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler, OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
# model
import pickle
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
# metrique d'erreur
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# ===================================
# Extactions des données
# ===================================

def postgres_execute_sql (cur) :
    cur.execute('''select tb.tconst, tb."primaryTitle", tb."titleType", tb.genres, 
    tb."startYear", tb."runtimeMinutes", 
    array_to_string(cbt."actor",','),array_to_string(cbt."director",','),array_to_string(cbt."writer",','),
                array_to_string(cbt."producer",','),array_to_string(cbt."cinematographer",','),
                array_to_string(cbt."composer",','),array_to_string(cbt."editor",','),
                array_to_string(cbt."production_designer",','),array_to_string(cbt."self",','),
                array_to_string(cbt."archive_footage",','),array_to_string(cbt."archive_sound",','),
    tr."averageRating", tr."numVotes",
    count(ta.title)
    from "titleBasics" tb 
    full outer join "titleAkas" ta 
    on tb.tconst = ta."titleId" 
    full outer join cast_by_title cbt 
    on tb.tconst = cbt.tconst 
    full outer join "titleRatings" tr 
    on tb.tconst  = tr.tconst 
    where
    tb."isAdult" = 0
    AND tb."titleType" LIKE 'movie'
    group by tb.tconst,
    cbt."actor",cbt."director",cbt."writer",cbt."producer",cbt."cinematographer",cbt."composer",
                cbt."editor",cbt."production_designer",cbt."self",cbt."archive_footage",cbt."archive_sound",
    tr."averageRating", tr."numVotes"
    ORDER BY tr."numVotes" DESC NULLS LAST, tr."averageRating" DESC NULLS LAST
    ''' )

    records = cur.fetchall()

    df = pd.DataFrame(records , columns=["tconst","title","titleType","genres","startYear","runtimeMinutes",
                                         "actor","director","writer","producer","cinematographer","composer",
                                         "editor","production_designer","self","archive_footage","archive_sound",
                                         "averageRating","numVotes","countTitleByRegion"])
    
    return df

# ===================================
# préparation du data frame
# ===================================

def cast(row):
    return row['actor']+" "+row["self"]+" "+row["director"]+" "+row["producer"]+" "+row["writer"]+" "+row["cinematographer"]+" "+row["composer"]+" "+row["editor"]+" "+row["production_designer"]+" "+row["archive_footage"]+" "+row["archive_sound"]

def preparation_df (df) :
    dfc = df.dropna(subset=['tconst', 'title', 'titleType', 'genres'])

    dfc['startYear'] = pd.to_numeric(dfc['startYear'], errors='coerce').fillna(0)
    dfc['runtimeMinutes'] = pd.to_numeric(dfc['runtimeMinutes'], errors='coerce').fillna(0)
    dfc['countTitleByRegion'] = pd.to_numeric(dfc['runtimeMinutes'], errors='coerce').fillna(0)
    dfc['numVotes'] = pd.to_numeric(dfc['runtimeMinutes'], errors='coerce').fillna(0)
    dfc['averageRating'] = pd.to_numeric(dfc['averageRating'], errors='coerce').fillna(dfc['averageRating'].mean())
    dfc['genres'] = dfc['genres'].fillna('')

    casting = ["actor","director","writer","producer","cinematographer",
               "composer","editor","production_designer","self","archive_footage","archive_sound"]
    
    for cast in casting:
        dfc[cast] = dfc[cast].fillna('')
    
    dfc['cast'] = dfc.apply(cast, axis=1)

    dfc["genres"] = dfc["genres"].str.replace(',',' ')

    return dfc

# ===================================
# Créations des fiture et label
# ===================================

def x_y_split (dfc) :
    # Combine features into a single string per row
    X = dfc.drop(['tconst', 'averageRating',"actor","director","writer","producer","cinematographer","composer","editor","production_designer","self","archive_footage","archive_sound"],axis=1)
    y = dfc['averageRating']  


    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# ===================================
# ouverture du model.sav
# ===================================

def load_model () :
    filename = 'predict_model_pred.sav'

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    return loaded_model

# ===================================
# test du model
# ===================================

def result_model (loaded_model, X_train, X_test, y_train, y_test) :

    result_score = loaded_model.score(X_test, y_test)
    result  = loaded_model.predict(X_test)

    mse = mean_squared_error(y_test, result)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, result)

    return rmse, result_score

def test_model (loaded_model, X_test) :

    result  = loaded_model.predict(X_test)

    return result

# ===================================
# 
# ===================================

"""
hostname = "localhost"
database = "netfloox"
username = "postgres"
password = "wqasxz"
con = psycopg2.connect(f"host={hostname} dbname={database} user={username} password={password}")
cur = con.cursor()
df = postgres_execute_sql(cur)
print(df)

dfc = preparation_df (df)
X_train, X_test, y_train, y_test = x_y_split (dfc)
print(dfc)
model = load_model()
print(model)
result = test_model (model, X_train, X_test, y_train, y_test) 
print (result)

("Animation", 
"Christian Clavier, Guillaume Briat, Daniel Mesguich, Lévanah Solomon, Alex Lutz, Alexandre Astier, 
Elie Semoun, Gérard Hernandez, François Morel, Lionnel Astier, Florence Foresti, Serge Papagalli, Alexandre Astier, 
Louis Clichy,  Alexandre Astier, Louis Clichy, Joël Savdie, Mariette Kelley, Philippe Rombi,
David Dulac", 
"Astérix : Le Secret de la potion magique", 
1,
2018,
87
tconst = tt8001346)

X_new = pd.DataFrame([{"title":"Astérix : Le Secret de la potion magique", "titleType":"movie", "genres":"Animation", "startYear":2018.0, "runtimeMinutes":85.0, "numVotes":0.0, "countTitleByRegion":20.0, 
                      "cast":"Christian Clavier, Guillaume Briat, Daniel Mesguich, Lévanah Solomon, Alex Lutz, Alexandre Astier, Elie Semoun, Gérard Hernandez, François Morel, Lionnel Astier, Florence Foresti, Serge Papagalli, Alexandre Astier, Louis Clichy,  Alexandre Astier, Louis Clichy, Joël Savdie, Mariette Kelley, Philippe Rombi, David Dulac"}])

print (X_new)

"""
    