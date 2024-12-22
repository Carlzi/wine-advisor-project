import pandas as pd
import os

from wine_advisor.classification import params
from wine_advisor.classification.cleaning_functions import cleaning, vintage_identification, remove_accents, classification_type_of_wine, classification_region

def get_data():
    """
    Recuperation du dataset Wine Review de Kaggle
    """

    return pd.read_csv(params.path_dataset)


def cleaning_general(dataset_initial: pd.DataFrame):
    """
    Cleaning general du dataset Wine Review de Kaggle : les données conservées
    seront celles présentées comme output à l'utilisateur après correspondance
    de l'etiquette avec la base (price, rating, review...)
    """

    # On enleve les features non utiles
    df_useful = dataset_initial.drop(columns=["reviewer"])

    # On enleve aussi "designation" a l'ordre 1 pour simplifier
    df_useful.drop(columns=["designation"], inplace=True)

    # On complète les lignes sans review pour qu'elles ne soient pas supprimées
    df_useful.review=df_useful.review.fillna('Not available')

    # On supprime les lignes avec des NA
    df_useful.dropna(inplace=True)

    # On renomme les colonnes pour plus de clarté
    df_useful=df_useful.rename(columns=
                            {'price':'price_usd',
                                'appellation':'origine',
                                'category':'type_of_wine'})

    # On cleane la colonne alcohol (ex : "14%" => 0.14)
    df_useful.alcohol=df_useful.alcohol.str.replace("%","")
    df_useful.alcohol=pd.to_numeric(df_useful.alcohol,errors="coerce")/100

    # On cleane la colonne price (ex : "$15" => 15.0)
    df_useful.price_usd=df_useful.price_usd.str.replace("$","")
    df_useful.price_usd=pd.to_numeric(df_useful.price_usd,errors="coerce")


    # On extrait les infos clés (appellation, région, pays) présentes dans
    # la colonne origine géographique (=origine dans le dataset)
    df_useful["appellation"]=df_useful.origine.apply(lambda x: x.split(",")[0].strip() if len(x.split(","))>1 else None)
    df_useful["region"]=df_useful.origine.apply(lambda x: x.split(",")[-2].strip() if len(x.split(","))>2 else None)
    df_useful["country"]=df_useful.origine.apply(lambda x: x.split(",")[-1].strip())

    # On retire les accents des principales colonnes
    df_useful.appellation=df_useful.appellation.apply(cleaning).apply(remove_accents)
    df_useful.region=df_useful.region.apply(cleaning).apply(remove_accents)
    df_useful.country=df_useful.country.apply(cleaning).apply(remove_accents)
    df_useful.type_of_wine=df_useful.type_of_wine.apply(cleaning).apply(remove_accents)
    df_useful.varietal=df_useful.varietal.apply(cleaning).apply(remove_accents)
    df_useful.wine=df_useful.wine.apply(cleaning).apply(remove_accents)
    df_useful.winery=df_useful.winery.apply(cleaning).apply(remove_accents)

    # On créé la colonne cépage (NON UTILISEE A DATE)
    df_useful["cepage"]=df_useful.varietal.apply(lambda x: x.split(",")[0].strip())

    # On récupère le vintage à partir du nom de la cuvée (=wine dans le dataset),
    # on filtre et on change le type de la colonne vintage
    df_useful["vintage"]=df_useful.wine.apply(vintage_identification).astype(float)

    # On refait un dropna car on a pas toujours récupéré le millésime à partir du nom du wine
    df_useful.dropna(inplace=True)

    # On créé la colonne cuvée en retirant les infos déjà collectées dans
    # d'autres colonnes (domaine, vintage, cepage, appellation)
    df_useful["cuvee"]=df_useful.apply(lambda x: (' '.join(word for word in x["wine"].split(" ") if word not in x["winery"].split(" "))),axis=1)
    df_useful["cuvee"]=df_useful.apply(lambda x: (' '.join(word for word in x["cuvee"].split(" ") if word not in str(x["vintage"]))),axis=1)
    df_useful["cuvee"]=df_useful.apply(lambda x: (' '.join(word for word in x["cuvee"].split(" ") if word not in x["cepage"].split(" "))),axis=1)
    df_useful["cuvee"]=df_useful.apply(lambda x: (' '.join(word for word in x["cuvee"].split(" ") if (x["appellation"] is None) or word not in x["appellation"].split(" "))),axis=1)

    # On filtre le dataset
    df_useful=df_useful[df_useful.alcohol > 0.08]
    df_useful=df_useful[df_useful.alcohol < 0.25]
    df_useful=df_useful[df_useful.vintage > 1975]
    df_useful=df_useful[df_useful.vintage <= 2024]
    df_useful=df_useful[df_useful.country=="france"]

    # Rationnalisation du nombre d'occurences par feature en reprenant les fonctions de transformation
    ## Type de vin
    df_used=df_useful
    df_used.type_of_wine=df_used.type_of_wine.apply(classification_type_of_wine)

    ## Region
    df_used.region=df_used.region.apply(classification_region)

    # Reset index pour faciliter la lecture
    df_used.reset_index(drop=True,inplace=True)

    X_train = df_used.drop(columns=["wine","varietal","origine","review","country"])
    X_train = X_train.reindex(['winery','cuvee', 'vintage', 'type_of_wine', 'region',
               'appellation', 'cepage','alcohol', 'price_usd', 'rating'], axis=1)

    X_train.to_csv(params.path_X_train_loading)

    return X_train


def dataset_X_train():
    if os.path.exists(params.path_X_train_loading):
        # On loade le dataset cleané X_train
        return pd.read_csv(params.path_X_train_loading)
    else:
        return cleaning_general(get_data())
