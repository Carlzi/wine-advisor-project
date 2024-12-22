import pandas as pd
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

from wine_advisor.classification import params
from wine_advisor.classification.cleaning_functions import remove_key_word_winery
from wine_advisor.classification.dataset_cleaning import *

def listing_words_for_w2v_calibration (X_train: pd.DataFrame):
    """
    Listing de tous les mots qui devront être embeddés par le Word2vec
    """

    X_agg=pd.concat([X_train['region'],X_train['type_of_wine'],X_train['appellation'],X_train['winery']])
    X_split=X_agg.apply(lambda x: x.split(" "))

    return [x for x in X_split.to_list()]


def load_w2v_model():
    """
    Téléchargement du modèle w2v calibré s'il existe déjà
    """

    if os.path.exists(params.model_w2v_pkl_file):
        # On loade le modèle
        wv = pickle.load(open(params.model_w2v_pkl_file,"rb"))
        return wv
    else:
        return None


def w2v_calibration (X_train: pd.DataFrame):
    """
    Instantiation et calibration du Word2vec
    """

    # OPTION 1 : Un modèle a déjà été enregistré
    if os.path.exists(params.model_w2v_pkl_file):
        # On loade le modèle
        return load_w2v_model()

    # OPTION 2 : Aucun modèle préexistant n'a été enregistré
    else:
        # On instantie le modèle
        word2vec = Word2Vec(listing_words_for_w2v_calibration(X_train),
                        vector_size=params.vector_size,min_count=1)
        wv = word2vec.wv

        # On enregistre le modèle
        with open(params.model_w2v_pkl_file, 'wb') as file:
            pickle.dump(wv, file)

        return wv


def embedding_features(df: pd.DataFrame):
    """
    Fonction permettant d'embedder des expressions intégrées dans un DataFrame
    """

    wv = load_w2v_model()

    df1=df.copy()
    df1=df1.apply(lambda x: x.split(" "))
    df1=df1.apply(lambda x: wv[x])
    df1=df1.apply(lambda x: x[0])

    return df1


def preprocessing_train():
    """
    Preprocessing complet du dataset Kaggle cleané X_train issu du dataset_cleaning.cleaning_general
    """

    # On appelle le X issu du dataset_cleaning.cleaning_general
    X_train = dataset_X_train()

    X_simplified=X_train.drop(columns=["cuvee","cepage","rating","price_usd","rating"])


    # On impute un MinMaxScaler sur les valeurs numériques
    num_transformer = MinMaxScaler()
    pipeline = Pipeline([('min_max_scaler', MinMaxScaler())])

    ## Alcohol, Vintage
    X_prep=pd.DataFrame(pipeline.fit_transform(X_simplified[["alcohol","vintage"]]))
    X_prep.columns=["alcohol","vintage"]

    # On applique l'embedding du word2vec
    ## On applique l'embdegging directement sur les features : region, type_of_wine et appellation
    X_prep["region"]=embedding_features(X_simplified["region"])
    X_prep["type_of_wine"]=embedding_features(X_simplified["type_of_wine"])
    X_prep["appellation"]=embedding_features(X_simplified["appellation"])

    ### On fait du feature engineering sur les 3eres colonnes textuelles
    X_prep["final"] = (X_prep.region + X_prep.type_of_wine + X_prep.appellation)/3

    ## On applique l'embedding sur les noms de domaines en retirant en plus les mots les plus courants
    X_prep["winery"]=embedding_features(X_simplified["winery"].apply(remove_key_word_winery))

    # On aggrège tous les résultats MinMaxScaler et Embedding dans un seul DataFrame
    X_preprocess=pd.DataFrame(X_prep["vintage"],columns=["vintage"])
    X_preprocess["alcohol"]=X_prep["alcohol"]

    for i in range (0,params.vector_size):
        X_preprocess[f'PC{i}'] = pd.DataFrame(X_prep["final"].apply(lambda x: x[i]))

    for i in range (0,params.vector_size):
        X_preprocess[f'PC{params.vector_size+i}'] = pd.DataFrame(X_prep["winery"].apply(lambda x: x[i]))

    return X_preprocess
