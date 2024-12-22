import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import pickle
import os

from wine_advisor.classification.dataset_cleaning import *
from wine_advisor.classification.cleaning_functions import remove_key_word_winery
from wine_advisor.classification.dataset_preprocessing import embedding_features
from wine_advisor.classification import params

from wine_advisor.loading_resources import load_model_gcp

def imputing_function_region():
    """
    Création d'une fonction d'imputing en cas d'absence de la région dans le DataFrame X_test
    On se base sur le DataFrame X_train issu du dataset_cleaning.cleaning_general
    """

    X_train = dataset_X_train()

    # On fait l'hypothèse que l'on aura systématiquement le type de vin et l'appellation
    data_model_imputing=X_train[["appellation","type_of_wine","vintage","region","alcohol"]].dropna()

    X_model_imputing=data_model_imputing[["appellation","type_of_wine"]]
    y_region=data_model_imputing["region"]

    # On fait une modélisation KNClassifier
    pipeline_model_region = make_pipeline(OneHotEncoder(handle_unknown='ignore',
                                                        sparse_output = False),
                                          KNeighborsClassifier(n_neighbors=params.n_neighbors_imputing))
    pipeline_model_region.fit(X_model_imputing,y_region)

    # Export Pipeline as pickle file
    with open(params.pipeline_region_pkl_file, "wb") as file:
        pickle.dump(pipeline_model_region, file)

    return pipeline_model_region


def imputing_function_alcohol():
    """
    Création d'une fonction d'imputing en cas d'absence du degré d'alcool dans le DataFrame X_test
    On se base sur le DataFrame X issu du dataset_cleaning.cleaning_general
    """

    X_train = dataset_X_train()

    # On fait l'hypothèse que l'on aura systématiquement le type de vin et l'appellation
    data_model_imputing=X_train[["appellation","type_of_wine","vintage","region","alcohol"]].dropna()

    X_model_imputing=data_model_imputing[["appellation","type_of_wine"]]
    y_alcohol=data_model_imputing["alcohol"]

    # On fait une modélisation KNRegressor
    pipeline_model_alcohol = make_pipeline(OneHotEncoder(handle_unknown='ignore',
                                                         sparse_output = False),
                                           KNeighborsRegressor(n_neighbors=params.n_neighbors_imputing))
    pipeline_model_alcohol.fit(X_model_imputing,y_alcohol)

    # Export Pipeline as pickle file
    with open(params.pipeline_alcohol_pkl_file, "wb") as file:
        pickle.dump(pipeline_model_alcohol, file)

    return pipeline_model_alcohol


def imputing_function_vintage():
    """
    Création d'une fonction d'imputing en cas d'absence du vintage dans le DataFrame X_test
    On se base sur le DataFrame X issu du dataset_cleaning.cleaning_general
    """

    X_train = dataset_X_train()

    # On fait l'hypothèse que l'on aura systématiquement le type de vin et l'appellation
    data_model_imputing=X_train[["appellation","type_of_wine","vintage","region","alcohol"]].dropna()

    X_model_imputing=data_model_imputing[["appellation","type_of_wine"]]
    y_vintage=data_model_imputing["vintage"]

    # On fait une modélisation KNRegressor
    pipeline_model_vintage = make_pipeline(OneHotEncoder(handle_unknown='ignore',
                                                         sparse_output = False),
                                           KNeighborsRegressor(n_neighbors=params.n_neighbors_imputing))
    pipeline_model_vintage.fit(X_model_imputing,y_vintage)

    # Export Pipeline as pickle file
    with open(params.pipeline_vintage_pkl_file, "wb") as file:
        pickle.dump(pipeline_model_vintage, file)

    return pipeline_model_vintage


def pipeline_min_max_scaler():
    """
    Création d'un pipeline min_max_scaler
    """

    X_train = dataset_X_train()

    pipeline_min_max = Pipeline([('min_max_scaler', MinMaxScaler())])

    ## Alcohol, Vintage
    pipeline_min_max.fit(X_train[["alcohol","vintage"]])

    # Export Pipeline as pickle file
    with open(params.pipeline_min_max_pkl_file, "wb") as file:
        pickle.dump(pipeline_min_max, file)

    return pipeline_min_max


def load_imputing_functions():
    l = []
    # if os.path.exists(params.pipeline_min_max_pkl_file):
    if os.path.exists(params.pipeline_min_max_pkl_file):
        # On loade le modèle
        pipeline_min_max = pickle.load(open(params.pipeline_min_max_pkl_file,"rb"))
        l.append(pipeline_min_max)

    # if os.path.exists(params.pipeline_region_pkl_file):
    if os.path.exists(params.pipeline_region_pkl_file):
        # On loade le modèle
        pipeline_model_region = pickle.load(open(params.pipeline_region_pkl_file,"rb"))
        l.append(pipeline_model_region)

    if os.path.exists(params.pipeline_alcohol_pkl_file):
        # On loade le modèle
        pipeline_model_alcohol = pickle.load(open(params.pipeline_alcohol_pkl_file,"rb"))
        l.append(pipeline_model_alcohol)

    if os.path.exists(params.pipeline_vintage_pkl_file):
        # On loade le modèle
        pipeline_model_vintage = pickle.load(open(params.pipeline_vintage_pkl_file,"rb"))
        l.append(pipeline_model_vintage)

    return l


def preprocessing_test_imputing(X_test_simplified: pd.DataFrame):
    """
    Preprocess du DataFrame X_test - Imputing
    """

    # 1e ETAPE : On appelle les 3 imputers définis au dessus pour gérér les données manquantes
    # de région, de degré d'alcool ou de vintage
    l = load_imputing_functions()

    pipeline_model_region = l[1]
    pipeline_model_alcohol = l[2]
    pipeline_model_vintage = l[3]

    X_test_simplified_imputed = X_test_simplified.copy()
    if X_test_simplified["region"].isna()[0]:
        X_test_simplified_imputed["region"] = pd.DataFrame(
            pipeline_model_region.predict(X_test_simplified[["appellation","type_of_wine"]]))
    else:
        X_test_simplified_imputed["region"]

    if X_test_simplified["alcohol"].isna()[0] or X_test_simplified["alcohol"][0] < 0.05 or X_test_simplified["alcohol"][0] > 0.25:
        X_test_simplified_imputed["alcohol"] = pd.DataFrame(
            pipeline_model_alcohol.predict(X_test_simplified[["appellation","type_of_wine"]]))
    else:
        X_test_simplified_imputed["alcohol"]

    if X_test_simplified["vintage"].isna()[0] or X_test_simplified["vintage"][0] > 2024:
        X_test_simplified_imputed["vintage"] = pd.DataFrame(
            pipeline_model_vintage.predict(X_test_simplified[["appellation","type_of_wine"]]))
    else:
        X_test_simplified_imputed["vintage"]

    return X_test_simplified_imputed


def preprocessing_test_embedding(X_test_simplified_imputed: pd.DataFrame):
    """
    Preprocess du DataFrame X_test - Embedding
    """

    # 2e ETAPE : On applique le même preprocess que celui appliqué a X_train
    ## On impute un MinMaxScaler sur les valeurs numériques calibré sur X_train
    l = load_imputing_functions()

    pipeline_min_max = l[0]

    ## Alcohol, Vintage
    X_test_simplified_imputed["winery"]=X_test_simplified_imputed.winery.apply(remove_key_word_winery)

    X_test_embedded=pd.DataFrame(pipeline_min_max.transform(X_test_simplified_imputed[["alcohol","vintage"]]))
    X_test_embedded.columns=["alcohol","vintage"]

    # On applique l'embedding du word2vec calibré sur X_train
    ## On applique l'embdegging directement sur les features : region, type_of_wine et appellation
    X_test_embedded["region"]=embedding_features(X_test_simplified_imputed["region"])
    X_test_embedded["type_of_wine"]=embedding_features(X_test_simplified_imputed["type_of_wine"])
    X_test_embedded["appellation"]=embedding_features(X_test_simplified_imputed["appellation"])
    X_test_embedded["winery"]=embedding_features(X_test_simplified_imputed["winery"])

    ### On fait du feature engineering sur les 3eres colonnes textuelles
    X_test_embedded["final"] = (X_test_embedded.region + X_test_embedded.type_of_wine
                                + X_test_embedded.appellation)/3

    # On aggrège tous les résultats MinMaxScaler et Embedding dans un seul DataFrame
    X_test_preprocess=pd.DataFrame(X_test_embedded["vintage"],columns=["vintage"])
    X_test_preprocess["alcohol"]=X_test_embedded["alcohol"]

    for i in range (0,params.vector_size):
        X_test_preprocess[f'PC{i}'] = pd.DataFrame(
            X_test_embedded["final"].apply(lambda x: x[i]))

    for i in range (0,params.vector_size):
        X_test_preprocess[f'PC{params.vector_size+i}'] = pd.DataFrame(
            X_test_embedded["winery"].apply(lambda x: x[i]))

    return X_test_preprocess


class WineDataImputing():
    def __init__(self, region:str=None, alcohol:str=None, vintage:str=None) -> None:
        print("Load Imputing Pipeline")

        # Poids du modèle Nearest Neighbours
        if region:
            self.pipeline_region = pickle.load(open(region,"rb"))
        else:
            self.pipeline_region = load_model_gcp(
                'pipeline_region.pkl',
                'classification_models',
                'models'
            )

        if alcohol:
            self.pipeline_alcohol = pickle.load(open(alcohol,"rb"))
        else:
            self.pipeline_alcohol = load_model_gcp(
                'pipeline_alcohol.pkl',
                'classification_models',
                'models'
            )

        if vintage:
            self.pipeline_vintage = pickle.load(open(vintage,"rb"))
        else:
            self.pipeline_vintage = load_model_gcp(
                'pipeline_vintage.pkl',
                'classification_models',
                'models'
            )

    def impute(self, X_test:dict):
        """
        Preprocess du DataFrame X_test - Imputing
        """
        X_test = pd.DataFrame(X_test)
        X_imputed = X_test.copy()

        # print(pd.Series(X_test['region']), pd.Series(X_test['region']).isna()[0])

        if X_test["region"].isna()[0]:
            X_imputed["region"] = pd.DataFrame(
                self.pipeline_region.predict(X_test[["appellation","type_of_wine"]]))
        else:
            X_imputed["region"]

        if X_test["alcohol"].isna()[0] or X_test["alcohol"][0] < 0.05 or X_test["alcohol"][0] > 0.25:
            X_imputed["alcohol"] = pd.DataFrame(
                self.pipeline_alcohol.predict(X_test[["appellation","type_of_wine"]]))
        else:
            X_imputed["alcohol"]

        if X_test["vintage"].isna()[0] or X_test["vintage"][0] > 2024:
            X_imputed["vintage"] = pd.DataFrame(
                self.pipeline_vintage.predict(X_test[["appellation","type_of_wine"]]))
        else:
            X_imputed["vintage"]

        return X_imputed
