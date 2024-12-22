from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import pickle
import os

from wine_advisor.classification import params
from wine_advisor.classification.dataset_preprocessing import preprocessing_train
from wine_advisor.classification.dataset_cleaning import *

def load_knn_model():
    if os.path.exists(params.model_knn_pkl_file):
        # On loade le modèle
        neigh = pickle.load(open(params.model_knn_pkl_file,"rb"))
        return neigh

def instantiation_NearestNeighbors():
    """
    Instatiation et fitting du NearestNeighbors modele avec en entrée
    un DataFrame X_preprocess preprocessé issu de dataset_processing.preprocessing
    """

    X_preprocess = preprocessing_train()

    # OPTION 1 : Un modèle a déjà été enregistré
    if os.path.exists(params.model_knn_pkl_file):
        # On loade le modèle
        return load_knn_model()

    # OPTION 2 : Aucun modèle préexistant n'a été enregistré
    else:
        # On instantie le modèle
        neigh = NearestNeighbors(n_neighbors=params.nb_neighbors_final_model)

        # On fit le modèle
        neigh.fit(X_preprocess)

        # On enregistre le modèle
        with open(params.model_knn_pkl_file, 'wb') as file:
            pickle.dump(neigh, file)

        return neigh

        # On enregistre le modèle
        with open(params.model_knn_pkl_file, 'wb') as file:
            pickle.dump(neigh, file)

        return neigh

def get_results(X_test_preprocess: pd.DataFrame):
    """
    Listing du top des alternatives de la bouteille de vin pris en photo
    """

    # On appelle le dataset Kaggle cleané pour le display final des top5
    # et on appelle le DataFrame Kaggle preprocessé pour entrainé le modèle
    X_train = dataset_X_train()

    # On instantie le modèle et on obtient les résultats
    neigh = instantiation_NearestNeighbors()
    distance, top_alternatives = neigh.kneighbors(X_test_preprocess, params.nb_alternatives, return_distance=True)


    results = (X_train.iloc[top_alternatives[0]].reset_index(drop=True)).merge(
        pd.DataFrame(np.round(distance,2).transpose(),columns=["distance"]),
        how="inner",left_index=True, right_index=True)

    return results
