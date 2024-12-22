import os
import pandas as pd
import pickle

from wine_advisor.classification import params

from wine_advisor.classification.cleaning_functions import remove_key_word_winery
from wine_advisor.loading_resources import load_model_gcp
from google.cloud import bigquery

class WineNearestNeighbour():
    def __init__(self, weight_neigh:str=None, weight_word2vec:str=None, min_max_scaler:str=None) -> None:

        # Poids du modèle Nearest Neighbours
        if weight_neigh:
            self.model_neigh = pickle.load(open(weight_neigh,"rb"))
        else:
            self.model_neigh = load_model_gcp(
                'knn_classifier_model.pkl',
                'classification_models',
                'models'
            )

        # Poids du modèle Word2Vec
        if weight_word2vec:
            self.w2v = pickle.load(open(weight_word2vec,"rb"))
        else:
            self.w2v = load_model_gcp(
                'word2vec_model.pkl',
                'classification_models',
                'models'
            )

        # Min Max Scaler
        if min_max_scaler:
            self.pipeline_min_max = pickle.load(open(min_max_scaler, "rb"))
        else:
            self.pipeline_min_max = load_model_gcp(
                'pipeline_min_max.pkl',
                'classification_models',
                'models'
            )

    def transform(self, X_test):
        """
        Preprocessing complet du dataset Kaggle cleané X_train issu du dataset_cleaning.cleaning_general
        """
        ## Alcohol, Vintage
        print(type(X_test))
        X_prep=pd.DataFrame(self.pipeline_min_max.transform(X_test[["alcohol","vintage"]]))
        X_prep.columns=["alcohol","vintage"]

        # On applique l'embedding du word2vec
        ## On applique l'embdegging directement sur les features : region, type_of_wine et appellation
        def embedding_features(df: pd.DataFrame):
            """
            Fonction permettant d'embedder des expressions intégrées dans un DataFrame
            """
            print(df)
            df1=df.copy()
            df1=df1.apply(lambda x: "" if x == None else x)
            df1=df1.apply(lambda x: x.split(" "))
            df1=df1.apply(lambda x: self.w2v[x])
            df1=df1.apply(lambda x: x[0])

            return df1

        X_prep["region"]=embedding_features(X_test["region"])
        X_prep["type_of_wine"]=embedding_features(X_test["type_of_wine"])
        X_prep["appellation"]=embedding_features(X_test["appellation"])

        ### On fait du feature engineering sur les 3eres colonnes textuelles
        X_prep["final"] = (X_prep.region + X_prep.type_of_wine + X_prep.appellation)/3

        ## On applique l'embedding sur les noms de domaines en retirant en plus les mots les plus courants
        X_prep["winery"]=embedding_features(X_test["winery"].apply(remove_key_word_winery))

        # On aggrège tous les résultats MinMaxScaler et Embedding dans un seul DataFrame
        X_preprocess=pd.DataFrame(X_prep["vintage"],columns=["vintage"])
        X_preprocess["alcohol"]=X_prep["alcohol"]

        for i in range (0,params.vector_size):
            X_preprocess[f'PC{i}'] = pd.DataFrame(X_prep["final"].apply(lambda x: x[i]))

        for i in range (0,params.vector_size):
            X_preprocess[f'PC{params.vector_size+i}'] = pd.DataFrame(X_prep["winery"].apply(lambda x: x[i]))

        return X_preprocess

    def get_infos(self, top_alternatives:list):
        query = f"""
            SELECT *
            FROM {os.getenv('GCP_PROJECT')}.{os.getenv('DATASET')}.{os.getenv('TABLE')}
            WHERE int64_field_0 IN ({",".join(map(str, top_alternatives))})
            """

        client = bigquery.Client(project=os.getenv('GCP_PROJECT'))
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        return df

    def predict(self, X_preprocessed):
        """
        Listing du top des alternatives de la bouteille de vin pris en photo
        """
        # On instantie le modèle et on obtient les résultats
        distance, top_alternatives = self.model_neigh.kneighbors(X_preprocessed, params.nb_alternatives, return_distance=True)

        results = self.get_infos(top_alternatives[0])
        results['distance'] = distance[0]

        # sauvegarde des résultats
        results.to_csv(f"{os.environ['UNET_RESULTS_PATH']}/recommandations.csv", index=False)
        return results

if __name__ == '__main__':
    nearest_neighbour = WineNearestNeighbour()

    data_test_2 = {
        'winery': ["berthier"],
        'type_of_wine': ["blanc"],
        'alcohol': [0.13],
        'appellation': ["pouilly fume"],
        'region': ['vallee de la loire'],
        'vintage': [2012]
    }

    X_test = nearest_neighbour.transform(pd.DataFrame(data_test_2))
    print(X_test)

    results = nearest_neighbour.predict(X_test)
    print(results)
