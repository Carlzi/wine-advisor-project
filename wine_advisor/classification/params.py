import pandas as pd
import os

#PARAMS
path_dataset="/Users/charles.dreyfus/code/ChDre24/wine_advisor/raw_data/wine_review_kaggle_dataset.csv"
path_X_train_loading = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/raw_data/X_train.csv"
vector_size=5
nb_neighbors_final_model=10
nb_alternatives=10
n_neighbors_imputing=20
REPO_PATH='/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor'
model_knn_pkl_file = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor/classification/knn_classifier_model.pkl"
model_w2v_pkl_file = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor/classification/word2vec_model.pkl"
#CLASSIFICATION_PATH = os.path.join(REPO_PATH,"wine_advisor", "classification")
pipeline_region_pkl_file = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor/classification/pipeline_region.pkl"
#pipeline_region_pkl_file = os.path.join(CLASSIFICATION_PATH, "pipeline_region.pkl")
pipeline_alcohol_pkl_file = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor/classification/pipeline_alcohol.pkl"
pipeline_vintage_pkl_file = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor/classification/pipeline_vintage.pkl"
pipeline_min_max_pkl_file = "/Users/charles.dreyfus/code/ChDre24/wine_advisor/wine_advisor/classification/pipeline_min_max.pkl"


data_test_0 = {
    'winery': ["guigal"],
    'type_of_wine': ["rouge"],
    'alcohol': [None],
    'appellation': ["cote rotie"],
    'region':[None],
    'vintage':[None]}

X_test_0=pd.DataFrame(data_test_0)

data_test_1 = {
    'winery': ["ribeauville"],
    'type_of_wine': ["rouge"],
    'alcohol': [None],
    'appellation': ["alsace"],
    'region':['alsace'],
    'vintage':[int(2019)]}

X_test_1=pd.DataFrame(data_test_1)

data_test_2 = {
    'winery': ["berthier"],
    'type_of_wine': ["blanc"],
    'alcohol': [0.13],
    'appellation': ["pouilly fume"],
    'region':['vallee de la loire'],
    'vintage':[2012]}

X_test_2=pd.DataFrame(data_test_2)
