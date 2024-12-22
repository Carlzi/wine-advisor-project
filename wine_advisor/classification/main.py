import os
import pandas as pd

from wine_advisor.classification import params
from wine_advisor.classification.dataset_cleaning import dataset_X_train
from wine_advisor.classification import test_preprocessing
from wine_advisor.classification import knn_model

if __name__ == '__main__':

    test_preprocessing.pipeline_min_max_scaler()
    test_preprocessing.imputing_function_region()
    test_preprocessing.imputing_function_alcohol()
    test_preprocessing.imputing_function_vintage()

    X_test=params.X_test_2

    print("____________________________________________________________________________")
    print("__________________________________ X_test___________________________________")
    print(X_test)
    print("____________________________________________________________________________")

    X_test_simplified_imputed = test_preprocessing.preprocessing_test_imputing(X_test)

    print("____________________________________________________________________________")
    print("_______________________________ X_test_imputed______________________________")
    print(X_test_simplified_imputed)

    X_test_preprocess = test_preprocessing.preprocessing_test_embedding(X_test_simplified_imputed)

    print("____________________________________________________________________________")
    print("______________________________Top10 Alternatives____________________________")
    print(f"{knn_model.get_results(X_test_preprocess)}")
