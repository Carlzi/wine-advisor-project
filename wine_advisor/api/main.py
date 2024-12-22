from wine_advisor.api.api_wine import *
from wine_advisor.classification.params import *

if __name__ == '__main__':
    predict(params.data_test_2["winery"][0],
            params.data_test_2["type_of_wine"][0],
            params.data_test_2["alcohol"][0],
            params.data_test_2["appellation"][0],
            params.data_test_2["region"][0],
            params.data_test_2["vintage"][0]
    )
