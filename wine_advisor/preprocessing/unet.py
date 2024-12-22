from wine_advisor.preprocessing.wineReader.utils import (
    load_label_to_read,
    clean_results_folder,
    setup_paths
)
from wine_advisor.preprocessing.wineReader.model import Unet
from wine_advisor.preprocessing.wineReader.labelVision import labelVision

import os
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model

# Custom wrapper for Conv2DTranspose to ignore the 'groups' argument
class CustomConv2DTranspose(Conv2DTranspose):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

class WineUnet():
    def __init__(self) -> None:
        # clean results folder
        setup_paths()
        clean_results_folder()

    def photo_to_img(self, img_path:str=None) -> list:
        # load source img to read and unet inputs
        X, srcs, fileNames = load_label_to_read(img_path)

        # load trained model
        # print("LOAD UNET MODEL")
        root_dir = os.path.dirname(__file__)
        model = load_model(
            os.path.join(root_dir,'models','unet.h5'),
            custom_objects={"Conv2DTranspose": CustomConv2DTranspose}
        )

        # get U-net label predictions
        # print("GET UNET MODEL PREDICTION")
        unet = Unet()
        unet_output = unet.predict(X, model, fileNames)

        # read labels
        # print("GET LABELVISION MODEL PREDICTION")
        label = labelVision()
        labels_clean = label.readLabels(unet_output, srcs, fileNames)

        return labels_clean
