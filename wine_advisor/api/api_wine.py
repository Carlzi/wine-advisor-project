from wine_advisor.segmentation.yolo import WineYOLO
from wine_advisor.preprocessing.unet import WineUnet
from wine_advisor.ocr.ocr import TesseractOCR
from wine_advisor.postprocessing.classifier import WinePostProcessing
from wine_advisor.classification.nearest_neighbour import WineNearestNeighbour
from wine_advisor.classification.test_preprocessing import WineDataImputing
from wine_advisor.api.intermediary import *

import cv2
import pandas as pd
import os
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

unet = WineUnet()
yolo = WineYOLO()
tess = TesseractOCR()
postproc = WinePostProcessing()
nearest_neighbour = WineNearestNeighbour()
data_imputing = WineDataImputing()

@app.get("/predict")
def predict(
        winery: str,  # domaine des berthiers
        type_of_wine: str,    # rouge
        alcohol: float,     # 0.13
        appellation: str,   # pouilly fume
        region: str,    # vallee de la loire
        vintage: int    # 2019
    ):
    if alcohol == 0.0:
        alcohol = None
    X_test = pd.DataFrame(locals(), index=[0])
    results = wine_advisor_dict_to_reco(X_test)
    print(results)
    return results.transpose().to_dict()


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END


@app.post("/uploadfile_identification")
async def uploadfile_identification(file: UploadFile = File(...)):

    if file.content_type not in ("image/png","image/jpg","image/jpeg"):
        return {"error_message": "please load a picture"}

    else:
        # Print information about the file to the console
        contents = await file.read()

        nparr = np.fromstring(contents, np.uint8)
        cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

        directory=os.environ['UNET_TO_READ_PATH']
        file_name="new_image.jpg"

        os.chdir(directory)
        path_file = os.path.join(directory,file_name)
        cv2.imwrite(file_name,cv2_img)

        infos = wine_advisor_img_to_dict(path_file)
        # print(infos)
        return infos


@app.post("/uploadfile_alternatives")
async def upload_file_alternatives(file: UploadFile = File(...)):

    if file.content_type not in ("image/png","image/jpg","image/jpeg"):
        return {"error_message": "please load a picture"}

    else:
        # Print information about the file to the console
        contents = await file.read()

        nparr = np.fromstring(contents, np.uint8)
        cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

        directory=os.environ['UNET_TO_READ_PATH']
        file_name="new_image.jpg"

        os.chdir(directory)
        path_file = os.path.join(directory,file_name)
        cv2.imwrite(file_name,cv2_img)

        infos = wine_advisor_img_to_dict(path_file)
        results=wine_advisor_dict_to_reco(infos).drop(columns=["int64_field_0"]).transpose().to_dict()
        # print(results)
        return results
