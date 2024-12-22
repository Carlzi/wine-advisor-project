from wine_advisor.segmentation.yolo import WineYOLO
from wine_advisor.preprocessing.unet import WineUnet
from wine_advisor.ocr.ocr import TesseractOCR
from wine_advisor.postprocessing.classifier import WinePostProcessing
from wine_advisor.classification.nearest_neighbour import WineNearestNeighbour
from wine_advisor.classification.test_preprocessing import WineDataImputing

import cv2
import pandas as pd
import os

unet = WineUnet()
yolo = WineYOLO()
tess = TesseractOCR()
postproc = WinePostProcessing()
nearest_neighbour = WineNearestNeighbour()
data_imputing = WineDataImputing()

def wine_advisor_from_img(img_path:str):
    # Photo -> Étiquette isolée et aplatie
    prediction = unet.photo_to_img(img_path)

    full_text = []
    i = 0
    # Pour chaque image : OCR et Segmentation
    for elmt in prediction:
        cv2.imwrite(f'temporary_file_{i}_1.jpg', elmt[1])
        i += 1
        # UNET
        ## OCR sur étiquette complète
        text_full_img = tess.img_to_text_ocr(elmt[1])
        full_text.extend(text_full_img)

        ## OCR sur étiquette labelisée
        boxes = yolo.predict(elmt[1])
        boxes_list = yolo.handle_results(
            elmt[1],
            boxes,
            saving_path=os.environ['UNET_RESULTS_PATH']
            )

        dtext = yolo.ocr_boxes(boxes_list, tess)

        full_text.extend(dtext['full_recompiled_text'])

        # SANS UNET
        raw_img = cv2.imread(img_path)
        ## OCR sur photo complète
        text_full_img = tess.img_to_text_ocr(raw_img)
        full_text.extend(text_full_img)

        ## OCR sur photo labelisée
        boxes = yolo.predict(raw_img)
        boxes_list = yolo.handle_results(
            raw_img,
            boxes,
            saving_path=os.environ['UNET_RESULTS_PATH']
            )

        dtext_raw = yolo.ocr_boxes(boxes_list, tess)
        full_text.extend(dtext_raw['full_recompiled_text'])


    X_input = postproc.text_to_dict(full_text, dtext)

    print("Input from photo >>\t",X_input)

    X_imputed = data_imputing.impute(X_input)

    X_test = nearest_neighbour.transform(pd.DataFrame(X_imputed))

    results = nearest_neighbour.predict(X_test)

    return results

def wine_advisor_img_to_dict(img_path:str):
    # Photo -> Étiquette isolée et aplatie
    prediction = unet.photo_to_img(img_path)

    full_text = []
    i = 0
    # Pour chaque image : OCR et Segmentation
    for elmt in prediction:
        cv2.imwrite(f'temporary_file_{i}_1.jpg', elmt[1])
        i += 1
        # UNET
        ## OCR sur étiquette complète
        text_full_img = tess.img_to_text_ocr(elmt[1])
        full_text.extend(text_full_img)

        ## OCR sur étiquette labelisée
        boxes = yolo.predict(elmt[1])
        boxes_list = yolo.handle_results(
            elmt[1],
            boxes,
            filename='unet',
            saving_path=os.environ['UNET_RESULTS_PATH']
            )

        dtext = yolo.ocr_boxes(boxes_list, tess)

        full_text.extend(dtext['full_recompiled_text'])

        # SANS UNET
        raw_img = cv2.imread(img_path)
        ## OCR sur photo complète
        text_full_img = tess.img_to_text_ocr(raw_img)
        full_text.extend(text_full_img)

        ## OCR sur photo labelisée
        boxes = yolo.predict(raw_img)
        boxes_list = yolo.handle_results(
            raw_img,
            boxes,
            saving_path=os.environ['UNET_RESULTS_PATH']
            )

        dtext_raw = yolo.ocr_boxes(boxes_list, tess)
        for label in dtext_raw['boxes']:
            if label in dtext['boxes']:
                dtext['boxes'][label].extend(dtext_raw['boxes'][label])
            else:
                dtext['boxes'][label] = dtext_raw['boxes'][label]
        full_text.extend(dtext_raw['full_recompiled_text'])

    print(dtext)

    X_input = postproc.text_to_dict(full_text, dtext)

    return X_input

def wine_advisor_dict_to_reco(infos:dict):
    X_imputed = data_imputing.impute(infos)

    X_test = nearest_neighbour.transform(pd.DataFrame(X_imputed))

    results = nearest_neighbour.predict(X_test)

    return results

img_path = './raw_data/IMG_6710.jpg'
# img_path = './raw_data/20241203_132626.jpg'
img_path = './raw_data/IMG_0629.jpg'
img_path = './raw_data/IMG_6711.jpg'

infos = wine_advisor_img_to_dict(img_path)

print(infos)


# infos = {
#     'winery': ['duc de castellac'],
#     'type_of_wine': ['blanc'],
#     'alcohol': [0.115],
#     'appellation': ['cotes de bergerac'],
#     'region': ['sud ouest'],
#     'vintage': [2023]
#     }
# infos = {
#     'winery': ['chateau fourcas dupre'],
#     'type_of_wine': ['rouge'],
#     'alcohol': [0.13],
#     'appellation': ['haut medoc'],
#     'region': ['bordeaux'],
#     'vintage': [2018]
#     }

results = wine_advisor_dict_to_reco(infos)

print(results)
