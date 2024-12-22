from ultralytics import YOLO
import numpy as np
import cv2
import torch
import os

from wine_advisor.segmentation.annotations import plot_bboxes, crop_bboxes
from wine_advisor.segmentation.contrast_image_transformation import (
    apply_full_contrast
    )
from wine_advisor.ocr.ocr import TesseractOCR

"""
rappel des classes possibles
{
    0: 'AlcoholPercentage',
    1: 'Appellation AOC DOC AVARegion',
    2: 'Appellation QualityLevel',
    3: 'CountryCountry',
    4: 'Distinct Logo',
    5: 'Established YearYear',
    6: 'Maker-Name',
    7: 'TypeWine Type',
    8: 'VintageYear'
}
"""

class WineYOLO():
    model = None
    labels = {}

    def __init__(self, weights: str=None) -> None:
        """Instancie un WineYolo en chargeant les poids à partir d'un fichier

        Args:
            weights (str): path vers les poids sauvegardés du modèle
        """
        if not weights:
            root_dir = os.path.dirname(__file__)
            weights = os.path.join(root_dir, 'weights/best.pt')
        self.model = YOLO(weights)
        self.labels = self.model.names

    def predict(self, data: str) -> torch.Tensor:
        """Fait la prédiction de la segmentation sur une photo. Renvoie les
        résultats sous la forme d'un Tensor.

        Args:
            data (str): path vers la photo à segmenter

        Returns:
            torch.Tensor: tensor de résultats
        """
        return self.model.predict(data)

    def handle_results(
        self,
        img: np.ndarray,
        results: torch.Tensor,
        saving_path: str=None,
        filename: str='raw'
        ) -> np.ndarray:
        """Traite les résultats en sortie de la prédiction.
        Sauvegarde l'image labellisée et les différents segments croppé dans un
        dossier <tmp_data>.

        Args:
            img (np.ndarray): path vers l'image concernée
            results (torch.Tensor): résultat de la prédiction : contient toutes les bbox
            saving_path (str, optional): path vers le dossier de sauvegarde. Defaults to None.

        Returns:
            np.ndarray: image labellisée
        """
        plot_bboxes(
            img,
            results[0].boxes.data,
            self.labels,
            saving_path=saving_path,
            filename=filename
            )

        boxes_list = crop_bboxes(
            image=img,
            boxes=results[0].boxes.data,
            image_name=filename, labels=self.labels,
            saving_path=saving_path,
            skipped_label=[4]
            )
        return boxes_list

    def ocr_boxes(self, boxes:list, tess:TesseractOCR):
        to_return = {
            "full_recompiled_text": [],
            "boxes": {
                # key : label
                # value : [texts]
            }
        }
        for value in boxes:
            box, label = value['img'], value['label']

            box = apply_full_contrast(box)

            # ocr du segment
            text = " ".join(tess.img_to_text_ocr(box))

            # sauvegarde de l'ocr dans un dictionnaire
            if value['label'] not in to_return['boxes']:
                to_return['boxes'][label] = []
            to_return['boxes'][label].append(text)

            # concaténation aux segments déjà passés dans l'ocr pour reconstituer
            # le texte complet
            to_return['full_recompiled_text'].append(text)

        return to_return

if __name__ == '__main__':
    yolo = WineYOLO()
    tess = TesseractOCR()

    img = '../../data/yolo/511_jpg.rf.e992e9df1f71acec9434979ade514fee.jpg'
    img = '../../data/yolo/120257_jpg.rf.c6565b5499d24c7eba5d82d548e7bc1d.jpg'
    img_array = cv2.imread(img)

    results = yolo.predict(img)
    boxes_list = yolo.handle_results(img_array, results, saving_path='../../tmp_data')

    print("OCR SUR IMAGE", tess.img_to_text_ocr(img))

    dtext = yolo.ocr_boxes(boxes_list, tess)

    print("OCR SUR BBOX", dtext['full_recompiled_text'])
