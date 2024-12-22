import pytesseract
import os
import pandas as pd

class TesseractOCR():
    def __init__(self) -> None:
        root_dir = os.path.dirname(__file__)
        os.environ['TESSDATA_PREFIX'] = os.path.join(root_dir, "tessdata")

    def clean_infos(infos: list) -> pd.Series:
        """
        /* TODO */

        Renvoie les informations trouvées par OCR en pd.Series
        pour être exploitées par un modèle de classification

        Args:
            infos (list): données en sortie de l'OCR

        Returns:
            pd.Series: informations à passer au modèle de classification
        """
        return dict()

    def img_to_text_ocr(self, img_path: str, lang: str='fra') -> list:
        """À partir du path d'une photo d'étiquette préprocessée, appelle la
        librairie tesseract pour faire un OCR et renvoyer les informations
        trouvées

        Args:
            img_path (str): path vers une image
            lang (str) : fra ou eng, indique la langue à charger pour le modèle

        Returns:
            dict: informations récupérées sur l'étiquette
        """
        text = pytesseract.image_to_string(img_path, lang=lang).split('\n')
        text = [t for t in text if t]

        return text

if __name__ == '__main__':
    tess = TesseractOCR()

    text = tess.img_to_text_ocr('../../data/ocr/test.png')
    print(text)
