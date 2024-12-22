import re
from wine_advisor.postprocessing.utils import find_closest_match, find_lcs
from wine_advisor.loading_resources import load_csv_gcp
from unidecode import unidecode

class WinePostProcessing():
    def __init__(self) -> None:
        self.wineries = load_csv_gcp(
            filepath='ocr_postprocessing',
            filename='winery.csv',
            saving_path='resources'
            )
        self.wine_types = load_csv_gcp(
            filepath='ocr_postprocessing',
            filename='type_of_wine_classification.csv',
            saving_path='resources'
            )
        self.appellations = load_csv_gcp(
            filepath='ocr_postprocessing',
            filename='appellation_classification.csv',
            saving_path='resources'
            )
        self.regions = load_csv_gcp(
            filepath='ocr_postprocessing',
            filename='region_classification.csv',
            saving_path='resources'
            )

    def preprocess_ocr_text(self, text_list:list) -> list:
        processed_text = []
        for text in text_list:
            # Replace accented characters
            text = unidecode(text)
            # Remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            # Convert to lowercase
            text = text.lower()
            # Strip extra spaces
            text = text.strip()
            processed_text.append(text)
        return processed_text

    def classify_wine(self, cleaned_text: list, dtext: dict=None):
        """
        Classifies wine attributes from cleaned OCR text by matching against datasets.

        Args:
            cleaned_text (list): List of preprocessed OCR text lines.

        Returns:
            dict: Dictionary containing classified attributes.

        labels = {
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
        data_test = {
            'winery': [None],
            'type_of_wine': [None],
            'alcohol': [None],
            'appellation': [None],
            'region': [None],
            'vintage': [None]
        }

        aoc_region = []
        if 1 in dtext['boxes']:
            aoc_region = self.preprocess_ocr_text(dtext['boxes'][1])
        elif 6 in dtext['boxes']:
            aoc_region = self.preprocess_ocr_text(dtext['boxes'][6])
        else:
            aoc_region = cleaned_text
        aoc_region = cleaned_text

        # Match winery
        tmp = []
        for line in cleaned_text:
            match = find_lcs(line, self.wineries)
            if match:
                tmp.append(match)

        if tmp:
            print(tmp)
            tmp = sorted(tmp, key = lambda t: len(t[0]), reverse=True)
            print(tmp)
            data_test['winery'] = [max(tmp, key = lambda t: t[1])[0]]

        # Match type of wine
        for line in cleaned_text:
            if line.lower() in self.wine_types:
                data_test['type_of_wine'] = [line.lower()]

        # Match appellation
        tmp = []
        for line in aoc_region:
            match = find_lcs(line, self.appellations)
            if match:
                tmp.append(match)

        if tmp:
            tmp = sorted(tmp, key = lambda t: len(t[0]), reverse=True)
            print(tmp)
            data_test['appellation'] = [max(tmp, key = lambda t: t[1])[0]]

        # Match region
        for line in aoc_region:
            words = line.split()
            for word in words:

                match = find_closest_match(word, self.regions)
                if match:
                    data_test['region'] = [match]
                    break

        # Extract vintage
        for line in cleaned_text:
            year_match = re.search(r'\b(18|19|20)\d{2}\b', line)
            if year_match:
                data_test['vintage'] = [int(year_match.group())]
                break

        # Extract alcohol percentage
        for line in cleaned_text:
            alcohol_match = re.search(r'(\d{1,2}[\.,]\d|\d{1,2})\s*%', line)
            if alcohol_match:
                alcohol_value = float(alcohol_match.group(1).replace(',', '.'))
                data_test['alcohol'] = [alcohol_value]
                break

        return data_test

    def text_to_dict(self, text:str, dtext:dict) -> dict:
        # Clean output OCR : suppression espaces excédentaires, caractères spéciaux
        cleaned_text = self.preprocess_ocr_text(text)

        # Classify wine attributes
        classified_data = self.classify_wine(cleaned_text, dtext)

        return classified_data

if __name__ == '__main__':
    from wine_advisor.ocr.ocr import TesseractOCR
    tess = TesseractOCR()
    postproc = WinePostProcessing()

    img = '../../tmp_data/results/20241201_172801/9_unwrapped_gray.jpg'
    ocr_output = tess.img_to_text_ocr(img)

    print(ocr_output)

    # Preprocess the OCR text
    # Split OCR output into lines
    cleaned_text = postproc.preprocess_ocr_text(ocr_output)

    # Classify wine attributes
    classified_data = postproc.classify_wine(cleaned_text)
    print(f"Classified Data: {classified_data}")
