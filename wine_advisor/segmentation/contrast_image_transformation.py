import cv2
import numpy as np
from wine_advisor.ocr.ocr import TesseractOCR

def apply_gamma(img, gamma=1.5):
    image = img.copy()
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)

    return res

def apply_contrast(img, alpha=1.46, beta=2.13):
    # print(img.shape)
    image = img.copy()
    new_image = np.zeros(image.shape, image.dtype)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
            else:
                new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)

    return new_image

def apply_full_contrast(
    img: np.ndarray,
    alpha: float=1.46,
    beta: float=2.13,
    gamma: float=1.5) -> np.ndarray:
    """Applique une transformation de l'image en jouant sur le contraste
    (alpha et beta) et la luminosité (gamma)

    Args:
        img (np.ndarray): image à modifier
        alpha (float, optional): niveau alpha. Defaults to 1.46.
        beta (float, optional): niveau beta. Defaults to 2.13.
        gamma (float, optional): niveau gamma. Defaults to 1.5.

    Returns:
        np.ndarray: _description_
    """
    img_contrast = apply_contrast(img, alpha, beta)
    img_transformed = apply_gamma(img_contrast, gamma)

    return img_transformed

if __name__ == '__main__':
    img = cv2.imread('../../tmp_data/tmp_test_box1_typewine_type.jpg')

    res = apply_full_contrast(img)
    cv2.imwrite('../../tmp_data/temporary_contrast_file.jpg', res)

    tess = TesseractOCR()

    text = tess.img_to_text_ocr(img)
    print(text)
    # text = tess.img_to_text_ocr(res)
    # print(text)
