from tensorflow.keras.preprocessing.image import save_img
import os

class Unet:
    def predict(self, X, model, fileNames):

        mask_vectors = model.predict(X)

        for filename, mask in zip(fileNames, mask_vectors):
            save_img(os.environ['UNET_RESULTS_PATH'] + filename + "/2_raw_predict.jpg", mask)

        return mask_vectors
