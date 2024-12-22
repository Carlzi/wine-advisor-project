import shutil
from tqdm import tqdm
import os
import cv2
import numpy as np
import urllib.request

def setup_paths():
    root_dir = os.path.dirname(__file__)

    os.environ['UNET_RESULTS_PATH'] = os.path.join(root_dir, '..', '..', '..', os.getenv('RESULTS_PATH'))
    os.environ['UNET_TO_READ_PATH'] = os.path.join(root_dir, '..', '..', '..', os.getenv('TO_READ_PATH'))

    if os.path.exists(os.environ['UNET_RESULTS_PATH']):
        shutil.rmtree(os.environ['UNET_RESULTS_PATH'])
    os.makedirs(os.environ['UNET_TO_READ_PATH'], exist_ok=True)
    os.makedirs(os.environ['UNET_RESULTS_PATH'], exist_ok=True)

def clean_results_folder():
    # rm results folder and all folder / files in it
    if os.path.exists(os.environ['UNET_RESULTS_PATH']):
        shutil.rmtree(os.environ['UNET_RESULTS_PATH'])
    # make a new one
    os.makedirs(os.environ['UNET_RESULTS_PATH'], exist_ok=True)

def load_label_to_read(img_path:str=None) -> tuple:
    fileNames = []
    srcs = []
    X = []

    if img_path: # to debug
        fileNames = [img_path.rsplit( ".",1)[0].split('/')[-1]]
        path = os.path.join(os.environ['UNET_RESULTS_PATH'], fileNames[0])
        if not os.path.exists(path):
            os.mkdir(path)

        srcs = [cv2.imread(img_path)]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256))
        X = [img]

    else:
        for file in tqdm(os.listdir(os.environ['UNET_TO_READ_PATH'])):
            if file.startswith('.'):
                pass
            else:
                #buid a file structure for results
                filename = file.rsplit( ".", 1)[0]
                fileNames.append(filename)
                parent_dir = os.environ['UNET_RESULTS_PATH']
                path = os.path.join(parent_dir, filename)
                if not os.path.exists(path):
                    os.mkdir(path)

                src=cv2.imread(os.environ['UNET_TO_READ_PATH']+file)
                srcs.append(src)
                img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
                img=cv2.resize(img,(256,256))
                X.append(img)

                cv2.imwrite(path + "/" + "0_src.jpg", src)
                cv2.imwrite(path + "/" + "1_unet.jpg", img)
    X=np.array(X)

    return X, srcs, fileNames

def img_url_to_input_unet(url):

    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    X=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X=cv2.resize(X,(256,256))

    return np.array([X]), img
