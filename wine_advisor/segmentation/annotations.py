import cv2
import numpy as np
import torch

def box_label(
    image: np.ndarray, box: torch.Tensor,
    label: str=None,
    color: tuple=(128, 128, 128),
    txt_color: tuple=(255, 255, 255)
    ) -> np.ndarray:
    """Affiche une seule de ces bounding boxes sur une image
        Affiche le label de l'objet détecté

    Args:
        image (np.ndarray): image au format array
        box (torch.Tensor): bbox (x, y, w, h)
        label (str, optional): label de la bbox. Defaults to None.
        color (tuple, optional): couleur associée au label. Defaults to (128, 128, 128).
        txt_color (tuple, optional): couleur du texte. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: renvoie l'image labellisée
    """
    image = image.copy()

    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    # Si le label est disponible, on l'ajoute sur l'image
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        image = cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        image = cv2.putText(
            image,
            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA
            )

    return image

def plot_bboxes(
    image: np.ndarray, boxes: torch.Tensor,
    labels: list,
    score: bool=True, conf: list=None,
    saving_path: str=None,
    filename: str='raw'
    ) -> np.ndarray:
    """
    Fonction qui plot toutes les bboxes sur une même image

    Args:
        image (np.ndarray): image au format np.ndarray (lecture avec opencv)
        boxes (torch.Tensor): bbox obtenues grâce à yolo
        labels (list, optional): liste des labels possibles
        score (bool, optional): ajout des scores aux labels. Defaults to True.
        conf (float, optional): indique un seuil à partir duquel afficher les bbox. Defaults to None.

    Returns:
        np.ndarray: image labellisée
    """

    image = image.copy()
    colors = [
        (89, 161, 197),
        (67, 161, 255),
        (19, 222, 24),
        (186, 55, 2),
        (167, 146, 11),
        (190, 76, 98),
        (130, 172, 179),
        (115, 209, 128),
        (204, 79, 135)
    ]
    #plot each boxes
    for box in boxes:
      #add score in label if score=True
      if score :
        label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
      else :
        label = labels[int(box[-1])]
      #filter every box under conf threshold if conf threshold setted
      if conf :
        if box[-2] > conf:
          color = colors[int(box[-1])]
          image = box_label(image, box, label, color)
      else:
        color = colors[int(box[-1])]
        image = box_label(image, box, label, color)

    if saving_path:
        cv2.imwrite(f"{saving_path}/bbox_{filename}.jpg", image)

    return image

def crop_bboxes(
    image: np.ndarray,
    boxes: torch.Tensor, labels: dict,
    padding=20,
    image_name: str='raw', saving_path: str=None,
    skipped_label:list=None
    ) -> list:
    """A partir d'une image et de ses bbox, découpe les différents segments.
    Les enregistre dans un dossier si <saving_path> est renseigné.

    Args:
        image (np.ndarray): image en array
        image_name (str): nom de l'image (pour la sauvegarde des fichiers)
        boxes (torch.Tensor): liste des bbox (results[0].boxes.data)
        labels (dict): liste des labels (pour la sauvegarde des fichiers)
        padding (int, optional): padding ajouté à la découpe. Defaults to 20.
        saving_path (str, optional): lien vers le dossier de sauvegarde. Defaults to None.
    """
    image = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    cropped_images = list()

    if saving_path:
        cv2.imwrite(f"{saving_path}/gray_{image_name}.jpg", gray)

    i = 0
    for segment in boxes:
        x, y, w, h, conf, label = tuple(map(int, segment.tolist()))
        x = x-1 if x-1 > 0 else x
        y = y-1 if y-1 > 0 else y
        h = h+1 if h+1 < gray.shape[0] else h
        w = w+1 if w+1 < gray.shape[1] else w

        if skipped_label and label in skipped_label:
            pass
        else:
            label_str = labels[label].lower().replace(' ', '_')
            cropped_image = gray[y:h, x:w]
            color = cropped_image[0,0].tolist()
            new_im = cv2.copyMakeBorder(cropped_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                value=color)
            cv2.imwrite(f'{saving_path}/tmp_{image_name}_box{i}_{label_str}.jpg', new_im )
            i += 1
            cropped_images.append({'label': label, 'label_str': label_str, 'img': new_im})

    return cropped_images
