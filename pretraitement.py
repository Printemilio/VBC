import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, Dense, Flatten, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D
from pycocotools.coco import COCO
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile


def unzip_data(zip_path, extract_to):
    """
    Décompresse un fichier zip dans un dossier spécifié.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Finished unzipping {zip_path} into {extract_to}")

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Charge et prétraite une image pour le modèle.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # Normaliser les valeurs des pixels
    return image

def load_images_and_annotations(images_dir, annotations_file, img_size=(224, 224)):
    """
    Charge les images et leurs annotations à partir d'un répertoire et d'un fichier d'annotations COCO.
    """
    coco = COCO(annotations_file)
    dataset = []
    labels = []

    for img_id in coco.getImgIds():
        img_dict = coco.loadImgs(img_id)[0]
        annotation_ids = coco.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)
        path = os.path.join(images_dir, img_dict['file_name'])
        image = load_and_preprocess_image(path, img_size)
        # À adapter selon vos besoins, ici on prend simplement l'image
        dataset.append(image.numpy())
        # Ici, vous devez adapter les annotations à votre format attendu pour l'entraînement
        labels.append([annotation['category_id'] for annotation in annotations])

    return np.array(dataset), np.array(labels)

# Adapter les chemins selon votre environnement
images_zip_path = 'C:/Users/user/Downloads/train2017.zip'
annotations_zip_path = 'C:/Users/user/Downloads/stuff_annotations_trainval2017.zip'
images_dir = 'C:/Users/user/Desktop/Code/VBC/images'
annotations_dir = 'C:/Users/user/Desktop/Code/VBC/annotations'

# Décompression des fichiers zip
unzip_data(images_zip_path, images_dir)
unzip_data(annotations_zip_path, annotations_dir)