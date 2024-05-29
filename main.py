import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, UpSampling2D, BatchNormalization, ReLU, concatenate

# Crée le modèle VBC en utilisant des modèles préentraînés pour extraire des caractéristiques et ajoute des têtes personnalisées.
def create_vbc_model(input_shape=(224, 224, 3), num_classes=1000, num_classes_seg=21, num_classes_action=10):
    inputs = Input(shape=input_shape)

    # Charger des modèles préentraînés
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)

    # Extraire des caractéristiques
    features_resnet = GlobalAveragePooling2D()(base_model_resnet.output)
    features_efficientnet = GlobalAveragePooling2D()(base_model_efficientnet.output)
    features_mobilenet = GlobalAveragePooling2D()(base_model_mobilenet.output)

    # Concaténer les caractéristiques
    combined_features = concatenate([features_resnet, features_efficientnet, features_mobilenet], axis=-1)

    # Tête de classification
    classification_output = Dense(num_classes, activation='softmax', name='classification')(combined_features)

    # Tête de détection d'objets
    detection_output = Dense(4 + 1 + num_classes, activation='sigmoid', name='detection')(combined_features)

    # Tête de segmentation sémantique
    seg = Conv2D(128, (3, 3), padding='same')(base_model_resnet.output)
    seg = BatchNormalization()(seg)
    seg = ReLU()(seg)
    seg = UpSampling2D(size=(4, 4))(seg)
    segmentation_output = Conv2D(num_classes_seg, (1, 1), activation='softmax', name='segmentation')(seg)

    # Tête de reconnaissance d'actions humaines
    action_output = Dense(num_classes_action, activation='softmax', name='action')(combined_features)

    model = Model(inputs=inputs, outputs=[classification_output, detection_output, segmentation_output, action_output], name='VBC_Model')

    return model

# Initialisation du modèle
vbc_model = create_vbc_model()

# Fonction pour prétraiter une image
def preprocess_image(image, target_size=(224, 224)):
    if image is None:
        raise ValueError("L'image est vide. Vérifiez le chemin du fichier.")
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Fonction pour effectuer l'inférence sur une image
def predict_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de lire l'image à partir du chemin: {image_path}")
    preprocessed_image = preprocess_image(image)
    classification, detection, segmentation, action = vbc_model.predict(preprocessed_image)
    return classification, detection, segmentation, action, image

# Vérification des modèles préentraînés
print("Modèles chargés:")
print(vbc_model.summary())

