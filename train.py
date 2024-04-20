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

def depthwise_separable_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """
    Bloc de convolution séparable en profondeur typique de l'architecture MobileNet.
    """
    # Convolution en profondeur
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Convolution ponctuelle (1x1) pour combiner les sorties
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """
    Bloc résiduel typique de l'architecture ResNet avec adaptation pour les raccourcis.
    """
    shortcut = x

    # Première couche du bloc résiduel
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)

    # Seconde couche du bloc résiduel
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)

    # Adaptation du raccourci pour correspondre à la forme de sortie du bloc
    shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)

    # Addition du raccourci avec la sortie du bloc
    x = Add()([x, shortcut])
    return x

def create_vbc_model(input_shape=(224, 224, 3), num_classes=1000, width_multiplier=1.0, depth_multiplier=1.0, detection_heads=1):
    """
    Crée le modèle VBC avec des paramètres pour ajuster la largeur, la profondeur, et inclut une tête de détection.
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(int(32 * width_multiplier), (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Intégrer les blocs séparables en profondeur
    filters_list = [int(width_multiplier * f) for f in [64, 128, 256, 512]]
    for filters in filters_list:
        for _ in range(int(depth_multiplier)):
            x = depthwise_separable_block(x, filters=filters)

    # GlobalAveragePooling suivi par la couche dense pour la classification
    pooled_features = GlobalAveragePooling2D()(x)
    classification_output = Dense(num_classes, activation='softmax', name='classification')(pooled_features)

    # Tête de prédiction pour la détection d'objets, simplifiée
    detection_output = Dense(detection_heads * (4 + 1 + num_classes), activation='sigmoid', name='detection')(pooled_features)

    model = Model(inputs=inputs, outputs=[classification_output, detection_output], name='VBC_Model')

    return model

class PatchCreator(Layer):
    def __init__(self, num_patches, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches,
            output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerBlock(Layer):
    def __init__(self, projection_dim, num_heads=4, transformer_units=[128, 64], **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.proj = Dense(projection_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dense = [Dense(units, activation="relu") for units in transformer_units]

    def call(self, inputs):
        x1 = self.norm1(inputs)
        attention_output = self.attention(x1, x1)
        proj_input = self.proj(attention_output)
        x2 = self.norm2(inputs + proj_input)
        return x2

def create_vbc_vit_yolo(input_shape=(300, 300, 3), num_classes=1000, detection_heads=1, num_patches=100, patch_size=16, projection_dim=64):
    inputs = Input(shape=input_shape)

    # Création et Encodage des Patchs
    patches = PatchCreator(num_patches, patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Application du Transformer
    for _ in range(3):  # Utilisez un petit nombre de blocs pour simplifier
        encoded_patches = TransformerBlock(projection_dim)(encoded_patches)

    # Pooling et Tête de Classification
    representation = GlobalAveragePooling2D()(encoded_patches)
    classification_output = Dense(num_classes, activation='softmax', name='classification')(representation)

    # Simplification de la Tête de Détection
    detection_output = Dense(detection_heads * (4 + 1 + num_classes), activation='sigmoid', name='detection')(representation)

    model = Model(inputs=inputs, outputs=[classification_output, detection_output], name='VBC_ViT_YOLO')

    return model

def create_vbc_model_effnet(input_shape=(300, 300, 3), num_classes=1000, width_multiplier=1.2, depth_multiplier=1.2, detection_heads=1):
    """
    Version améliorée du modèle VBC incorporant des idées d'EfficientNet.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(int(32 * width_multiplier), (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Ajustement basé sur EfficientNet
    filters_list = [int(width_multiplier * f) for f in [64, 128, 256, 512, 1024]]  # Ajustement des filtres pour le scaling
    for filters in filters_list:
        for _ in range(int(depth_multiplier)):
            x = depthwise_separable_block(x, filters=filters)

    pooled_features = GlobalAveragePooling2D()(x)
    classification_output = Dense(num_classes, activation='softmax', name='classification')(pooled_features)
    detection_output = Dense(detection_heads * (4 + 1 + num_classes), activation='sigmoid', name='detection')(pooled_features)

    model = Model(inputs=inputs, outputs=[classification_output, detection_output], name='VBC_Model_EffNet')
    return model

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
        if not os.path.exists(path):
            print(f"Le fichier {path} n'existe pas.")
            continue
        image = load_and_preprocess_image(path, img_size)
        dataset.append(image.numpy())
        labels.append([annotation['category_id'] for annotation in annotations])

    return np.array(dataset), np.array(labels)

# Assurez-vous que les chemins sont écrits de manière correcte
images_dir = os.path.join('C:', os.sep, 'Users', 'user', 'Desktop', 'Code', 'VBC', 'images', 'train2017')
annotations_dir = os.path.join('C:', os.sep, 'Users', 'user', 'Desktop', 'Code', 'VBC', 'annotations')

annotations_file_path = os.path.join(annotations_dir, "stuff_val2017.json")
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Vérifiez si les images et les annotations ont été chargées avec succès
if images is None or labels is None:
    print("Les images ou annotations n'ont pas été chargées correctement.")
else:
    # Si les données sont chargées avec succès, divisez-les en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Création des datasets TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    # Création et compilation du modèle
    model = create_vbc_model_effnet()
    model.compile(optimizer='adam',
                  loss={'classification': 'sparse_categorical_crossentropy', 'detection': 'mean_squared_error'},
                  metrics={'classification': 'accuracy'})

    # Entraînement du modèle
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=10)