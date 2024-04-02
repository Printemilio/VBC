import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, DepthwiseConv2D, GlobalAveragePooling2D, Dense

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

# Créer le modèle VBC
vbc_model = create_vbc_model()
print(vbc_model.summary())
