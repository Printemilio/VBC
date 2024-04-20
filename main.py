import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Télécharger les modèles pré-entraînés
efficientnet_url = "https://github.com/balavenkatesh3322/CV-pretrained-model/releases/download/1.0/efficientnet.h5"
mobilenet_url = "https://github.com/balavenkatesh3322/CV-pretrained-model/releases/download/1.0/mobilenet.h5"
resnet_url = "https://github.com/balavenkatesh3322/CV-pretrained-model/releases/download/1.0/resnet.h5"

# Charger les modèles pré-entraînés localement
efficientnet_model = tf.keras.models.load_model(efficientnet_url)
mobilenet_model = tf.keras.models.load_model(mobilenet_url)
resnet_model = tf.keras.models.load_model(resnet_url)

# Créer un modèle combiné
input_tensor = Input(shape=(None, None, 3))
efficientnet_output = efficientnet_model(input_tensor)
mobilenet_output = mobilenet_model(input_tensor)
resnet_output = resnet_model(input_tensor)

# Concaténer les sorties des trois modèles
combined_output = tf.concat([efficientnet_output, mobilenet_output, resnet_output], axis=-1)

# Créer le modèle combiné
combined_model = Model(inputs=input_tensor, outputs=combined_output)

# Afficher le résumé du modèle combiné
print(combined_model.summary())
