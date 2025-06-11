# models/amc_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from models.base_model import BaseTFModel

class NN(BaseTFModel):

    def build_model(self):
        # 1) Parámetros (igual que antes)
        mp = self.model_params
        seq_len = int(mp.get('seq_len', 4096))
        n_classes = int(mp.get('output_size', mp.get('n_classes', 5)))

        # 2) Entrada (seq_len, 2) pero no la usamos realmente
        inp = layers.Input(shape=(seq_len, 2), name='IQ_input')

        # 3) Lambda que ignora la entrada y genera un tensor de ceros [batch_size, n_classes]
        x = layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], n_classes))
        )(inp)

        # 4) Softmax sobre los ceros → distribución uniforme
        outputs = layers.Activation('softmax')(x)

        # 5) Construcción del modelo
        model = models.Model(inputs=inp, outputs=outputs)
        return model