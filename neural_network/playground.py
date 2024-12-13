import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv("data/BlackScholesCall.csv")[:100000]

class MyModel(tf.keras.models.Model):
    def train_step(self, data):
        x, y = data
        analytical_gradients = x[:, self.trainable_variables[0].shape[0]:]
        x = x[:, :self.trainable_variables[0].shape[0]]

        with tf.GradientTape() as tape2:
            # tape2.watch(x)
            with tf.GradientTape(persistent=True) as tape1:
                # tape1.watch(x)
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
            trainable_vars = self.trainable_variables
            gradients = tape1.gradient(loss, trainable_vars)
            partial_derivatives = tape1.batch_jacobian(y_pred, x)
            loss_gradients = self.compiled_loss(analytical_gradients, partial_derivatives[:, 0, :4])
        partial_gradients = tape2.gradient(loss_gradients, trainable_vars)
        del tape1
        partial_gradients[-1] = 0 * gradients[-1]

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.optimizer.apply_gradients(zip([grad for grad in partial_gradients], trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

inputs = tf.keras.Input(shape=(5,))
x = tf.keras.layers.Dense(10, activation='relu',
                        #   kernel_regularizer=tf.keras.regularizers.L2(2)
                          )(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = MyModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

# input_data = df[['underlier_price', 'strike', 'expiry', 'interest_rate', 'volatility']].values
# target_data = df[['price']].values
# additional_vars = df[['delta', 'theta', 'vega', 'rho']].values

# def data_generator(input_data, target_data, additional_vars, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data, additional_vars))
#     dataset = dataset.batch(batch_size)
#     return dataset
# batch_size = 32
# dataset = data_generator(input_data, target_data, additional_vars, batch_size)
model.fit(df[["underlier_price", "expiry", "interest_rate", "volatility", "strike",
              "delta", "theta", "rho", "vega"]], df["price"], epochs=10)
model.predict(df[["underlier_price", "expiry", "interest_rate", "volatility", "strike"]])
print("yes")
