import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from pricer.analytical import BlackScholesCall
from neural_network.pipeline import Pipeline
from neural_network.config import bs_put_pipeline_config, bs_call_pipeline_config
import keras_tuner as kt

def model_builder(hp):
    pipeline_config = bs_call_pipeline_config

    hp_neuron_per_layer = hp.Choice('neuron_per_layer', values=[16, 25, 32, 48])
    hp_layer_number = hp.Int('layer_number', min_value=4, max_value=11, step=2)
    hp_hidden_layer_activation = hp.Choice('hidden_layer_activation', values=['relu', 'tanh', 'leaky_relu'])
    hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0025, 0.005, 0.0075, 0.01])
    hp_greeks_rel_weight = hp.Choice('greeks_rel_weight', values=[0.01, 0.05, 0.1, 0.2, 0.5])

    pipeline_config.model.neuron_per_layer = hp_neuron_per_layer
    pipeline_config.model.layer_number = hp_layer_number
    pipeline_config.model.hidden_layer_activation = hp_hidden_layer_activation
    pipeline_config.model.learning_rate = hp_learning_rate
    pipeline_config.model.greeks_relative_weight = hp_greeks_rel_weight

    pipeline = Pipeline(pipeline_config)
    return pipeline.build_model()

tuner = kt.Hyperband(model_builder,
                        objective='val_loss',
                        max_epochs=20,
                        factor=10,
                        directory='hyperparameter_optimization',
                        project_name='blackscholescall_price')

train_data = pd.read_parquet("data/20250317/BlackScholesCall_train.parquet")
# train_data = train_data.sample(frac=1).reset_index(drop=True)
validation_data = pd.read_parquet("data/20250317/BlackScholesCall_validation.parquet")
# validation_data = validation_data.sample(frac=1).reset_index(drop=True)

input_variables = bs_call_pipeline_config.model.input_variables
train_X = train_data[input_variables]
target_variables = list(bs_call_pipeline_config.model.target_variables.keys())
train_y =train_data[target_variables]
greek_variables = list(bs_call_pipeline_config.model.greeks.keys())
train_greeks = train_data[greek_variables]

validation_X = validation_data[input_variables]
validation_y = validation_data[target_variables]
validation_greeks = validation_data[greek_variables]

train_dataset = dataset = tf.data.Dataset.from_tensor_slices(
    (train_X.to_numpy(), train_y.to_numpy(), train_greeks.to_numpy())
).batch(2**12)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_X.to_numpy(), validation_y.to_numpy(), validation_greeks.to_numpy())
).batch(2**11)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

tuner.search(train_dataset, epochs=20, validation_data=validation_dataset, callbacks=[stop_early])
