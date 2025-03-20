import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from keras.src import tree

class PricerNet(tf.keras.models.Model):
    def __init__(self, greeks_in_scope, greeks_rel_weight=0.1, *args, **kwargs):
        super(PricerNet, self).__init__(*args, **kwargs)
        self.greeks_in_scope = greeks_in_scope
        self.greeks_rel_weight = greeks_rel_weight
        self.x_var = None

    @tf.function
    def train_step(self, data):
        if self.greeks_in_scope:
            x, y, greeks = data
        else:
            x, y = data

        if self.x_var is None:
            self.x_var = tf.Variable(x, name='x', trainable=True)
        else:
            self.x_var.assign(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            with tf.GradientTape(persistent=True) as greek_tape:
                y_pred_greek = self(self.x_var, training=True)
            
            price_loss = self.compute_loss(x, y, y_pred, sample_weight=None, training=True)


            greek_loss = 0
            greeks_gradient = {greek: {} for greek in self.greeks_in_scope.keys()}
            jacobian = greek_tape.batch_jacobian(y_pred_greek, self.x_var)
            for i, (g, (j, w)) in enumerate(self.greeks_in_scope.items()):
                greeks_gradient[g][f"pred_{g}"] = jacobian[:, 0, j]
                greeks_gradient[g]["loss"] = self.compute_loss(x, greeks[:, i], greeks_gradient[g][f"pred_{g}"],
                                                               sample_weight=None, training=True
                                                               )
                greek_loss += greeks_gradient[g]["loss"]

            loss = price_loss
            for g in self.greeks_in_scope:
                greeks_gradient[g]["weight"] = (greeks_gradient[g]["loss"] / greek_loss)
                loss += self.greeks_rel_weight * greeks_gradient[g]["weight"] * greeks_gradient[g]["loss"]
        gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        loss_output = {"loss": loss, "price_loss": price_loss}
        loss_output.update({f"{g}_loss": greeks_gradient[g]["loss"] for g in self.greeks_in_scope.keys()})
        loss_output.update({f"{g}_weight": greeks_gradient[g]["weight"] for g in self.greeks_in_scope.keys()})

        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return loss_output

    @tf.function
    def test_step(self, data):
        x, y, _ = data
        sample_weight = None
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss, sample_weight=tf.shape(tree.flatten(x)[0])[0]
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
