import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf

class PricerNet(tf.keras.models.Model):
    def __init__(self, greeks_in_scope, *args, **kwargs):
        super(PricerNet, self).__init__(*args, **kwargs)
        self.greeks_in_scope = greeks_in_scope
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
            
            price_loss = self.compiled_loss(y, y_pred)
            
            loss = price_loss
            greeks_gradient = {greek: {} for greek in self.greeks_in_scope.keys()}
            for i, (g, j) in enumerate(self.greeks_in_scope.items()):
                greeks_gradient[g][f"pred_{g}"] = greek_tape.batch_jacobian(y_pred_greek, self.x_var)[:, 0, j]
                greeks_gradient[g]["loss"] = self.compiled_loss(greeks[:, i], greeks_gradient[g][f"pred_{g}"])
                loss += greeks_gradient[g]["loss"]
        gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        loss_output = {"price_loss": price_loss}
        loss_output.update({f"{g}_loss": greeks_gradient[g]["loss"] for g in self.greeks_in_scope.keys()})
        return loss_output
    
    def get_config(self):
        config = super(PricerNet, self).get_config()
        print(config)
        config.update({
            "greeks_in_scope": self.greeks_in_scope,
        })
        return config
