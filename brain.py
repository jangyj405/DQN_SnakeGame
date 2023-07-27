import tensorflow as tf
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def create_model(input_shape=(16,16,4), n_output = 4)->tf.keras.Model:
    input = tf.keras.layers.Input(input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input)
    relu1 = tf.keras.layers.ReLU()(conv1)
    mxpool2 = tf.keras.layers.MaxPool2D()(relu1)
    flatten = tf.keras.layers.Flatten()(mxpool2)

    dense1 = tf.keras.layers.Dense(128)(flatten)
    output = tf.keras.layers.Dense(n_output)(dense1)
    model = tf.keras.Model(input, output)
    model.summary()
    return model


class Brain(tf.keras.Model):
    def __init__(self, input_shape=(32,32,4), n_output = 4, gamma = 0.9,epsilon = 1, load_model = False):
        super(Brain, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = create_model(input_shape, n_output)
        if load_model:
            self.model = tf.keras.models.load_model('brain')
        self.optimizer = tf.keras.optimizers.RMSprop()
        
        self.target_model = create_model(input_shape, n_output)
        self.target_model.set_weights(self.model.get_weights())
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
    def train_step(self, replay_batch):
        s, s_prime, r, a, d = replay_batch
        with tf.GradientTape() as tape:
            Q_values = self.model(s)
            Q_action = tf.gather(Q_values, tf.reshape(a, shape=(-1,1)), batch_dims=1)
            a_prime = self.target_model(s_prime)
            a_prime = tf.stop_gradient(a_prime)
            Q_prime = tf.reduce_max(a_prime, axis=1)
            Q_target = r + self.gamma* (1-d) * Q_prime 
            loss = (tf.keras.losses.MSE(tf.reshape(Q_target, shape=(-1,1)),Q_action))

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss
    def get_action(self, state, use_epsilon = True):
        if use_epsilon and np.random.uniform(0,1) < self.epsilon:
            return np.random.randint(0,4), None
        else:
            output = self.model(state)
            action = tf.squeeze(tf.argmax(output, axis = 1))
            return int(action), output
