import tensorflow as tf

class PatchGAN(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_init = tf.random_normal_initializer(0.0, 0.02)

    def call(self, inputs):

        fake_image = inputs[0]
        target_image = inputs[1]
    
        x = tf.keras.layers.concatenate([fake_image, target_image])
    
        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=self.kernel_init)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=self.kernel_init)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=self.kernel_init)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
        x = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=self.kernel_init)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
        x = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same', kernel_initializer=self.kernel_init)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
        return tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=self.kernel_init, activation='sigmoid')(x)