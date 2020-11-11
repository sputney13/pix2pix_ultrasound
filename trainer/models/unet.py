import tensorflow as tf

class UNet(tf.keras.Model):

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.kernel_init = tf.random_normal_initializer(0.0, 0.02)
        self.out_channels = out_channels

    def encode(self, inputs, filters, kernels):
        x = tf.keras.layers.Conv2D(filters, kernels, strides=2, padding='same', kernel_initializer=self.kernel_init)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x

    def decode(self, inputs, filters, kernels, dropout = False):
        x = tf.keras.layers.Conv2DTranspose(filters, kernels, strides=2, padding='same', kernel_initializer=self.kernel_init)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        if dropout:
            x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def call(self, inputs):
        skip_cons = [] # skip connections, to pair encoding with decoding (one for each encoding layer)
    
        # downsampling
        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=self.kernel_init)(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x) # no BatchNorm
        skip_cons.append(x)
    
        x = self.encode(x, 128, 4)
        skip_cons.append(x)
    
        x = self.encode(x, 256, 4)
        skip_cons.append(x)
    
        for i in range(5):
            x = self.encode(x, 512, 4)
            skip_cons.append(x)
        
        # upsampling
        reverse_skips = skip_cons[:-1] # do not include last one - only 7 layers in upsampling (before last layer)
        reverse_skips.reverse() # will match the encoded in reverse order as upsampled
    
        for i in range(3):
            x = self.decode(x, 512, 4, True)
            x = tf.keras.layers.Concatenate()([x, reverse_skips[i]])
        
        x = self.decode(x, 512, 4)
        x = tf.keras.layers.Concatenate()([x, reverse_skips[3]])
    
        x = self.decode(x, 256, 4)
        x = tf.keras.layers.Concatenate()([x, reverse_skips[4]])
    
        x = self.decode(x, 128, 4)
        x = tf.keras.layers.Concatenate()([x, reverse_skips[5]])
    
        x = self.decode(x, 64, 4)
        x = tf.keras.layers.Concatenate()([x, reverse_skips[6]])
    
        # last layer
        return tf.keras.layers.Conv2DTranspose(self.out_channels, 4, strides=2, padding='same', kernel_initializer=self.kernel_init, activation='tanh')(x)
        