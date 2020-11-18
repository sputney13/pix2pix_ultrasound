import tensorflow as tf

def encode(inputs, filters, kernels):
    weight_init = tf.random_normal_initializer(0.0, 0.02)
    x = tf.keras.layers.Conv2D(filters, kernels, strides=2, padding='same', kernel_initializer=weight_init)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def decode(inputs, filters, kernels, dropout=False):
    weight_init = tf.random_normal_initializer(0.0, 0.02)
    x = tf.keras.layers.Conv2DTranspose(filters, kernels, strides=2, padding='same', kernel_initializer=weight_init)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def UNet(input_shape, out_channels):
    inputs = tf.keras.Input(shape=input_shape) # all images resized to this shape
    weight_init = tf.random_normal_initializer(0.0, 0.02)
    skip_cons = [] # skip connections, to pair encoding with decoding (one for each encoding layer)
    
    # downsampling
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=weight_init)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x) # no BatchNorm
    skip_cons.append(x)
    
    x = encode(x, 128, 4)
    skip_cons.append(x)
    
    x = encode(x, 256, 4)
    skip_cons.append(x)
    
    for i in range(5):
        x = encode(x, 512, 4)
        skip_cons.append(x)
        
    # upsampling
    reverse_skips = skip_cons[:-1] # do not include last one - only 7 layers in upsampling (before last layer)
    reverse_skips.reverse() # will match the encoded in reverse order as upsampled
    
    for i in range(3):
        x = decode(x, 512, 4, True)
        x = tf.keras.layers.Concatenate()([x, reverse_skips[i]])
        
    x = decode(x, 512, 4)
    x = tf.keras.layers.Concatenate()([x, reverse_skips[3]])
    
    x = decode(x, 256, 4)
    x = tf.keras.layers.Concatenate()([x, reverse_skips[4]])
    
    x = decode(x, 128, 4)
    x = tf.keras.layers.Concatenate()([x, reverse_skips[5]])
    
    x = decode(x, 64, 4)
    x = tf.keras.layers.Concatenate()([x, reverse_skips[6]])
    
    # last layer
    x = tf.keras.layers.Conv2DTranspose(out_channels, 4, strides=2, padding='same', kernel_initializer=weight_init, activation='tanh')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)