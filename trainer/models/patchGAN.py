import tensorflow as tf

def PatchGAN(input_shape):
    weight_init = tf.random_normal_initializer(0.0, 0.02)
    
    fake_image = tf.keras.layers.Input(shape=input_shape)
    target_image = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.concatenate([fake_image, target_image])
    
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=weight_init)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=weight_init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=weight_init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=weight_init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same', kernel_initializer=weight_init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=weight_init, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=[fake_image, target_image], outputs=x)