import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def set_optimizers(lr = 0.002, beta_1 = 0.5):
    gen_optimizer = tf.keras.optimizers.Adam(lr, beta_1 = beta_1)
    disc_optimizer = tf.keras.optimizers.Adam(lr, beta_1 = beta_1)
    return gen_optimizer, disc_optimizer


def generator_loss(disc_gen_output, gen_output, target, LAMBDA):
    
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_gen_output), disc_gen_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    return gan_loss + (LAMBDA * l1_loss)


def discriminator_loss(disc_real_output, disc_gen_output):

    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_gen_output), disc_gen_output)

    return (real_loss + gen_loss) / 2


def generate_images(gen_model, input_image, target, color = True):
    prediction = gen_model(input_image, training=True)
    prediction = (prediction-np.min(prediction))/(np.max(prediction)-np.min(prediction)) # normalized
    display_list = [input_image[0], target[0], prediction[0]]
  
    titles = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        if color:
            plt.imshow(tf.squeeze(display_list[i]))
        else:
            plt.imshow(tf.squeeze(display_list[i]), cmap='gray')
        plt.axis('off')
    plt.show()


def show_segmentations(model, input_image, mask):
    titles = ['Input Image', 'Ground Truth', 'Predicted Image']

    pred_mask = model.predict(input_image)
    pred_mask = tf.argmax(pred_mask, axis = -1)[..., tf.newaxis]
    display_list = [input_image[0], mask[0], pred_mask[0]]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        plt.imshow(tf.squeeze(display_list[i]), cmap='gray')
        plt.axis('off')
    plt.show()


def fit(train_ds, generator, discriminator, epochs, test_ds = None, **kwargs):
    
    lr = kwargs.get('lr', 0.002)
    beta_1 = kwargs.get('beta_1', 0.5)
    LAMBDA = kwargs.get('LAMBDA', 100)
    
    gen_optimizer, disc_optimizer = set_optimizers(lr, beta_1)

    for epoch in range(epochs):
        
        print("Epoch: ", epoch + 1)
        if test_ds:
            for example_input, example_target in test_ds.take(1):
                generate_images(generator, example_input, example_target)
        else:
            train_ds = train_ds.shuffle(200, reshuffle_each_iteration=True)
            for example_input, example_target in train_ds.take(1):
                generate_images(generator, example_input, example_target)

        for _, (input_image, target) in train_ds.enumerate():
            
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_image, training=True)

                disc_real_output = discriminator([input_image, target], training=True)
                disc_gen_output = discriminator([input_image, gen_output], training=True)

                gen_loss = generator_loss(disc_gen_output, gen_output, target, LAMBDA)
                disc_loss = discriminator_loss(disc_real_output, disc_gen_output)

            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        
        print("Generator Loss: {}, Discriminator Loss: {}".format(gen_loss, disc_loss))
    
    return generator, discriminator