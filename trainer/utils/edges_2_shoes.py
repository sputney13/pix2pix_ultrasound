import tensorflow as tf
import os

DATA_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz"

def download_data():
    zip_path = tf.keras.utils.get_file('edges2shoes.tar.gz', origin=DATA_URL, extract=True)
    path = os.path.join(os.path.dirname(zip_path), 'edges2shoes/')
    return path

def process_image(file):
    image = tf.image.decode_jpeg(tf.io.read_file(file)) # load

    width = tf.shape(image)[1]
    width = width // 2
    
    edge = image[:, :width, :] # left side = edges
    shoe = image[:, width:, :] # right side = shoe
    
    edge = tf.cast(edge, tf.float32)
    shoe = tf.cast(shoe, tf.float32)
    
    edge = tf.image.resize(edge, [256, 256])
    shoe = tf.image.resize(shoe, [256, 256])
    
    edge = (edge / 255.0)
    shoe = (shoe / 255.0)

    return edge, shoe

def create_dataset(batch_size = 4):
    path = download_data()

    train_ds = tf.data.Dataset.list_files(path + "train/*.jpg")
    test_ds = tf.data.Dataset.list_files(path + "val/*.jpg")
    train_ds = train_ds.map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds
