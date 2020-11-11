import tensorflow as tf
import os

DATA_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz"

def download_data():
    zip_path = tf.keras.utils.get_file('cityscapes.tar.gz', origin=DATA_URL, extract=True)
    path = os.path.join(os.path.dirname(zip_path), 'cityscapes/')
    return path

def load_image(file):
    image = tf.image.decode_jpeg(tf.io.read_file(file)) # load

    width = tf.shape(image)[1]
    width = width // 2
    
    city = image[:, :width, :]
    label = image[:, width:, :]
    
    city = tf.cast(city, tf.float32)
    label = tf.cast(label, tf.float32)
    
    city = tf.image.resize(city, [256, 256])
    label = tf.image.resize(label, [256, 256])

    return label, city

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(stacked_image, size=[2, 256, 256, 3])

  return cropped_image[0], cropped_image[1]

@tf.function()
def random_jitter(input_image, real_image):
  # resize to random crop
  input = tf.image.resize(input_image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real = tf.image.resize(real_image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  input, real = random_crop(input, real)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input = tf.image.flip_left_right(input)
    real = tf.image.flip_left_right(real)

  return input, real

def process_image_train(image_file):
  input, real = load_image(image_file)
  input, real = random_jitter(input, real)

  return (input / 255.0), (real / 255.0)

def process_image_test(image_file):
  input, real = load_image(image_file)
  input = tf.image.resize(input, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real = tf.image.resize(real, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return (input / 255.0), (real / 255.0)

def create_dataset(batch_size = 1):
    path = download_data()

    train_ds = tf.data.Dataset.list_files(path + "train/*.jpg")
    test_ds = tf.data.Dataset.list_files(path + "val/*.jpg")
    
    train_ds = train_ds.map(process_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(process_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds
