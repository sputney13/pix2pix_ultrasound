import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def process_oasbud_for_gan(tfds_set, batch_size = 1):
    
    bmode_extract = []
    for entry in tfds_set['train']:

        image1, image2, mask1, mask2 = extract_crop_normalize(entry)
        bmode_extract.append((tf.cast(mask1, 'float32'), image1))
        bmode_extract.append((tf.cast(mask2, 'float32'), image2))
    
    bmode_slice = tf.data.Dataset.from_tensor_slices(bmode_extract)
    bmode_slice = bmode_slice.map(lambda x: (x[0], x[1]))

    return bmode_slice.batch(batch_size)


def process_oasbud_for_gan_by_class(tfds_set, batch_size = 1):

    benign_extract = []
    malignant_extract = []

    for entry in tfds_set['train']:

        image1, image2, mask1, mask2 = extract_crop_normalize(entry)

        if entry['label'].numpy() == 0:
            benign_extract.append((tf.cast(mask1, 'float32'), image1))
            benign_extract.append((tf.cast(mask2, 'float32'), image2))
        else:
            malignant_extract.append((tf.cast(mask1, 'float32'), image1))
            malignant_extract.append((tf.cast(mask2, 'float32'), image2))

    malignant_slice = tf.data.Dataset.from_tensor_slices(malignant_extract)
    benign_slice = tf.data.Dataset.from_tensor_slices(benign_extract)

    malignant_slice = malignant_slice.map(lambda x: (x[0], x[1]))
    benign_slice = benign_slice.map(lambda x: (x[0], x[1]))

    malignant_ds = malignant_slice.batch(batch_size)
    benign_ds = benign_slice.batch(batch_size)

    return malignant_ds, benign_ds


def process_oasbud_for_classification(tfds_set, test_split = 0.2):
    bmode_extract = []
    bmode_label = []
    
    for entry in tfds_set['train']:
        image1, image2, _, _ = extract_crop_normalize(entry)

        bmode_extract.append(image1)
        bmode_extract.append(image2)
        bmode_label.extend([tf.one_hot(entry['label'].numpy(), 2) for _ in range(2)])

    bmode_train, bmode_test = make_train_test_ds(bmode_extract, bmode_label, test_split)

    return bmode_train.batch(1), bmode_test.batch(1)


def process_oasbud_for_classification_with_aug(tfds_set, benign_gen, malignant_gen, test_split = 0.2):
    bmode_extract = []
    bmode_label = []

    for entry in tfds_set['train']:
        image1, image2, mask1, mask2 = extract_crop_normalize(entry)
        
        mask1 = mask1[tf.newaxis, :, :, :]
        mask2 = mask2[tf.newaxis, :, :, :]

        bmode_extract.append(image1[tf.newaxis, :, :, :])
        bmode_extract.append(image2[tf.newaxis, :, :, :])
        bmode_label.extend([tf.one_hot(entry['label'].numpy(), 2) for i in range(4)])

        if entry['label'].numpy() == 0:
            bmode_extract.append(benign_gen(tf.cast(mask1, 'float32')))
            bmode_extract.append(benign_gen(tf.cast(mask2, 'float32')))
        else:
            bmode_extract.append(malignant_gen(tf.cast(mask1, 'float32')))
            bmode_extract.append(malignant_gen(tf.cast(mask2, 'float32')))
    
    bmode_train, bmode_test = make_train_test_ds(bmode_extract, bmode_label, test_split)

    return bmode_train, bmode_test
    

def process_oasbud_for_segmentation(tfds_set, test_split = 0.2):
    bmode_ims = []
    bmode_masks = []

    for entry in tfds_set['train']:
        image1, image2, mask1, mask2 = extract_crop_normalize(entry)
        bmode_ims.append(image1)
        bmode_ims.append(image2)
        bmode_masks.append(tf.cast(mask1, 'float32'))
        bmode_masks.append(tf.cast(mask2, 'float32'))

    bmode_train, bmode_test = make_train_test_ds(bmode_ims, bmode_masks, test_split)

    return bmode_train.batch(1), bmode_test.batch(1)


def process_oasbud_for_segmentation_with_aug(tfds_set, oasbud_generator, test_split = 0.2):
    bmode_ims = []
    bmode_masks = []

    for entry in tfds_set['train']:
        image1, image2, mask1, mask2 = extract_crop_normalize(entry)
        
        mask1 = mask1[tf.newaxis, :, :, :]
        mask2 = mask2[tf.newaxis, :, :, :]

        bmode_ims.append(image1[tf.newaxis, :, :, :])
        bmode_ims.append(oasbud_generator(tf.cast(mask1, 'float32')))
        bmode_ims.append(image2[tf.newaxis, :, :, :])
        bmode_ims.append(oasbud_generator(tf.cast(mask2, 'float32')))

        bmode_masks.extend([tf.cast(mask1, 'float32') for _ in range(2)])
        bmode_masks.extend([tf.cast(mask2, 'float32') for _ in range(2)])

    bmode_train, bmode_test = make_train_test_ds(bmode_ims, bmode_masks, test_split)

    return bmode_train, bmode_test


def extract_crop_normalize(entry):
    image1 = entry['bmode_1'][:, :, tf.newaxis]
    image2 = entry['bmode_2'][:, :, tf.newaxis]
    mask1 = entry['mask_1'][:, :, tf.newaxis]
    mask2 = entry['mask_2'][:, :, tf.newaxis]

    image1 = tf.image.resize_with_crop_or_pad(image1, 1024, 512)
    image2 = tf.image.resize_with_crop_or_pad(image2, 1024, 512)
    mask1 = tf.image.resize_with_crop_or_pad(mask1, 1024, 512)
    mask2 = tf.image.resize_with_crop_or_pad(mask2, 1024, 512)

    image1 = (image1-np.min(image1))/(np.max(image1)-np.min(image1)) # normalized
    image2 = (image2-np.min(image2))/(np.max(image2)-np.min(image2)) # normalized

    return image1, image2, mask1, mask2


def make_train_test_ds(inputs, labels, test_split):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=test_split)

    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)

    train_ds = tf.data.Dataset.zip((X_train, y_train))
    test_ds = tf.data.Dataset.zip((X_test, y_test))

    return train_ds, test_ds
