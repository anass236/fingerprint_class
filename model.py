import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")
os.add_dll_directory("C:/tools/cuda/bin")

from utils.conf import ROOT_DIR
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import random
# confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

def extract_label(img_path, train=True):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    subject_id, etc = filename.split('__')
    if train:
        gender, lr, finger, _ = etc.split('_')[:4]
    gender = 0 if gender == 'M' else 1
    return gender


def loading_data(path, train, img_size):
    print("loading data from: ", path)
    data = []
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img_array, (img_size, img_size))
        label = extract_label(os.path.join(path, img), train)
        data.append([label, img_resize])
    return data


def results(model, epoch, batch_size):
    r = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_test, y_test), verbose=1)
    acc = model.evaluate(x_test, y_test)
    print("test set loss: ", acc[0])
    print("test set accuracy: ", acc[1] * 100)
    epoch_range = range(1, epoch + 1)
    plt.plot(epoch_range, r.history['accuracy'])
    plt.plot(epoch_range, r.history['val_accuracy'])
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, r.history['loss'])
    plt.plot(epoch_range, r.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()


src_data = os.path.join(ROOT_DIR, "SOCOFing")
easy_data = loading_data(os.path.join(src_data, os.path.join("Altered", "Altered-Easy")), True, 28)
medium_data = loading_data(os.path.join(src_data, os.path.join("Altered", "Altered-Medium")), True, 28)
hard_data = loading_data(os.path.join(src_data, os.path.join("Altered", "Altered-Hard")), True, 28)
data = np.concatenate([easy_data, medium_data, hard_data], axis=0)


random.shuffle(data)

img, labels = [], []
for label, feature in data:
    labels.append(label)
    img.append(feature)
img = np.array(img).reshape(-1, 28, 28, 1)
labels = tf.keras.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(img, labels, test_size=0.2, random_state=0)

print("x_train dimensions: ", x_train.shape)
print("x_test dimensions: ", x_test.shape)
print("y_train dimensions: ", y_train.shape)
print("y_test dimensions: ", y_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=0)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)


def normalize(data):
    data = data.astype("float32")
    data = data / 255
    return data


print("x_train dimensions: ", x_train.shape)
print("x_val dimensions: ", x_val.shape)
print("y_train dimensions: ", y_train.shape)
print("y_val dimensions: ", y_val.shape)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

tf.keras.backend.clear_session()
weight_decay = 1e-4
with mirrored_strategy.scope():
    model_f = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
               input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
               padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
               padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=
        tf.keras.regularizers.l2(weight_decay), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=
        tf.keras.regularizers.l2(weight_decay), padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=
        tf.keras.regularizers.l2(weight_decay), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model_f.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_f.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    results(model_f, 30, 128)

# test in Real data

real_data = loading_data(os.path.join(src_data, os.path.join("Real")), True, 28)
img, labels = [], []
for label, feature in real_data:
    labels.append(label)
    img.append(feature)
img = np.array(img).reshape(-1, 28, 28, 1)

tf.keras.models.save_model(
    model_f,
    'model_class',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

Y_pred = model_f.predict(img)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(labels, y_pred))
print('Classification Report')
target_names = ['Male', 'Female']
print(classification_report(labels, y_pred, target_names=target_names))


