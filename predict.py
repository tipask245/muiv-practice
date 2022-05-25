from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

loaded_model = tf.keras.models.load_model('digits.h5')
image = load_img('digit/7.png', color_mode="grayscale")


def predict_d(img):
    img = img.resize((28, 28))
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img/255
    res = loaded_model.predict([img])[0]
    return np.argmax(res), max(res)

print(predict_d(image))