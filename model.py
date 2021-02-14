import tensorflow as tf

def model():
    model = tf.keras.models.load_model('my_model.h5')
    return model