import tkinter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from gui import run_gui

print(tf.__version__)
print(tkinter.TkVersion)

mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()


# Adding Channels
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
train_images = train_images / 255.0
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
test_images = test_images / 255.0

# Encoding Labels
train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)
test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)

# Building Model
batch_size = 64
epochs = 5
num_classes = 10

layers = tf.keras.layers
model = tf.keras.Sequential([
    layers.Conv2D(32, (5, 5), padding = 'same', activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (5, 5), padding = 'same', activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, (5, 5), padding = 'same', activation='relu'),
    layers.Conv2D(64, (5, 5), padding = 'same', activation='relu'),
    layers.MaxPool2D(strides=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compiling Model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-8),
    loss='categorical_crossentropy',
    metrics=['acc']
)

# Define Callback
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = Callback()

# Training Model
history = model.fit(train_images, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])

# Evaluate Model
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label='Training Loss')
ax[0].plot(history.history['val_loss'], color='r', label='Validation Loss')
ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label='Training Accuracy')
ax[1].plot(history.history['val_acc'], color='r', label='Validation Accuracy')
ax[1].legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# Save Model
model.save('mnist_cnn_model.keras')

model.summary()

run_gui(model)