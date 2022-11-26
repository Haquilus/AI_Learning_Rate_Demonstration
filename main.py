import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


def main():
    label_as_binary = LabelBinarizer()

    # Making the dataset
    (x_train, y_train_input), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = label_as_binary.fit_transform(y_train_input)
    print("Original Data: " + str(x_train.shape) + ", " + str(y_train.shape))

    # Making the second dataset

    # CNN:
    # The models are individualized so that they cant use what they learned when being retrained with fewer samples
    model1 = tf.keras.models.Sequential([
        # layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),

        # layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # connecting layer
        Flatten(),
        Dense(100, activation='relu'),

        # output layer
        Dense(10, activation="softmax")
    ])
    model2 = tf.keras.models.Sequential([
        # layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),

        # layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

         # connecting layer
        Flatten(),
        Dense(100, activation='relu'),

        # output layer
        Dense(10, activation="softmax")
    ])
    model3 = tf.keras.models.Sequential([
        # layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),

        # layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # connecting layer
        Flatten(),
        Dense(100, activation='relu'),

        # output layer
        Dense(10, activation="softmax")
    ])
    model4 = tf.keras.models.Sequential([
        # layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),

        # layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # connecting layer
        Flatten(),
        Dense(100, activation='relu'),

        # output layer
        Dense(10, activation="softmax")
    ])

    # Train the models
    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m1 = model1.fit(x_train, y_train, batch_size=2048, epochs=10)
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m2 = model2.fit(x_train[0:30000], y_train[0:30000], batch_size=2048, epochs=10)
    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m3 = model3.fit(x_train[0:15000], y_train[0:15000], batch_size=2048, epochs=10)
    model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m4 = model4.fit(x_train[0:7500], y_train[0:7500], batch_size=2048, epochs=10)

    # Show what's going on in a pretty plot
    plt.plot(m1.history['accuracy'], label='60k samples')
    plt.plot(m2.history['accuracy'], label='30k samples')
    plt.plot(m3.history['accuracy'], label='15k samples')
    plt.plot(m4.history['accuracy'], label='7.5k samples')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
