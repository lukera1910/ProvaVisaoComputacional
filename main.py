import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def carregar_imagens_personalizadas(diretorios, tamanho=(128, 128)):
    imagens = []
    rotulos = []
    for label, caminho in diretorios.items():
        for arquivo in os.listdir(caminho):
            img_path = os.path.join(caminho, arquivo)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, tamanho)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            imagens.append(img)
            rotulos.append(label)
    return np.array(imagens), np.array(rotulos)

diretorios = {
    0: '/content/drive/MyDrive/Colab Notebooks/imagens/gatos',     # rótulo 0 para gatos
    1: '/content/drive/MyDrive/Colab Notebooks/imagens/cachorros'  # rótulo 1 para cachorros
}

x_custom, y_custom = carregar_imagens_personalizadas(diretorios)

(x_cifar, y_cifar), _ = tf.keras.datasets.cifar10.load_data()
y_cifar = y_cifar.flatten()

indices = np.where((y_cifar == 3) | (y_cifar == 5))[0]
x_cifar = x_cifar[indices][:2000]
x_cifar = np.array([cv2.resize(img, (128, 128)) for img in x_cifar])
y_cifar = y_cifar[indices][:2000]
y_cifar = (y_cifar == 5).astype(int) 

x_cifar = x_cifar / 255.0

X = np.concatenate((x_custom, x_cifar), axis=0)
y = np.concatenate((y_custom, y_cifar), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

y_pred_prob = modelo.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Gato", "Cachorro"]))