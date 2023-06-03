# Entrenamiento de la CGAN con MNIST

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports de TensorFlow y Keras para realizar la implementación
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Imports para generar los gráficos y tablas de la red
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


# La siguiente línea evita que se pare la ejecución por
# duplicidad de la librería libiomp5 (como libiomp5 y libiomp5md)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def buildGenerador(inputs, etiqueta, imgSz):
    
    # Parámetros de la red 
    # imgRsz: tamaño de reescalado de la imagen
    # kernel: tamaño del kernel de las convoluciones
    # filtros: numero de filtros de las convoluciones
    imgRsz = imgSz // 4
    kernel = 5
    filtros = [128, 64, 32, 1]

    # Primeros pasos: concatenacion de etiqueta e imagen, capa densa y reescalado
    x = concatenate([inputs, etiqueta], axis=1)
    x = Dense(imgRsz * imgRsz * filtros[0])(x)
    x = Reshape((imgRsz, imgRsz, filtros[0]))(x)

    # Las dos primeras capas convolucionales tienen salto 2 y las últimas salto 1
    for flt in filtros:
        if flt > filtros[-2]:
            salto = 2
        else:
            salto = 1
        # Aplicamos normalizacion de lotes, activacion ReLU y la convolucion con los parametros
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=flt,
                            kernel_size=kernel,
                            strides=salto,
                            padding='same')(x)

    # La ultima capa de activacion es una sigmoide
    x = Activation('sigmoid')(x)
    
    # Recordemos que la entrada esta condicionada por la etiqueta de digito. Guardamos el modelo
    generador = Model([inputs, etiqueta], x, name='generador')

    return generador


def buildDiscriminador(inputs, etiqueta, imgSz):
    
    # Parametros para definir las convoluciones
    kernel = 5
    filtros = [32, 64, 128, 256]

    x = inputs

    # Primeras capas del discriminador
    y = Dense(imgSz * imgSz)(etiqueta)
    y = Reshape((imgSz, imgSz, 1))(y)
    x = concatenate([x, y])

    # Las tres primeras convoluciones tienen salto 2, la ultima salto 1
    for flt in filtros:
        if flt == filtros[-1]:
            salto = 1
        else:
            salto = 2
        # Activacion y convolucion
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=flt,
                   kernel_size=kernel,
                   strides=salto,
                   padding='same')(x)

    # Capas finales para producir el veredicto sobre la imagen generada
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    # De nuevo, la entrada (inputs) esta condicionada por la etiqueta de digito. Guardamos el modelo
    discriminador = Model([inputs, etiqueta], x, name='discriminador')

    return discriminador


def train(modelos, data, params):

    # Los modelos que conforman la GAN
    generador, discriminador, adversarial = modelos

    # Imagenes de entrenamiento y etiquetas
    x_train, y_train = data

    # Parametros de la red
    loteSz, latentSz, pasosTrain, numEtiqueta, modelName = params
    
    # Pasos de entrenamiento tras los que se guarda una imagen generada
    genImg = 500

    # Vector de ruido inicial para el generador
    vectRuido = np.random.uniform(-1.0, 1.0, size=[16, latentSz])
    
    # La etiqueta que condiciona al ruido
    etqRuido = np.eye(numEtiqueta)[np.arange(0, 16) % numEtiqueta]

    # Numero de elementos de entrenamiento
    trainSz = x_train.shape[0]

    print(modelName,
          "Etiqueta para las imágenes generadas: ",
          np.argmax(etqRuido, axis=1))
    
    # Entrenamos el discriminador para un lote con imagenes reales y falsas
    for i in range(pasosTrain):

        # Imagen real aleatoria de MNIST
        indRandom = np.random.randint(0, trainSz, size=loteSz)
        imgReal = x_train[indRandom]
        etqReal = y_train[indRandom]

        # Imagen falsa generada con ruido y etiqueta aleatoria
        ruido = np.random.uniform(-1.0,
                                  1.0,
                                  size=[loteSz, latentSz])
        etqFalsa = np.eye(numEtiqueta)[np.random.choice(numEtiqueta,
                                                          loteSz)]
        imgFalsa = generador.predict([ruido, etqFalsa])

        # La imagen real y la imagen falsa son el lote de entrenamiento
        x = np.concatenate((imgReal, imgFalsa))
        etiqueta = np.concatenate((etqReal, etqFalsa))

        # Definimos las etiquetas para el veredicto (real = 1, falso = 0)
        y = np.ones([2 * loteSz, 1])
        y[loteSz:, :] = 0.0
        
        # Perdida y precision del discriminador
        perd, prec = discriminador.train_on_batch([x, etiqueta], y)
        log = "%d: [Pérdida del discriminador: %f, precisión: %f]" % (i, perd, prec)

        # Entrenamos la red GAN entera congelando los pesos del discriminador con un lote
        # Intentamos engañar al discriminador etiquetando imagenes falsas con 1
        ruido = np.random.uniform(-1.0,
                                  1.0,
                                  size=[loteSz, latentSz])
        etqFalsa = np.eye(numEtiqueta)[np.random.choice(numEtiqueta,
                                                          loteSz)]
        y = np.ones([loteSz, 1])

        # Ahora no es necesario guardar las imagenes falsas, las pasamos como entrada
        perd, prec = adversarial.train_on_batch([ruido, etqFalsa], y)
        log = "%s [Pérdida de la GAN: %f, precisión: %f]" % (log, perd, prec)
        print(log)

        # Cuando se alcanzan 500 pasos consecutivos, salvamos la imagen
        if (i + 1) % genImg == 0:
            guardarImg(generador,
                        vectRuido=vectRuido,
                        etqRuido=etqRuido,
                        mostrar=False,
                        paso=(i + 1),
                        modelName=modelName)
    
    # Una vez hemos terminado, guardamos el modelo del generador
    generador.save(modelName + ".h5")


def guardarImg(generador,
                vectRuido,
                etqRuido,
                mostrar=False,
                paso=0,
                modelName="gan"):
    
    
    os.makedirs(modelName, exist_ok=True)
    filename = os.path.join(modelName, "%05d.png" % paso)

    images = generador.predict([vectRuido, etqRuido])
    print(modelName , " etiqueta para las imagenes generadas: ", np.argmax(etqRuido, axis=1))
    plt.figure(figsize=(2.2, 2.2))
    
    numImgs = images.shape[0]
    imgSz = images.shape[1]
    numFilas = int(math.sqrt(vectRuido.shape[0]))
    
    for i in range(numImgs):
        plt.subplot(numFilas, numFilas, i + 1)
        image = np.reshape(images[i], [imgSz, imgSz])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    
    if mostrar:
        plt.show()
    else:
        plt.close('all')


def construirYEntrenar():

    # Cargamos MNIST
    (x_train, y_train), (_, _) = mnist.load_data()

    # Reescalamos las imagenes como tensores y normalizamos las entradas
    imgSz = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, imgSz, imgSz, 1])
    x_train = x_train.astype('float32') / 255

    numEtiqueta = np.amax(y_train) + 1
    y_train = to_categorical(y_train)

    # Parametros de la red
    modelName = "CGAN-MNIST"
    latentSz = 100
    loteSz = 64
    pasosTrain = 20000
    lr = 2e-4
    decay = 6e-8
    entradaSz = (imgSz, imgSz, 1)
    etqSz = (numEtiqueta, )

    # Construimos el discriminador
    inputs = Input(shape=entradaSz, name='Entrada del discriminador')
    etiqueta = Input(shape=etqSz, name='Etiqueta')
    discriminador = buildDiscriminador(inputs, etiqueta, imgSz)

    # Definimos el optimizador, la perdida y la metrica principal para el discriminador
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminador.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    
    # Mostramos una tabla para los parametros a entrenar del discriminador
    discriminador.summary()

    # Construimos el generador
    entradaSz = (latentSz, )
    inputs = Input(shape=entradaSz, name='Vector latente')
    generador = buildGenerador(inputs, etiqueta, imgSz)
    generador.summary()

    # Construimos la red GAN completa (congelando el discriminador)
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminador.trainable = False
    outputs = discriminador([generador([inputs, etiqueta]), etiqueta])
    adversarial = Model([inputs, etiqueta],
                        outputs,
                        name=modelName)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # Entrenamos al discriminador y la red GAN
    modelos = (generador, discriminador, adversarial)
    data = (x_train, y_train)
    params = (loteSz, latentSz, pasosTrain, numEtiqueta, modelName)
    train(modelos, data, params)


def testGenerador(generador, etiqueta=None):
    vectRuido = np.random.uniform(-1.0, 1.0, size=[16, 100])
    paso = 0
    if etiqueta is None:
        numEtiqueta = 10
        etqRuido = np.eye(numEtiqueta)[np.random.choice(numEtiqueta, 16)]
    else:
        etqRuido = np.zeros((16, 10))
        etqRuido[:,etiqueta] = 1
        paso = etiqueta

    guardarImg(generador,
                noise_input=vectRuido,
                noise_class=etqRuido,
                mostrar=True,
                step=paso,
                modelName="Test salidas")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Cargamos el generador ya entrenado"
    parser.add_argument("-g", "--generador", help=help_)
    help_ = "Especifique un digito a generar: "
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generador:
        generador = load_model(args.generador)
        etiqueta = None
        if args.digit is not None:
            etiqueta = args.digit
        testGenerador(generador, etiqueta)
    else:
        construirYEntrenar()
