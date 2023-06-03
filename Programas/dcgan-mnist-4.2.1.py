# Entrenamiento de la DCGAN con MNIST

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
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
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


def buildGenerador(inputs, imgSz):

    # Parámetros de la red 
    # imgRsz: tamaño de reescalado de la imagen
    # kernel: tamaño del kernel de las convoluciones
    # filtros: numero de filtros de las convoluciones
    imgRsz = imgSz // 4
    kernel = 5
    filtros = [128, 64, 32, 1]

    # Primeros pasos: capa densa y reescalado
    x = Dense(imgRsz * imgRsz * filtros[0])(inputs)
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
    generador = Model(inputs, x, name='generador')
    return generador


def buildDiscriminador(inputs):
    
    # Parametros para definir las convoluciones
    kernel = 5
    filtros = [32, 64, 128, 256]

    x = inputs
    
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
    
    discriminador = Model(inputs, x, name='discriminador')
    return discriminador


def train(modelos, x_train, params):
    
    # Los modelos que conforman la GAN
    generador, discriminador, adversarial = modelos
    
    # Parametros de la red
    loteSz, latentSz, pasosTrain, modelName = params
    
    # Pasos de entrenamiento tras los que se guarda una imagen generada
    genImg = 500
    
    # Vector de ruido inicial para el generador
    vectRuido = np.random.uniform(-1.0, 1.0, size=[16, latentSz])
    
    # Numero de elementos de entrenamiento
    trainSz = x_train.shape[0]
    
    # Entrenamos el discriminador para un lote con imagenes reales y falsas
    for i in range(pasosTrain):

        # Imagen real aleatoria de MNIST
        indRandom = np.random.randint(0, trainSz, size=loteSz)
        imgReal = x_train[indRandom]
        
        # Imagen falsa generada con ruido y etiqueta aleatoria
        ruido = np.random.uniform(-1.0,
                                  1.0,
                                  size=[loteSz, latentSz])
        imgFalsa = generador.predict(ruido)
        
        # La imagen real y la imagen falsa son el lote de entrenamiento
        x = np.concatenate((imgReal, imgFalsa))
        
        # Definimos las etiquetas para el veredicto (real = 1, falso = 0)
        y = np.ones([2 * loteSz, 1])
        y[loteSz:, :] = 0.0
        
        # Perdida y precision del discriminador
        perd, prec = discriminador.train_on_batch(x, y)
        log = "%d: [Pérdida del discriminador: %f, precisión: %f]" % (i, perd, prec)

        # Entrenamos la red GAN entera congelando los pesos del discriminador con un lote
        # Intentamos engañar al discriminador etiquetando imagenes falsas con 1
        ruido = np.random.uniform(-1.0,
                                  1.0,
                                  size=[loteSz, latentSz])
        y = np.ones([loteSz, 1])
        
        # Ahora no es necesario guardar las imagenes falsas, las pasamos como entrada
        perd, prec = adversarial.train_on_batch(ruido, y)
        log = "%s [Pérdida de la GAN: %f, precisión: %f]" % (log, perd, prec)
        print(log)

        # Cuando se alcanzan 500 pasos consecutivos, salvamos la imagen
        if (i + 1) % genImg == 0:
            guardarImg(generador,
                        vectRuido=vectRuido,
                        mostrar=False,
                        paso=(i + 1),
                        modelName=modelName)
   
    # Una vez hemos terminado, guardamos el modelo del generador
    generador.save(modelName + ".h5")


def guardarImg(generador,
                vectRuido,
                mostrar=False,
                paso=0,
                modelName="gan"):
    
    os.makedirs(modelName, exist_ok=True)
    filename = os.path.join(modelName, "%05d.png" % paso)

    images = generador.predict(vectRuido)
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
    (x_train, _), (_, _) = mnist.load_data()

    # Reescalamos las imagenes como tensores y normalizamos las entradas
    imgSz = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, imgSz, imgSz, 1])
    x_train = x_train.astype('float32') / 255

    # Parametros de la red
    modelName = "DCGAN-MNIST"
    latentSz = 100
    loteSz = 64
    pasosTrain = 2000
    lr = 2e-4
    decay = 6e-8
    entradaSz = (imgSz, imgSz, 1)

    # Construimos el discriminador
    inputs = Input(shape=entradaSz, name='Entrada del discriminador')
    discriminador = buildDiscriminador(inputs)
    
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
    generador = buildGenerador(inputs, imgSz)
    generador.summary()

    # Construimos la red GAN completa (congelando el discriminador)
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminador.trainable = False
    adversarial = Model(inputs, 
                        discriminador(generador(inputs)),
                        name=modelName)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # train discriminator and adversarial networks
    modelos = (generador, discriminador, adversarial)
    params = (loteSz, latentSz, pasosTrain, modelName)
    train(modelos, x_train, params)


def testGenerador(generador):
    vectRuido = np.random.uniform(-1.0, 1.0, size=[16, 100])
    guardarImg(generador,
                noise_input=vectRuido,
                mostrar=True,
                modelName="Test salidas")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Cargamos el generador ya entrenado"
    parser.add_argument("-g", "--generador", help=help_)
    args = parser.parse_args()
    if args.generador:
        generador = load_model(args.generador)
        testGenerador(generador)
    else:
        construirYEntrenar()
