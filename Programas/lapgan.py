# Modelo de la LAPGAN

# Imports para trabajar con Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from torch.autograd import Variable


# Hereda de la clase Module
class DiscCero(nn.Module):

    # Inicializamos las capas de la red
    def __init__(self):

        super(DiscCero, self).__init__()
        
        # Dos capas convolucionales
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2)

        # Una capa de reasignado (para dar forma de matriz a las entradas)
        self.sz = ((32-5+1)-5)//2+1

        # Una capa lineal
        self.fc1 = nn.Linear(128*self.sz*self.sz, 1)

        # Dos normalizaciones de lotes
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)


    def forward(self, x, condicion=None):
        
        # Condicionamos la entrada con la imagen residual
        x = x + condicion

        # Definimos los pasos: convolución -> activacion -> normalizacion
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))

        # Ejecutamos el reasignado
        x = x.view(-1, 128*self.sz*self.sz)

        # Utilizamos la capa lineal como ultima capa y activamos con una sigmoide
        x = torch.sigmoid(self.fc1(x))

        # Devolvemos el resultado de la red
        return x


class GenCero(nn.Module):

    def __init__(self):

        super(GenCero, self).__init__()

        # Definimos las tres capas convolucionales traspuestas
        self.conv1 = nn.ConvTranspose2d(3+1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, padding=1)

        # Definimos las dos normalizaciones de lotes
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)


    def forward(self, x, condicion=None):

        # Reasignamos la entrada a un tensor para trabajar con el en la red
        x = x.view(-1, 1, 32, 32)

        # Concatenamos las entradas formateadas con las imagenes residuales
        x = torch.cat((condicion, x), 1)

        # Ejecutamos los pasos de la red
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.conv3(x)

        # Devolvemos el resultado obtenido
        return x


class DiscUno(nn.Module):

    def __init__(self):

        super(DiscUno, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)

        self.sz = ((16-5+1)-5)//2+1

        self.fc1 = nn.Linear(64*self.sz*self.sz, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        

    def forward(self, x, condicion=None):

        x = x + condicion

        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))

        x = x.view(-1, 64*self.sz*self.sz)

        x = torch.sigmoid(self.fc1(x))

        return x


class GenUno(nn.Module):

    def __init__(self):

        super(GenUno, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(3+1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x, condicion=None):
        
        x = x.view(-1, 1, 16, 16)

        x = torch.cat((condicion, x), 1)
        
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.conv3(x)

        return x


class DiscDos(nn.Module):

    def __init__(self):

        super(DiscDos, self).__init__()

        # Para el discriminador 2 (que va el primero), utilizamos tres capas lineales
        self.fc1 = nn.Linear(3*8*8, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)

    def forward(self, x, condicion=None):

        # Reasignamos la imagen
        x = x.view(-1, 3*8*8)

        # Pasos: lineal -> activacion
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        # Devolvemos el resultado
        return x


class GenDos(nn.Module):

    def __init__(self):

        super(GenDos, self).__init__()

        # De nuevo, solo usamos tres capas lineales para el generador (que intentara deshacer el discriminador 2)
        self.fc1 = nn.Linear(100, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 8*8*3)

    def forward(self, x, condicion=None):

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        # Formateamos la salida a una matriz de 3 canales de color y 8x8
        x = x.view(-1, 3, 8, 8)

        return x


class LAPGAN(object):

    def __init__(self, numNvls, usarGPU=False, numCanales=3):

        self.numNvls = numNvls
        self.numCanales = numCanales
        self.usarGPU = usarGPU
        self.modelosDisc = []
        self.modelosGen = []

        # Las redes generativas 0 y 1 estan condicionadas
        mDisc0 = DiscCero()
        if usarGPU: mDisc0 = mDisc0.cuda()
        self.modelosDisc.append(mDisc0)

        mGen0 = GenCero()
        if usarGPU: mGen0 = mGen0.cuda()
        self.modelosGen.append(mGen0)

        mDisc1 = DiscUno()
        if usarGPU: mDisc1 = mDisc1.cuda()
        self.modelosDisc.append(mDisc1)

        mGen1 = GenUno()
        if usarGPU: mGen1 = mGen1.cuda()
        self.modelosGen.append(mGen1)
        
        # La red 2 no esta condicionada
        mDisc2 = DiscDos()
        if usarGPU: mDisc2 = mDisc2.cuda()
        self.modelosDisc.append(mDisc2)

        mGen2 = GenDos()
        if usarGPU: mGen2 = mGen2.cuda()
        self.modelosGen.append(mGen2)

        print(self.modelosGen)
        print(self.modelosDisc)

    def generar(self, loteSz, nivel=None, generador=False):

        for G in self.modelosGen:
            G.eval()
            
        self.outputs = []
        self.salidaGen = []

        for Nvl in range(self.numNvls):
            modeloGen = self.modelosGen[self.numNvls - Nvl - 1]

            # Generacion de ruido
            if Nvl == 0: 
                self.dimRuido = 100
            elif Nvl == 1: 
                self.dimRuido = 16*16
            else: 
                self.dimRuido = 32*32
            
            ruido = Variable(genRuido(loteSz, self.dimRuido))
            if self.usarGPU:
                ruido = ruido.cuda()

            x = []

            if Nvl == 0:
                # Generamos las imagenes directamente (en el nivel 0)
                imgGen = modeloGen.forward(ruido)
                if self.usarGPU:
                    imgGen = imgGen.cpu()
                imgGen = imgGen.data.numpy()
                x.append(imgGen)
                self.salidaGen.append(imgGen)
            else:
                # Proceso de reescalado con la piramide
                imgEnt = np.array([[cv2.pyrUp(imgGen[i, j, :])
                                      for j in range(self.numCanales)]
                                      for i in range(loteSz)])
                imgCond = Variable(torch.Tensor(imgEnt))
                if self.usarGPU:
                    imgCond = imgCond.cuda()

                # Generamos las imgenes con informacion extra (las residuales)
                imgRes = modeloGen.forward(ruido, imgCond)
                if self.usarGPU:
                    imgRes = imgRes.cpu()
                imgGen = imgRes.data.numpy() + imgEnt
                self.salidaGen.append(imgRes.data.numpy())
                x.append(imgGen)

            self.outputs.append(x[-1])

        if nivel is None:
            nivel = -1

        # Generaremos una tabla de imagenes donde se pueda ver el reescalado (upscaling)
        x = self.outputs[0]
        t = np.zeros(loteSz * self.numCanales * 32 * 32).reshape(loteSz, self.numCanales, 32, 32)
        t[:, :, :x.shape[2], :x.shape[3]] = x
        imgFinales = t

        x = self.outputs[1]
        t = np.zeros(loteSz * self.numCanales * 32 * 32).reshape(loteSz, self.numCanales, 32, 32)
        t[:, :, :x.shape[2], :x.shape[3]] = x
        imgFinales = np.concatenate([imgFinales, t], axis=0)

        x = self.outputs[2]
        t = np.zeros(loteSz * self.numCanales * 32 * 32).reshape(loteSz, self.numCanales, 32, 32)
        t[:, :, :x.shape[2], :x.shape[3]] = x
        imgFinales = np.concatenate([imgFinales, t], axis=0)

        return imgFinales


def genRuido(numInstancias, numDim):
    return torch.Tensor(np.random.normal(loc=0, scale=0.1, size=(numInstancias, numDim)))
