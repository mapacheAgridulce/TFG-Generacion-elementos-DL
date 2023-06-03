# Entrenamiento de la LAPGAN

# Imports para trabajar con Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

import time
from torchvision.utils import save_image

from torch.autograd import Variable

# Importamos la clase LAPGAN y la funcion para generar ruido
from lapgan import LAPGAN, genRuido


# Descargamos el dataset de CIFAR10 (si ya esta descargado nos avisara en la linea de comandos)
def cargarCIFAR10(loteSz=256, descargar=True):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    conjTrain = torchvision.datasets.CIFAR10(root='../data', train=True,
                                          download=descargar,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(conjTrain, batch_size=loteSz,
                                              shuffle=True, num_workers=2)
    conjPrueba = torchvision.datasets.CIFAR10(root='../data', train=False,
                                         download=descargar,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(conjPrueba, batch_size=loteSz,
                                             shuffle=True, num_workers=2)

    return trainloader, testloader


# Funcion principal para entrenar la red LAPGAN
# Funcion principal para entrenar la red LAPGAN
def trainLAPGAN(modLAPGAN, numNvls, critDisc, critGen, optDisc, optGen, trainloader, epocas, loteSz, dimRuido, cambiosDisc=1, cambiosGen=1, usarGPU = True, mostrarCada=5, maxCambios=None):
    

    for G in modLAPGAN.modelosGen:
        G.train()
    for D in modLAPGAN.modelosDisc:
        D.train()

    one = torch.Tensor([1])
    mone = one * -1
    
    if torch.cuda.is_available():
        one = one.cuda()
        mone = mone.cuda()
        
    for epc in range(epocas):

        perdDisc = [0.0 for i in range(numNvls)]
        perdGen = [0.0 for i in range(numNvls)]

        for ind, data in enumerate(trainloader, 0):
            
            # Tomamos las entradas desde los datos y hacemos downsampling
            inputReal, lab = data
            imgDwn = inputReal.numpy()
            numMiniLotes, numCanales, _, _ = imgDwn.shape


            for l in range(numNvls):
                
                # Calculamos las entradas dependiendo del nivel en el que estemos
                if l == (numNvls - 1):

                    condicionEnt = None
                    inputReal = Variable(torch.Tensor(imgDwn))
                    if usarGPU:
                        inputReal = inputReal.cuda()
                else:

                    nuevoImgDwn = []
                    imagenesUp = []
                    imgRes = []

                    # Ejecutamos la piramide Laplaciana
                    for i in range(numMiniLotes):

                        imgReducidas = []
                        imgAumentadas = []
                        imgResiduales = []

                        for j in range(numCanales):

                            previous = imgDwn[i, j, :]
                            imgReducidas.append(cv2.pyrDown(previous))
                            imgAumentadas.append(cv2.pyrUp(imgReducidas[-1]))
                            imgResiduales.append(previous - imgAumentadas[-1])

                        nuevoImgDwn.append(imgReducidas)
                        imagenesUp.append(imgAumentadas)
                        imgRes.append(imgResiduales)

                    imgDwn = np.array(nuevoImgDwn)
                    imagenesUp = np.array(imagenesUp)
                    imgRes = np.array(imgRes)

                    condicionEnt = Variable(torch.Tensor(imagenesUp))
                    inputReal = Variable(torch.Tensor(imgRes))
                    if usarGPU:
                        condicionEnt = condicionEnt.cuda()
                        inputReal = inputReal.cuda()

                # Obtenemos las entradas para los discriminadores (tanto generadas como reales)
                if l == 0: 
                    dimRuido = 32*32
                elif l == 1: 
                    dimRuido = 16*16
                else: 
                    dimRuido = 100

                ruido = Variable(genRuido(loteSz, dimRuido))
                if usarGPU:
                    ruido = ruido.cuda()
                entFalsa = modLAPGAN.modelosGen[l](ruido, condicionEnt)
                
                # Actualizamos los parametros de los discriminadores
                optDisc[l].zero_grad()
                etiquetas = torch.zeros(2 * loteSz)
                etiquetas = Variable(etiquetas)
                etiquetas[:loteSz] = 1
                if torch.cuda.is_available():
                    etiquetas = etiquetas.cuda()
                
                salidaReal = modLAPGAN.modelosDisc[l](inputReal, condicionEnt)
                discReal = critDisc[l](salidaReal[:, 0], etiquetas[:loteSz])
                discReal.backward()
                
                salidaFalsa = modLAPGAN.modelosDisc[l](entFalsa.detach(), condicionEnt)
                discFalsa = critDisc[l](salidaFalsa[:, 0], etiquetas[loteSz:])
                discFalsa.backward()
                
                perdidaDisc = discReal + discFalsa
                
                optDisc[l].step()
                
                # Actualizamos los parametros del generador
                optGen[l].zero_grad()
                output = modLAPGAN.modelosDisc[l](entFalsa, condicionEnt)
                perdidaGen = critGen[l](output[:, 0], etiquetas[:loteSz])
                perdidaGen.backward()
                
                optGen[l].step()
                
                
                perdDisc[l] += perdidaDisc.item()
                perdGen[l] += perdidaGen.item()
                if ind % mostrarCada == (mostrarCada - 1):
                    print('[%d, %5d, %d] Pérdida del discriminador: %.3f ; Pérdida del generador: %.3f' %
                          (epc+1, ind+1, l+1,
                           perdDisc[l] / mostrarCada,
                           perdGen[l] / mostrarCada))
                    perdDisc[l] = 0.0
                    perdGen[l] = 0.0

            if maxCambios and ind > maxCambios:
                break

    print('Entrenamiento terminado')


def ejecLAPGAN(numNvls=3, epocas=1, loteSz=256, usarGPU=True, dimRuido=100, cambiosDisc=1, cambiosGen=1, numCanales=3, numMuestreo=32, maxCambios=None):
    
    # Cargamos los datos de CIFAR10
    trainloader, testloader = cargarCIFAR10(loteSz=loteSz)

    # Inicializamos los modelos
    modLAPGAN = LAPGAN(numNvls, usarGPU, numCanales)

    # Asignaremos las funciones de perdida y los optimizadores de cada discriminador y generador
    critDisc = []
    critGen = []
    optDisc = []
    optGen = []

    # Fijamos los learning rates de cada red
    lrsDisc = [0.0001, 0.0001, 0.0001]
    lrsGen = [0.0003, 0.0005, 0.003]

    for l in range(numNvls):
        critDisc.append(nn.BCELoss())
        optimizadorDisc = optim.Adam(modLAPGAN.modelosDisc[l].parameters(),
                             lr=lrsDisc[l], betas=(0.5, 0.999))
        optDisc.append(optimizadorDisc)
  
        critGen.append(nn.BCELoss())
        optimizadorGen = optim.Adam(modLAPGAN.modelosGen[l].parameters(),
                             lr=lrsGen[l], betas=(0.5, 0.999))
        optGen.append(optimizadorGen)

    trainLAPGAN(modLAPGAN=modLAPGAN, numNvls=numNvls, critDisc=critDisc, critGen=critGen, optDisc=optDisc, optGen=optGen, trainloader=trainloader, epocas=epocas, loteSz=loteSz, dimRuido=dimRuido, cambiosDisc=cambiosDisc, cambiosGen=cambiosGen,maxCambios=None)
    
    muestreo = modLAPGAN.generate(numMuestreo)
    tiempoActual = time.strftime('%Y-%m-%d %H%M')
    muestreo = torch.Tensor(muestreo)
    save_image(muestreo, './result/%s epc%d.png'% (tiempoActual, epocas), normalize=True)
    return muestreo.numpy()


if __name__ == '__main__':
    ejecLAPGAN(epocas=10, maxCambios=25)
