import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pylab import *
import scipy as sp
from matplotlib.pyplot import figure
import random
import math

#Funcion que permite hacer señales sinusoidales
def sin_wave(A, f, fs, phi, t):
    '''
    : Parametro A: Amplitud
         : parametro F: Frecuencia de señal
         : Parametro FS: Frecuencia de muestreo
         : Parametro phi: fase
         : parametro t: longitud de tiempo
    '''
    # Si la longitud de la serie de tiempo es T = 1S, 
    # Frecuencia de muestreo FS = 1000 Hz, intervalo de tiempo de muestreo TS = 1 / FS = 0.001S
    # Para los puntos de muestreo de la secuencia de tiempo es n = t / t ts = 1 / 0,001 = 1000, hay 1000 puntos, cada intervalo de punto es TS
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

#Declaracion de las señales que iran en los 3 canales a multiplexar
fs = 4000
y1 = sin_wave(A=0.73, f=440, fs=fs, phi=0, t=0.01)
y2 = sin_wave(A=0.37, f=710, fs=fs, phi=47, t=0.01)
y3 = sin_wave(A=1, f=260, fs=fs, phi=72, t=0.01)
#Declaracion del vector de tiempo
x = np.arange(0, 0.01, 1/fs)
#Grafica de la señal muestreada
plt.xlabel('t/s')
plt.ylabel('y')
plt.grid()
plt.plot(x, y1, 'k')
plt.plot(x, y2, 'r')
plt.plot(x, y3, 'g')
plt.show()

#Proceso de multiplexacion TDM
k = 0
y_ch = np.zeros(120)
for l1 in range(len(x)):
    y_ch[k] = y1[l1]
    k = k + 1
    y_ch[k] = y2[l1]
    k = k + 1
    y_ch[k] = y3[l1]
    k = k + 1
#Grafica de la señal del canal multiplexado
plt.plot(y_ch)
plt.show()
#Definicion de los vectores que llevaran la señal recuperada
y1_rec = np.zeros(40)
y2_rec = np.zeros(40)
y3_rec = np.zeros(40)

#Proceso de recuperacion de la señal
k2 = 1
c1 = 0
c2 = 0
c3 = 0
for l2 in range(len(y_ch)):
    
    if(k2 == 1):
        y1_rec[c1] = y_ch[l2]
        c1 = c1 + 1
        k2 = 2
    elif(k2 == 2):
        y2_rec[c2] = y_ch[l2]
        c2 = c2 + 1
        k2 = 3
    elif(k2 == 3):
        y3_rec[c3] = y_ch[l2]
        c3 = c3 + 1
        k2 = 1
y1_sinm = sin_wave(A=0.73, f=440, fs=16000, phi=0, t=0.01)
y2_sinm = sin_wave(A=0.37, f=710, fs=16000, phi=47, t=0.01)
y3_sinm = sin_wave(A=1, f=260, fs=16000, phi=72, t=0.01)

#Grafica de la señal real(superior) vs graficaa de la señal recuperada (inferior)
x2 = np.arange(0, 0.01, 1/16000)
plt.subplot(211)
plt.xlabel('t/s')
plt.ylabel('y')
plt.grid()
plt.plot(x2, y1_sinm, 'k')
plt.plot(x2, y2_sinm, 'r')
plt.plot(x2, y3_sinm, 'g')

plt.subplot(212)
plt.xlabel('t/s')
plt.ylabel('y')
plt.grid()
plt.plot(x, y1_rec, 'k')
plt.plot(x, y2_rec, 'r')
plt.plot(x, y3_rec, 'g')
plt.show()

