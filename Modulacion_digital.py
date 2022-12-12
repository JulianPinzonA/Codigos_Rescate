import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pylab import *
import scipy as sp
from matplotlib.pyplot import figure
import random
import math

#Declaracion de vectores a utilizaar
bits = []
vect = []
niv = [0]

#Generacion de bits aleatorios
for i in range(0,96):
  bits.append(random.randint(0,1))
  vect.append(bits[i])
  vect.append(bits[i])
print(bits)

#Creacion de los niveles logicos
for i in range(0,96,4):
  niv.append((bits[i]*(2**3))+(bits[i+1]*(2**2))+(bits[i+2]*2)+bits[i+3])
  niv.append((bits[i]*(2**3))+(bits[i+1]*(2**2))+(bits[i+2]*2)+bits[i+3])
print(niv)

Xm = []
Fs = 1000 #Frecuencia de muestreo
ts=np.arange(0,24,1/Fs) #Tiempo de muestra
t = np.arange(0,24.5,0.5) #vector de tiempo
mod = [-3,3,-3,1,-3,-1,-3,-3,-1,3,-1,1,-1,-1,-1,-3,1,3,1,1,1,-1,1,-3,3,3,3,1,3,-1,3,-3] #vector con los valores para modulacion

#Ubicacion de los puntos en el diagrama de constelacion
for i in range(1,49,2):
  phase = math.atan2(mod[(2*niv[i])+1],mod[(2*niv[i])])
  if phase < 0:
    phase = phase + (2*pi)
  tphase = int(phase*(500/(2*pi)))
  for j in range(0,1000):
    Xm.append(sqrt((mod[2*niv[i]]**2)+(mod[(2*niv[i])+1]**2))*np.cos((2*np.pi*2*ts[j+tphase])))

#Figura de los niveles logicos, modulacion en el tieempo y diagrama de constelacion
#figure(figsize=(6, 8), dpi=80)
subplot(3,1,1)
plt.step(t,niv, color = 'g'), title("Niveles de bits")
for i in range(0,25):
  plt.axvline(x = i, color = 'r',linestyle="dashed",linewidth=0.5)
for i in range(0,16):
  plt.axhline(y = i, color = 'r',linestyle="dashed",linewidth=0.5)

#figure(figsize=(6, 8), dpi=80)
subplot(3,1,2)
plt.plot(ts,Xm, color = 'g'), title("Señal modulada")
for i in range(0,25):
  plt.axvline(x = i, color = 'r',linestyle="dashed",linewidth=1)

#figure(figsize=(6, 8), dpi=80)
subplot(3,1,3)
for i in range(1,49,2):
  plt.plot(mod[(2*niv[i])] + random.uniform(-0.4,0.4), mod[(2*niv[i])+1] + random.uniform(-0.4,0.4), 'o', color='g'),title('Mapa de constelación')
for i in range(-4,5,2):
  plt.axvline(x = i, color = 'r',linestyle="dashed",linewidth=1)
  plt.axhline(y = i, color = 'r',linestyle="dashed",linewidth=1)
plt.axvline(x = 0, color = 'black',linewidth=1.5)
plt.axhline(y = 0, color = 'black',linewidth=1.5)
plt.show()