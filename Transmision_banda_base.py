import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
from pylab import *
from matplotlib.pyplot import figure
import random
import math

bits = 10 #numero de bits
Ts = 9 #tiempo de muestreo
t = np.arange(0,bits,1/Ts) #generacion del vextor tiempo
x = []

#Generacion trama de bits de entrada con 3 niveles 1 0 y -1
for i in range(0,bits):
  for j in range(0,int(Ts/2),1):
    x.append(0)
  x.append(random.randint(-1,1))
  for i in range(0,int(Ts/2),1):
    x.append(0)

#Grafica de la señal de entrada
figure(figsize=(8, 4), dpi=80)
plt.plot(t,x,'.-')
for i in range(0,bits,1):
  plt.axvline(x = i+0.4444444, color = 'r',linestyle="dashed",linewidth=1)
plt.show();

#Implementacion del filtro
beta = float(input("Escriba el factor de roll-off para el filtro: "))
while (beta <= 0 or beta > 1):
  beta = float(input("Valor inválido. Escriba el factor de roll-off para el filtro: "))
tf = arange(-45,45)
h = np.sinc(tf/Ts) * np.cos(np.pi*beta*tf/Ts) / (1 - (2*beta*tf/Ts)**2) #funcion coseno realxzado

#Figura de la respuesta del coseno realzado ante un impulso
plt.plot(t, h, '.-'),title("Respuesta al impulso del filtro"),xlabel("Tiempo(s)"),ylabel("Amplitud")
plt.show()

#Se realiza la convolucion entre la entrada y el filtro coseno realzado
x_shaped = np.convolve(x, h) 
tsh = np.arange(0,len(x_shaped)/Ts,1/Ts) #Nuevo vector de tiempo para la convolucion

#Figura señal de entrda vs salida filtrada por el coseno realzado
#figure(figsize=(8, 6), dpi=80)
subplot(2,1,1)
plt.plot(t,x,'.-'),title("Cadena de bits antes del filtro"),ylabel("Amplitud")
for i in range(0,bits,1):
  plt.axvline(x = i+0.4444444, color = 'r',linestyle="dashed",linewidth=1)

#figure(figsize=(8, 6), dpi=80)
subplot(2,1,2)
plt.plot(tsh[0:179], x_shaped[0:179], '.-'),title("Cadena de bits filtrada"),xlabel("Tiempo(s)"),ylabel("Amplitud")

for i in range(45,len(x_shaped)-45,Ts):
  plt.axvline(x = i*(1/Ts)+0.4444444, color = 'r',linestyle="dashed",linewidth=1)

plt.show()