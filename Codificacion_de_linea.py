import matplotlib.pyplot as plt
import scipy.fftpack as fourier
import numpy as np
n =10; #numero de bits de la trama a codificar
t=np.arange(0,0.001*n,0.00001); #se define el vector del tiempo, en este se tiene un tiempo de bit de 10us

Unipolar = np.zeros(1000); #se define el vector que sera la codificacion de canal unipolar
Manchester = np.zeros(1000); #se define el vector que sera la codificacion de canal Manchester
AMI_RZ = np.zeros(1000); #se define el vector que sera la codificacion de canal AMI con retorno a cero
AMI_NRZ = np.zeros(1000); #se define el vector que sera la codificacion de canal AMI sin retorno a cero

#trama de datos
x=np.random.randint(2,size=n); #la trama es aleatoria

#Unipolar
t1 = 0;
for i in range(n):
    if x[i] == 0: #dependiendo del valor del bit se configura la señal codificada
        Value=np.zeros(100); #si es cero  el 100% del tiempo de bit es 0
    if x[i] == 1:
        Value=np.ones(100); #si es cero  el 100% del tiempo de bit es 1
    for i in range(100):
        Unipolar[t1+i] = Value[i] #se le asigna el valor al vector de salida
    t1 = t1+100;


#Manchester
t2 = 0;
for l in range(n):
    if x[l] == 0: #dependiendo del valor del bit se configura la señal codificada
        for k in range(50): 
           Manchester[t2+k] = -1; #si es 0 el 50% del tiempo de bit se encontrara en -1
        t2 = t2 + 50;
        for c in range(50):
           Manchester[t2+c] = 1; #el otro 50% del tiempo se encontrara en 1
        t2 = t2 + 50;
    if x[l] == 1:
        for k in range(50):
           Manchester[t2+k] = 1; #si es 0 el 50% del tiempo de bit se encontrara en 1
        t2 = t2 + 50;
        for c in range(50):
           Manchester[t2+c] = -1; #el otro 50% del tiempo se encontrara en -1
        t2 = t2 + 50;

print(x)
#se grafican las señales de condificacion unipolar y manchester
plt.subplot(211)
plt.plot(t, Unipolar)
plt.title('Unipolar NRZ')
plt.grid(True)
plt.xticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

plt.subplot(212)
plt.plot(t, Manchester)
plt.title('Manchester')
plt.grid(True)
plt.xticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

plt.show()

#AMI RZ
t3 = 0;
w = 0;
for u in range(n):
    if x[u] == 1: #dependiendo del valor del bit se configura la señal codificada
        for s in range(50):
           AMI_RZ[t3+s] = -(2*(w%2)-1); # si es 1 se utiliza el modullo a 2 de la cantidad de unos en la señal para ir cambiando entre -1 y 1
        t3 = t3 + 50;
        for c in range(50):
           AMI_RZ[t3+c] = 0; #retorna a 0
        t3 = t3 + 50;
        w = w + 1;
    if x[u] == 0:
        for s in range(50):
           AMI_RZ[t3+s] = 0; #si es 0 se mantendra el 100% del tiempo de bit en 0
        t3 = t3 + 50;
        for c in range(50):
           AMI_RZ[t3+c] = 0;
        t3 = t3 + 50;

#se grafican las señales de condificacion unipolar y AMI RZ
plt.subplot(211)
plt.plot(t, Unipolar)
plt.title('Unipolar NRZ')
plt.grid(True)
plt.xticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

plt.subplot(212)
plt.plot(t, AMI_RZ)
plt.title('AMI RZ')
plt.grid(True)
plt.xticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

plt.show()

#AMI NRZ
t3 = 0;
w = 0;
for u in range(n):
    if x[u] == 1: #dependiendo del valor del bit se configura la señal codificada
        for s in range(100):
           AMI_NRZ[t3+s] = -(2*(w%2)-1); # si es 1 se utiliza el modullo a 2 de la cantidad de unos en la señal para ir cambiando entre -1 y 1
        t3 = t3 + 100;
        w = w + 1;
    if x[u] == 0:
        for s in range(100):
           AMI_NRZ[t3+s] = 0; #si es 0 se mantiene en 0  el 100% del tiempo de nit
        t3 = t3 + 100;

#se grafican las señales de condificacion unipolar y AMI NRZ
plt.subplot(211)
plt.plot(t, Unipolar)
plt.title('Unipolar NRZ')
plt.grid(True)
plt.xticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

plt.subplot(212)
plt.plot(t, AMI_NRZ)
plt.title('AMI NRZ')
plt.grid(True)
plt.xticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

plt.show()