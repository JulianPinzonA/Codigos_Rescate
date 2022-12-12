import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pylab import *
import scipy as sp
from matplotlib.pyplot import figure
import random
import math

K = 32 # numero de portadoras en el bloque de OFDM
CP = K//4  # longitud del ciclo de bits de cabecera: 25% del bloque

P = 4 # Numero de portadoras piloto en el bloque
pilotValue = 3+3j # El valor de las portadoras piloto

allCarriers = np.arange(K)  # Indica la posicion de todas las portadoras

pilotCarriers = allCarriers[::K//P] #Indica la posiicion de las portadoras piloto

# Se asigna la ultima portadora tambien como piloto
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# Las portadoras de datos son las restantes (las que  no son piloto)
dataCarriers = np.delete(allCarriers, pilotCarriers)

# Se muestra las portadoras piloto como puntos azules y las portadoras de datos como puntos rojos
print ("allCarriers:   %s" % allCarriers)
print ("pilotCarriers: %s" % pilotCarriers)
print ("dataCarriers:  %s" % dataCarriers)
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
plt.show()

mu = 4 # bits per symbol (16PSK)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_table = {
    (0,0,0,0) :  1-0j,
    (0,0,0,1) :  0.923-0.382j,
    (0,0,1,0) :  0.382-0.923j,
    (0,0,1,1) :  0.707-0.707j,
    (0,1,0,0) : -0.923-0.382j,
    (0,1,0,1) : -0.707-0.707j,
    (0,1,1,0) :  0-1j,
    (0,1,1,1) : -0.382-0.923j,
    (1,0,0,0) :  0.923+0.382j,
    (1,0,0,1) :  0.707+0.707j,
    (1,0,1,0) :  0+1j,
    (1,0,1,1) :  0.382+0.923j,
    (1,1,0,0) : -1+0j,
    (1,1,0,1) :  -0.923+0.382j,
    (1,1,1,0) :  -0.382+0.923j,
    (1,1,1,1) :  -0.707+0.707j
}
for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
plt.show()
demapping_table = {v : k for k, v in mapping_table.items()}

channelResponse = np.array([1, 0, 0.3+0.3j])  # la respuesta al impulso del canal
H_exact = np.fft.fft(channelResponse, K)
plt.plot(allCarriers, abs(H_exact))
plt.show()
SNRdb = 25  # SNR (en dB)

bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
print ("Cantidad de bits: ", len(bits))
print ("Bits: ", bits)
print ("Mean of bits (valor cercano a 0.5): ", np.mean(bits))

def SP(bits):
    return bits.reshape((len(dataCarriers), mu))
bits_SP = SP(bits)
print ("Primeros 5 grupos de bits")
print (bits_SP[:5,:])

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])
PSK= Mapping(bits_SP)
print ("Primeras 5 modulaciones PSK:")
print (bits_SP[:5,:])
print (PSK[:5])

def OFDM_symbol(QAM_payload): #Posicionamiento de simbolos
    symbol = np.zeros(K, dtype=complex) 
    symbol[pilotCarriers] = pilotValue  
    symbol[dataCarriers] = QAM_payload  
    return symbol


OFDM_data = OFDM_symbol(PSK)
print ("Numero de portadoras OFDM carriers en el dominio de la frecuencia: ", len(OFDM_data))

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
OFDM_time = IDFT(OFDM_data)
print ("Numerro de muestras OFDM samples en el dominio del tiempo antes de CP: ", len(OFDM_time))

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               
    return np.hstack([cp, OFDM_time]) 
OFDM_withCP = addCP(OFDM_time)
print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))

def channel(signal):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # Calculo de la potencia de ruido y SNR
    
    print ("Potencia señal: %.4f. Potencia ruido: %.4f" % (signal_power, sigma2))
    
    
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX)
plt.figure(figsize=(8,2))
plt.plot(abs(OFDM_TX), label='señal TX')
plt.plot(abs(OFDM_RX), label='señal RX')
plt.legend(fontsize=10)
plt.xlabel('Tiempo'); plt.ylabel('$|x(t)|$');
plt.grid(True);
plt.show()