import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fourier

m = 12; # de bits de la trama a codificar
t=np.arange(0,0.01,0.00001); #vector de tiempo

hamming = np.zeros(20) 
#trama de datos
#x=np.random.randint(2,size=m);
x = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
#calcular numero de bits del codigo hamming
n = 1;
while(2**n < m+n+1):
    n=n+1;

#Pocisionar bits en el nuevo mensaje
N = 0;
M = 0;
for w in range (m+n):
    if((2**N)-1 == w):
        hamming[w] = 0;
        N = N + 1;
    else:
        op = m - M - 1;
        hamming[w] = x[op];
        M = M + 1;

#calcular los bits del codigo hamming (modulo a 2)
a = np.zeros(10);
b = 0;
for l in range(m+n):
    if hamming[l] == 1:
        r  = l;
        b = 0;
        while(r>=1):
            if int(r)%2 == 1:
                a[b] = a[b] + 1;
                r = r/2
                b = b + 1;
            else:
                r= r/2;
                b = b + 1;
z = np.zeros(10);
for k in range(n):
    z[k] = a[k] % 2;



#Calcular el modulo a 2 del hamming
mod2 = np.zeros(10);
b2 = 0;
for c2 in range (len(hamming)):
    if hamming[c2] == 1:
        b2 = 0;
        r2 = c2+1;
        while(r2>=1):
            if(int(r2)%2 == 1):
                mod2[b2] = mod2[b2] + 1;
                r2 = r2/2;
                b2 = b2 + 1;
            else:
                r2 = r2/2;
                b2 = b2 + 1;
palabra = np.zeros(20);
for c3 in range (len(mod2)):
    palabra[c3] = mod2[c3] % 2;

#Pocisionar bits en el nuevo mensaje
N = 0;
M = 0;
hamming2 = np.zeros(20);
for w in range (m+n):
    if((2**N)-1 == w):
        hamming2[w] = palabra[N];
        N = N + 1;
    else:
        op = m - M - 1;
        hamming2[w] = x[op];
        M = M + 1;

#calcular los bits del codigo hamming (modulo a 2)
a2 = np.zeros(10);
b2 = 0;
for l in range(m+n):
    if hamming2[l] == 1:
        r2  = l+1;
        b2 = 0;
        while(r2>=1):
            if int(r2)%2 == 1:
                a2[b2] = a2[b2] + 1;
                r2 = r2/2
                b2 = b2 + 1;
            else:
                r2= r2/2;
                b2 = b2 + 1;
z2 = np.zeros(n);
for k in range(n):
    z2[k] = a2[k] % 2;

#bit con error
error_bit = np.random.randint(m);

#Pocisionar bits en el nuevo mensaje
x2 = np.zeros(m);
for w in range (m):
    if(error_bit == w):
        x2[w] = not(x[w]);
    else:
        x2[w] = x[w];

#Pocisionar bits en el nuevo mensaje
N = 0;
M = 0;
hamming3 = np.zeros(20);
for w in range (m+n):
    if((2**N)-1 == w):
        hamming3[w] = palabra[N];
        N = N + 1;
    else:
        op = m - M - 1;
        hamming3[w] = x2[op];
        M = M + 1;

#calcular los bits del codigo hamming (modulo a 2)
a3 = np.zeros(10);
b3 = 0;
for l in range(m+n):
    if hamming3[l] == 1:
        r3  = l+1;
        b3 = 0;
        while(r3>=1):
            if int(r3)%2 == 1:
                a3[b3] = a3[b3] + 1;
                r3 = r3/2
                b3 = b3 + 1;
            else:
                r3= r3/2;
                b3 = b3 + 1;
z3 = np.zeros(n);
for k in range(n):
    z3[k] = a3[k] % 2;

#print(a)
print("trama de entrada:", x)
#print(z)
#print(hamming)
#print(mod2)
print("modulo a2 (palabra hamming):", palabra)
print("trama con los bits de hamming posicionados: ", hamming2)
print("prueba de que no tiene errores:", z2)
print("bit de error:", 12-error_bit)
print("trama de entrada con error:", x2)
print("nueva palabra de hamming:", hamming3)
print("bit de error en hamming (izquierda menos significativo)", z3)

