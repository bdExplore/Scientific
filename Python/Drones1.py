from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft, fft, fftfreq

dataset1 = loadmat('2023.03.31-11.44.02_channel_0.mat')
print(dataset1)
dataset2 = loadmat('2023.03.31-11.59.31_channel_0.mat')
print(dataset2)

Fs=48000 #Частота дискретизации
data1=[]
data2=[]
data11 = []
data12 = []

dat1 = [[element for element in upperElement] for upperElement in dataset1['dataa']]
dat2 = [[element for element in upperElement] for upperElement in dataset2['dataa']]

for i in range (0,len(dat1[0])):
    data1.append(dat1[0][i])

for i in range (0,len(dat2[0])):
    data2.append(dat2[0][i])

if len(dat1[0])%2==1:
    data1=data1[:len(dat1[0])-1]
if len(dat2[0])%2==1:
    data2=data2[:len(dat2[0])-1]

for i in range(0,int(len(data1)/2)):
    data11.append(data1[i])
    
for i in range(int(len(data1)/2),int(len(data1))):
    data12.append(data1[i])

def corr(data1,data2):
    corr = np.fft.fftshift(fft(ifft(data1)*np.conj(ifft(data2))))
    return corr
def norm(corr):
    norm = np.max(corr)
    return norm
def corr_corr(data1, data2):
    corrr = np.fft.fftshift(fft(np.abs(ifft(data1))**2*(np.abs(ifft(data2)))**2))
    corrr = corrr.real
    return corrr

mn=min([len(data1), len(data2)])

data1 = data1[0:mn]
data2 = data2[0:mn]

t = np.arange(mn)/Fs
tc = t-mn/Fs/2

plt.plot(t, data2, color='blue', label='Big')
plt.plot(t, data1, color='red', label='Small')
plt.xlabel("t, с")
plt.ylabel("A")
plt.show()


y1 = ifft(data1)
y2 = ifft(data2)
f = fftfreq(len(data1), 1 / Fs)


plt.plot(f, np.abs(y2)/max(np.abs(y2)), color='blue', label='Big')
plt.plot(f, np.abs(y1)/max(np.abs(y1)), color='red', label='Small')
plt.xlabel("f, Гц")
plt.xlim(0,2000)
plt.show()

plt.plot(tc, corr_corr(data1, data2))

s1 = np.zeros(mn)
s2 = np.zeros(mn)

for i in range (0, mn):
    if f[i] > -1000:
        if f[i] < 1000:
            s1[i] = y1[i]
            s2[i] = y2[i]


# plt.plot(f, np.abs(s1), color='red', label='Small')
# plt.plot(f, np.abs(s2), color='blue', label='Big')
# plt.xlabel("f")
# plt.ylabel("X(f)")
# plt.xlim([-1000, 1000])
# plt.show()

# plt.plot(tc, corr(data1,data2) / norm(corr(data1,data1)), color='blue', label='Small')
# plt.xlabel("t")
# plt.ylabel("X(f)")
# plt.show()

# сигнал шума
# rnd = np.random.randn(mn)*10**5

# S = np.std(data2)
# N = np.std(rnd)
# S_N_in = 20*np.log10(S/N)
# # print(S_N)
# # sp_rnd = ifft(rnd)
# plt.plot(t, rnd, color='blue', label='Small')
# plt.plot(t, data1, color='red', label='Small')
# plt.xlabel("t")
# plt.ylabel("X")
# plt.show()

# data11 = data2 + rnd
# plt.plot(f, np.abs(ifft(data11)), color='blue', label='Small')
# plt.xlabel("t")
# plt.ylabel("X(f)")
# plt.show()

# plt.plot(tc, corr(data1,data11) / norm(corr(data1,data1)), color='blue', label='Small')
# plt.xlabel("t")
# plt.ylabel("Corr")
# plt.show()


# c12 = corr_corr(data1,data11) / norm(corr_corr(data1,data1))

# plt.plot(tc, c12, color='blue', label='Small')
# plt.xlabel("t")
# plt.ylabel("Corr_Corr")
# plt.show()

# S = np.max(c12)
# N = np.std(c12[tc>10])
# S_N_out = 20*np.log10(S/N)
