from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft, fft, fftfreq

dataset1 = loadmat('2023.03.31-11.59.31_channel_0.mat')
print(dataset1)

Fs=48000 #Частота дискретизации
f1 = 1000 #Частота фильтрации
data1=[]
data2=[]
data11 = []
data12 = []
S_N_in_arr = []
S_N_out_arr_corr_corr = []
S_N_out_arr_corr = []

dat1 = [[element for element in upperElement] for upperElement in dataset1['dataa']]

for i in range (0,len(dat1[0])):
    data1.append(dat1[0][i])


if len(dat1[0])%2==1:
    data1=data1[:len(dat1[0])-1]


for i in range(0,int(len(data1)/2)):
    data11.append(data1[i])
    
for i in range(int(len(data1)/2),int(len(data1))):
    data12.append(data1[i])

def corr(data1, data2):
    corr = np.fft.fftshift(fft(ifft(data1)*np.conj(ifft(data2))))
    return corr

def norm(corr):
    norm = np.max(corr)
    return norm

def corr_corr(data1, data2, f1, mn):
    data10 = ifft(data1)
    data20 = ifft(data2)
    s1 = np.zeros(mn)
    s2 = np.zeros(mn)
    f = fftfreq(mn, 1 / Fs)
    for i in range (0, mn):
        if f[i] > -f1:
            if f[i] < f1:
                s1[i] = data10[i]
                s2[i] = data20[i]
    corrr = np.fft.fftshift(fft(np.abs(s1)**2*(np.abs(s2))**2))
    # corrr = np.fft.fftshift(fft(ifft(data1)*np.conj(ifft(data2))))
    corrr = corrr.real
    return corrr

def plot1 (X1, Y1, ylabel, xlabel):
    plt.plot(X1, Y1, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def plot2 (X1, Y1, X2, Y2, ylabel, xlabel):
    plt.plot(X1, Y1, color='blue')
    plt.plot(X2, Y2, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
mn=min([len(data11), len(data12)])

data11 = data11[0:mn]
data12 = data12[0:mn]

t = np.arange(mn)/Fs
tc = t-mn/Fs/2

n = norm(corr_corr(data11,data11,f1,mn))
S1 = np.std(data12)
rnd = np.random.randn(mn)*S1*2 # массив данных шума

for i in range (1, 51, 2):
    rnd1 = rnd*i
    data111 = data12 + rnd1
    c12 = corr_corr(data11, data111, f1, mn) # корреляция сигнала и шум+сигнал
    c13 = corr_corr(data11, rnd1, f1, mn) # корреляция сигнала и шума
    c14 = corr(data11, data111) # корреляция сигнала и шум+сигнал
    c15 = corr(data11, rnd1) # корреляция сигнала и шума
    # plt.plot(tc, data12, color='blue')
    # plt.xlabel("tc")
    # plt.ylabel("Rnd")
    # plt.show()
    
    # plt.plot(tc, c12, color='red')
    # plt.plot(tc, c13, color='blue')
    # plt.xlabel("tc")
    # plt.ylabel("Corr_Corr")
    # plt.show()
    # print(i)
    
    N1 = np.std(rnd1)
    S_N_in = 20*np.log10(S1 / N1)
    S_N_in_arr.append(S_N_in)
    
    S2 = np.max(c12)
    # N2 = np.std(c12[tc>5])
    N2 = np.max(c13)
    S_N_out_corr = 20*np.log10(S2 / N2)
    S_N_out_arr_corr.append(S_N_out_corr)
    
    S3 = np.max(c14)
    # N2 = np.std(c12[tc>5])
    N3 = np.max(c15)
    S_N_out_corr_corr = 20*np.log10(S3 / N3)
    S_N_out_arr_corr_corr.append(S_N_out_corr_corr)
    
plt.plot(S_N_in_arr, S_N_out_arr_corr, color='red', label='Small')
plt.plot(S_N_in_arr, S_N_out_arr_corr_corr, color='blue', label='Small')
plt.xlabel("S_N_in_arr")
plt.ylabel("S_N_out_arr")
plt.show()




