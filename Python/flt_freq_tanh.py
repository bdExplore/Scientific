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

# plt.plot(t, data2, color='red', label='Big')

plt.plot(t, data2, color='blue', label='Small')
plt.xlabel("t, с")
plt.ylabel("A")
plt.show()

# nt = np.arange(2*mn)/Fs
# ntc = nt-2*mn/Fs/2
    
# ndata1 = np.zeros(2*len(data1))
# ndata2 = np.zeros(2*len(data2))

# for i in range (int(mn/2), int(3*mn/2)):
#     ndata1[i] = data1[i - int(mn/2)]
#     ndata2[i] = data2[i - int(mn/2)]

# plt.plot(nt, ndata2, color = 'blue', label = 'Big')
# plt.plot(nt, ndata1, color = 'red', label = 'Small')
# plt.xlabel("t, с")
# plt.ylabel("B")
# plt.show()

y1 = ifft(data1)
y2 = ifft(data2)
f = fftfreq(len(data1), 1 / Fs)
m_1 = np.zeros(len(f))
m_2 = np.zeros(len(f))
f_mask = np.zeros(len(f))
y_n = np.zeros(len(f))

plt.plot(f, np.abs(y2), color='red', label='Small')
plt.xlabel("f, Гц")
plt.ylabel("X(f), отн.ед")
plt.xlim(0,2500)
plt.show()


def flt (f, y, begin, end):
    z = np.zeros(mn, dtype=complex)
    for i in range (0, mn):
        if f[i] > -end:
            if f[i] < -begin:
                z[i] = y[i]
                
    for i in range (0, mn):
        if f[i] > begin:
            if f[i] < end:
                z[i] = y[i]
    return z


def flt_tanh(sgnl, f, f_down, f_up, df_down, df_up):
    m_1 = 1/2 * ( np.tanh((f - f_down)/df_down) + 1)  +  1/2 * (1- np.tanh((f - (Fs-f_down))/df_down) ) -1          
    m_2 = 2-1/2 * ( np.tanh((f - f_up)/df_up) + 1)  -  1/2 * (1- np.tanh((f - (Fs-f_up))/df_up) )   
    f_mask = m_1 * m_2
    y_n = f_mask * y1
    sig_flt = 2*np.real(fft(y_n))
    return f_mask, sig_flt

[a, b] = flt_tanh(data1, f, 290, 320, 5, 5)
c = flt(f, y1, 290, 320)

plt.plot(t, 2 * np.real(fft(a)), color='blue', label='Big')
plt.xlabel("t, c")
plt.ylabel("X(t), отн.ед")
plt.xlim(-1500,1500)
plt.show()

plt.plot(t, np.real(fft(c)), color='blue', label='Big')
plt.xlabel("t, c")
plt.ylabel("X(t), отн.ед")
plt.xlim(30.4,30.45)
plt.show()

plt.plot(f, a, color='blue', label='Big')
plt.xlabel("f, Гц")
plt.ylabel("X(f), отн.ед")
plt.xlim(-1500,1500)
plt.show()

plt.plot(f, c, color='blue', label='Big')
plt.xlabel("f, Гц")
plt.ylabel("X(f), отн.ед")
plt.xlim(-1500,1500)
plt.show()
