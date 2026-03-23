from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft, fft, fftfreq, fftshift, ifft2, fft2
from scipy import signal
import sklearn
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
dataset1 = loadmat('2023.03.31-11.44.02_channel_0.mat')
dataset2 = loadmat('2023.03.31-11.59.31_channel_0.mat')
dat1 = [[element for element in upperElement] for upperElement in dataset1['dataa']]
dat2 = [[element for element in upperElement] for upperElement in dataset2['dataa']]

Fs=48000 #Частота дискретизации
fl = 2000 #Частота до которой фильтруем
NFFT = 12000

# Нахождение минимума длины начального массива
ml = min(len(dat1[0]), len(dat2[0]))
if ml%2==1:
    ml = ml - 1
    
ml_half = int(ml/2)
if ml_half%2==1:
    ml_half = ml_half - 1
    
#Создание массивов
sl1 = []
sl2 = []
sl11 = np.array([])
sl22 = np.array([])
sl12 = []
S_N_in_arr = []
S_N_out_arr = []
corr1 = []
corr2 = []
sl1_half_1 = []
sl1_half_2 = []
sl1_filt = []

#Достаем массив данных
for i in range (0, int(ml)):
    sl1.append(dat1[0][i])
    
for i in range (0, int(ml)):
    sl2.append(dat2[0][i])
    
for i in range(0, int(ml_half)):
    sl1_half_1.append(sl1[i])
    
for i in range(int(ml_half), int(2*ml_half)):
    sl1_half_2.append(sl1[i])

# Набор функций
def corr_t(data1, data2, f1):
    mn = len(data1)
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
    corr_t = np.fft.fftshift(fft(s1*np.conj(s2)))
    return corr_t

def corr_f(data1, data2, f1):
    mn = len(data1)
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
    corr_f = ifft(fft((np.abs(s1))**2)*np.conj(fft((np.abs(s2))**2)))
    return corr_f

def norm(corr):
    norm = np.max(corr)
    return norm

def corr_corr(data1, data2, f1):
    mn = len(data1)
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

def plot1 (Y1, X1):
    plt.plot(X1, Y1, color='blue')
    # plt.xlim(-freq_filt,freq_filt)
    plt.show()
    
def plot2 (Y1, X1, Y2, X2):
    # plt.plot(X1, Y1, label = label1, color='blue')
    plt.plot(X2, Y2, color='red')
    plt.plot(X1, Y1, color='blue')
    # plt.xlim(-2000,2000)
    plt.legend()
    plt.grid(True)
    plt.show()

def db (S1, N1):
    return (20*np.log10(S1 / N1))

def plot_spectrogram(data, sampling_rate, NFFT):
    plt.figure(figsize=(10, 7))
    plt.specgram(data, Fs=sampling_rate, cmap='plasma', NFFT = NFFT)
    plt.ylim(0,2000)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.title('Spectrogram')
    plt.show()
    
# def flt_tanh(sgnl, f, f_down, f_up, df_down, df_up):
#     m_1 = 1/2 * ( np.tanh((f - f_down)/df_down) + 1)  +  1/2 * (1- np.tanh((f - (Fs-f_down))/df_down) ) -1          
#     m_2 = 2-1/2 * ( np.tanh((f - f_up)/df_up) + 1)  -  1/2 * (1- np.tanh((f - (Fs-f_up))/df_up) )   
#     f_mask = m_1 * m_2
#     y_n = f_mask * y1
#     sig_flt = 2*np.real(fft(y_n))
#     return f_mask, sig_flt
# [a, b] = flt_tanh(data1, f, 290, 320, 5, 5)
# c = flt(f, y1, 290, 320)



t = np.arange(int(ml))/Fs # Массив времен
t_half = np.arange(ml_half)/Fs

tc = t-int(ml)/Fs/2 # Массив времен отцентрованный
tc_half = t_half-ml_half/Fs/2

f = fftfreq(int(ml), 1 / Fs) # Массив частот

f_half = fftfreq(ml_half, 1 / Fs) # Массив частот

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=t, y=sl1))
# fig.show()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=t, y=sl2))
# fig.show()

# Ообрезать диапазаон частот тангенсом
# [a, b] = flt_tanh(data1, f, 290, 320, 5, 5)
# c = flt(f, y1, 290, 320)

# Попытка обрезать сигнал в частотном диапазоне
# sl1_f = ifft(sl1)
# sl1_f_filt = np.zeros(ml)
# for i in range (0, ml):
#     if f[i] > -fl:
#         if f[i] < fl:
#             sl1_f_filt[i] = sl1[i]
# s1_filt = fft(sl1_f_filt)

# plot1(corr_t(sl1, sl1, 20000), tc)

d1 = (np.abs(fft(sl1)))**2
d2 = (np.abs(fft(sl2)))**2
y = corr_f(d1, d2, 1000)
plot_spectrogram(sl2, Fs, NFFT = 12000)

x = np.arange(0, 5, 0.1)
fig = px.scatter(x=x, y=x)
fig.show()

# Для разных половинок
# corr1 = corr_corr(sl1_half_1, sl1_half_1, 2000)
# corr2 = corr_corr(sl1_half_1, sl1_half_2, 2000)

# Для разных сигналов
# corr1 = corr_corr(sl1, sl1, 2000)
# corr2 = corr_corr(sl1, sl2, 2000)


# sl11 = np.append(sl11, sl1)
# X1 = f1, t1, Sxx1 = signal.spectrogram(sl11, Fs, nfft = NFFT)
# sl22 = np.append(sl22,sl1)
# X2 = f2, t2, Sxx2 = signal.spectrogram(sl22, Fs, nfft = NFFT)
# sl11_f = fft2(Sxx1)
# sl22_f = fft2(Sxx2)
# corr_spec = sl11_f*np.conj(sl22_f)
# corr_spec = ifft2(corr_spec)

# index = f1 < 2401
# index1 = t1 <  63
# Sxx1 = Sxx1[index]

# plt.figure(figsize=(10, 7))
# plt.imshow(Sxx1, cmap='plasma', vmin = 0, vmax= 7000)
# plt.colorbar()
# plt.show()


# # Построение различных графиков
# plot1(sl1, t)
# plot1(sl2, t)

# plot1(corr_t(sl1, sl2, 750), tc)
# plot1(corr_t(sl1_half_1, sl1_half_2, 750), tc_half)
# plot1(corr_f(sl1, sl2, 2000), f)
# plot1(corr_f(sl1_half_1, sl1_half_2, 2000), f_half)
# plot1(corr_corr(sl1, sl2, 750), tc)
# plot1(corr_corr(sl1_half_1, sl1_half_1, 750), tc_half)


# plt.plot(f, corr_f(sl1, sl2, 750), color='blue')
# plt.xlim(-2000, 2000)
# plt.show()

# plt.plot(f_half, corr_f(sl1_half_1, sl1_half_2, 750), color='blue')
# plt.xlim(-2000, 2000)
# plt.show()

# plt.plot(f, np.abs(ifft(sl1)), color='blue')
# plt.xlim(-2000, 2000)
# plt.show()

# plt.plot(f, np.abs(ifft(sl2)), color='blue')
# plt.xlim(-2000, 2000)
# plt.show()


# plot1(np.abs(ifft(sl1)), f)
# plot1(np.abs(ifft(sl2)), f)
# plot2(sl1, t, sl2, t,'','','','','')
# plot2(corr1, tc, corr2, tc,'','','','','')

# plt.plot(corr_corr(sl1_half_1,sl1_half_2, 20000))
# # plt.xlim(1250000,1750000)
# plt.show

# Построение графика зависимости сигнал-шум
# S1 = np.std(sl1) 
# n = np.random.randn(ml)*S1*2 # массив данных шума
# N1 = np.std(n)

# print(20*np.log10(N1/(2*10**(-5))))

# plot1(sl1, t, 'Усл.ед', 't, c', 'Signal')
# plot1(n, t, 'Усл.ед', 't, c', 'Noise')
# print(db(S1,N1))

# sl1n = sl1 + n
# spec1 = corr(sl1, sl1n)
# plot1(spec1, tc, 'Усл.ед', 't, c', 'Correlation')

# for i in range (1, 51, 2):
#     n1 = n*i
#     sl1n1 = sl1 + n1
#     corr_sl1sl1n = corr(sl1, sl1n1) # корреляция сигнала и шум+сигнал
#     corr_cl1n1 = corr(sl1, n1) # корреляция сигнала и шума
#     #plot2(corr_sl1sl1n, tc, corr_cl1n1, tc, 'Усл.ед', 't, c', 'Корелляции сигнал, шум и сигнал, сигнал + шум', 'Сигнал и сигнал + шум', 'Сигнал и шум')
    
#     N1 = np.std(n1)
#     S_N_in = 20*np.log10(S1 / N1)
#     S_N_in_arr.append(S_N_in)
    
#     it1 = (tc > -0.5) & (tc < 0.5)
#     it2 = (tc > 10) & (tc < 20)
#     corr_centre = corr_sl1sl1n[it1]
#     corr_noise = corr_cl1n1[it2]
    
#     S2 = np.max(corr_centre)
#     N2 = np.std(corr_noise)
#     S_N_out_corr = 20*np.log10(S2 / N2)
#     S_N_out_arr.append(S_N_out_corr)
    
# plot1(S_N_out_arr, S_N_in_arr, 'Сигнал на выходе, Дб', 'Сигнал на входе, Дб', 'Сигнал/Шум')




















