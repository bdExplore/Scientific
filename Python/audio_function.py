
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft, fft, fftfreq, fftshift, ifft2, fft2
from scipy import signal
from scipy.io import wavfile
import sklearn
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta


# Загрузка экспериментальных данных и используемые методы
sample_rate, audio_data49_3 = wavfile.read('/Users/bogda/Desktop/Acoustics/Scientific/Исходные данные/ExpSummer2024/TASCAM_Files/TASCAM_0049S3.wav')
sample_rate, audio_data49_2 = wavfile.read('/Users/bogda/Desktop/Acoustics/Scientific/Исходные данные/ExpSummer2024/TASCAM_Files/TASCAM_0049S2.wav')
sample_rate, audio_data16_3 = wavfile.read('/Users/bogda/Desktop/Acoustics/Scientific/Исходные данные/ExpSummer2024/TASCAM_Files/TASCAM_0016S3.wav')
sample_rate, audio_data16_2 = wavfile.read('/Users/bogda/Desktop/Acoustics/Scientific/Исходные данные/ExpSummer2024/TASCAM_Files/TASCAM_0016S3.wav')
df = pd.read_excel('/Users/bogda/Desktop/Acoustics/Scientific/Таблицы Данных/Conducting an experiment 30_08.xlsx')

t = np.arange(int(len(audio_data49_3)))/sample_rate # Массив времен


def plot1 (X1, Y1, name, xlabel = '', ylabel = 'Амплитуда, у.е.'):
    plt.figure(figsize=(10,4))
    plt.plot(X1, Y1, color='blue')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(name)
    plt.grid(True)
    plt.show()
    
    
def plot1_f (X1, Y1, name, freq_sep:int, xlabel = '', ylabel = 'Амплитуда, у.е.'):
    plt.figure(figsize=(10,4))
    plt.plot(X1, Y1, color='blue')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(name)
    plt.grid(True)
    plt.xlim(-freq_sep,freq_sep)
    plt.show()   
    

def corr_t(data1, data2, f_filt):
    mn=min([len(data1), len(data2)])
    data1 = data1[0:mn]
    data2 = data2[0:mn]
    data1_f = ifft(data1)
    data2_f = ifft(data2)

    data1_f = filt_freq(data1_f, f_filt)
    data2_f = filt_freq(data2_f, f_filt)

    return np.fft.fftshift(fft(data1_f*np.conj(data2_f)))


def corr_f(data1, data2, f1):
    mn = min(len(data1), len(data2))
    data1_f = ifft(data1)
    data2_f = ifft(data2)
    s1 = np.zeros(mn)
    s2 = np.zeros(mn)
    f = fftfreq(mn, 1 / sample_rate)

    for i in range (0, mn):
        if f[i] > -f1:
            if f[i] < f1:
                s1[i] = data1_f[i]
                s2[i] = data2_f[i]

    fft_abs_s1 = fft((np.abs(s1))**2)
    fft_abs_s2 = fft((np.abs(s2))**2)
    
    corr_f = ifft((fft_abs_s1)*np.conj(fft_abs_s2))
    return corr_f


def count_time(first, df):
    cf_time = '15:26:16'
    cd_time = str(df.loc[first, 'Start time'])
    dur_time = str((df.loc[first, 'Duration']))
    c_time_dt = datetime.strptime(cf_time, "%H:%M:%S")
    x_time_dt = datetime.strptime(cd_time, "%H:%M:%S")
    duration = datetime.strptime(dur_time, "%H:%M:%S")
    # print(type(x_time_dt))
    delta = (x_time_dt -  c_time_dt).seconds%86400
    time_b_1 = delta
    duration_sec = duration.hour*3600 + duration.minute*60 + duration.second
    print(duration_sec)
    time_b_2 = delta + duration_sec
    return time_b_1, time_b_2, duration_sec


def cr_arr_t(first):
    t1_beg, t1_end, duration = count_time(first, df)
    t_int1 = (t > t1_beg) & (t <= t1_end)
    t1 = t[t_int1]
    f1 = fftfreq(int(len(t1)), 1 / sample_rate) # Массив частот
    
    array = [t_int1, t1, f1, duration]
    return array


def norm_max (data):
    return data/max(data)


def load_mat(data_1_str: str):
    dataset = loadmat(data_1_str)
    dat = [[element for element in upperElement] for upperElement in dataset['dataa']]
    data=[]
    
    for i in range (0,len(dat[0])):
        data.append(dat[0][i])

    if len(dat[0])%2==1:
        data=data[:len(dat[0])-1]

    return data


def first_second_part(data):
    data_first = []
    data_second = []
    for i in range(0,int(len(data)/2)):
        data_first.append(data[i])
    
    for i in range(int(len(data)/2),int(len(data))):
        data_second.append(data[i])

    return data_first, data_second


def filt_freq(data, f_filt):
    l = len(data)
    array = np.zeros(l)
    data_f = (ifft(data))
    f = fftfreq(int(l), 1 / sample_rate)
    f_int = (f > -f_filt) & (f < f_filt)
    array[f_int] = (data_f[f_int])
    return (fft(array))


def cos_sim (data1, data2, f_filt):
    # mn = min(len(data1), len(data2))
    # data1 = data1[:mn]
    # data2 = data2[:mn]
    data1 = filt_freq(data1, f_filt)
    data2 = filt_freq(data2, f_filt)
    data1 = np.array(ifft(data1))
    data2 = np.array(ifft(data2))
    
    data12 = np.sum(data1*data2)
    norm1 = np.sum(data1**2)
    norm2 = np.sum(data2**2)
    if norm1 == 0 :
        print('На ноль не делим_1')
    if norm2 == 0 :
        print('На ноль не делим_2')

    data_norm12 = np.sqrt(norm1) * np.sqrt(norm2)
    SN = data12/data_norm12

    return SN


def mean_portrait_signal(portrait, signal, f_filt):
   max1 = np.max(np.abs(corr_t(signal, signal, f_filt) * corr_t(portrait, portrait, f_filt)))
   max_signal = np.sqrt(np.max(np.abs(corr_t(portrait, portrait, f_filt) * corr_t(portrait, portrait, f_filt)))) * np.sqrt(np.max(np.abs(corr_t(signal, signal, f_filt) * corr_t(signal, signal, f_filt))))
   a = max1 / max_signal
   return a


def mean_data_sep(data, t_sep):
    l = len(data)
    n = int(l/sample_rate/t_sep)
    mean_ifft_data = np.zeros(int(sample_rate/n), dtype=complex)

    if l%n != 0:
       data = data[:n*sample_rate*t_sep]
        
    for i in range (0, n):
        data_t_sec = data[int(i*sample_rate*t_sep) : int((i+1)*sample_rate*t_sep)]
        mean_ifft_data += np.abs(ifft(data_t_sec))
    
    return np.array(fft(mean_ifft_data))



def variation_data(data, t_sep, s_r = sample_rate):
    l = len(data)
    l_sec = l * s_r
    counts_per_t_sep = s_r*t_sep
    n = int(l/counts_per_t_sep)
    var_arr = []
    if (l_sec%t_sep) != 0:
        data = data[:int(n*counts_per_t_sep)]
    for i in range (0, n):
        j = int((i+1)*counts_per_t_sep)
        data_t_sec = data[:j]
        f = fftfreq(int(len(data_t_sec)), 1 / sample_rate)
        # plot1_f(f, np.abs(ifft(data_t_sec)), 'спектры частей отрезка', 1000)
        variation = np.std(np.abs(ifft(data_t_sec)))/np.mean(np.abs(ifft(data_t_sec)))
        var_arr.append(variation)
    return var_arr


def t_central(t, sample_rate):
    t_c = t - len(t)/sample_rate/2
    return t_c


def t_arr_for_corr_t(data):
    l = len(data)
    t = np.arange(l)/sample_rate
    t = t_central(t, sample_rate)
    return t
