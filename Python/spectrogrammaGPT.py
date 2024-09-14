import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

from scipy.io import loadmat, wavfile

# # Загрузка файла .mat (замените 'input.mat' на имя вашего файла)
# mat_data = loadmat('2023.03.31-11.59.31_channel_0.mat')

# sample_rate = 250

# # Извлечение аудиоданных из .mat файла (предположим, что данные находятся в переменной 'audio_data')
# audio_data = mat_data['dataa']

# # Запись аудиоданных в файл .wav (замените 'output.wav' на имя выходного файла)
# wavfile.write('Ice.wav', sample_rate, audio_data)

# Загрузка аудиофайла (замените 'audio.wav' на имя вашего файла)
sample_rate, audio_data = wavfile.read('TASCAM_0034S1.wav')
t = np.arange(len(audio_data))/sample_rate
# nt = (t < 450) & (t > 420)
# t1 = t[nt]
# t2 = np.zeros(len(t))

# audio_data1 = audio_data[nt]
 
plt.plot(t, audio_data, color='blue', label='Small')
plt.show()

# # Параметры спектрограммы
# window_size = 1024*64*2  # Размер окна для вычисления спектра
# overlap = window_size/2    # Перекрытие между окнами
# nfft = window_size   # Размер FFT (преобразования Фурье)

# # Вычисление спектрограммы
# frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate, window='hann', nperseg=window_size, noverlap=overlap, nfft=nfft)

# # Визуализация спектрограммы
# plt.figure(figsize=(10, 6))
# plt.imshow(10 * np.log10(Sxx), aspect='auto', cmap='inferno', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
# plt.colorbar(label='dB')
# plt.xlabel('Время (секунды)')
# plt.ylabel('Частота (Гц)')

# plt.title('Спектрограмма аудиофайла')
# plt.show()
