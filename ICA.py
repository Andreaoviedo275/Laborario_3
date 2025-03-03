import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import sounddevice as sd  # Librería para reproducir audio

# Función para cargar y normalizar señales
def cargar_y_normalizar_audio(archivo):
    fs, señal = wavfile.read(archivo)
    # Normalizar la señal
    señal = señal.astype(np.float32) / np.max(np.abs(señal))
    return fs, señal

# Cargar las señales de audio
fs_ruido, ruido = cargar_y_normalizar_audio('RuidoAmbiente.wav')
fs_voz1, voz1 = cargar_y_normalizar_audio('VozAndrea.wav')
fs_voz2, voz2 = cargar_y_normalizar_audio('VozLaura.wav')

# Asegurar que todas las señales tengan la misma longitud
min_length = min(len(ruido), len(voz1), len(voz2))
ruido = ruido[:min_length]
voz1 = voz1[:min_length]
voz2 = voz2[:min_length]

# Aumentar la amplitud de las señales de voz para mejorar el SNR
factor_ganancia = 10  # Ajusta este valor según sea necesario
voz1_amplificada = voz1 * factor_ganancia
voz2_amplificada = voz2 * factor_ganancia

# Crear una señal combinada de las voces sobre el ruido
combinado_voz1 = ruido + voz1_amplificada
combinado_voz2 = ruido + voz2_amplificada

# Función para calcular la relación señal a ruido (SNR)
def calcular_snr(señal_original, señal_combinada):
    ruido = señal_combinada - señal_original
    potencia_señal = np.var(señal_original)
    potencia_ruido = np.var(ruido)
    snr = 10 * np.log10(potencia_señal / potencia_ruido)
    return snr

# Calcular SNR para Voz Andrea y Voz Laura
snr_voz1 = calcular_snr(voz1_amplificada, combinado_voz1)
snr_voz2 = calcular_snr(voz2_amplificada, combinado_voz2)

# Mostrar los valores de SNR
print(f"SNR Voz Andrea sobre Ruido: {snr_voz1:.2f} dB")
print(f"SNR Voz Laura sobre Ruido: {snr_voz2:.2f} dB")

# Graficar las señales juntas
plt.figure(figsize=(10, 6))
tiempo = np.arange(len(ruido)) / fs_ruido

plt.plot(tiempo, combinado_voz1, label='Voz Andrea sobre Ruido', color='orange', alpha=0.7)
plt.plot(tiempo, combinado_voz2, label='Voz Laura sobre Ruido', color='pink', alpha=0.7)
plt.plot(tiempo, ruido, label='Ruido', color='yellow', alpha=0.5)

plt.title('Señales combinadas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()

# Apilar las señales en una matriz de 2 filas (una para cada mezcla) 
X = np.c_[combinado_voz1, combinado_voz2]  # Aquí las mezclas se apilan en columnas

# Aplicar ICA para separar las señales
ica = FastICA(n_components=2)  # Queremos separar 2 señales
X_separadas = ica.fit_transform(X)  # Esto te da las señales separadas

# Recuperar las señales separadas
voz1_separada = X_separadas[:, 0]
voz2_separada = X_separadas[:, 1]

# Graficar las señales separadas junto con los espectros en escala semilogarítmica
plt.figure(figsize=(10, 12))

# Graficar la señal separada Voz Andrea
plt.subplot(3, 2, 1)
plt.plot(np.arange(len(voz1_separada)) / fs_ruido, voz1_separada, label='Voz Andrea Separada', color='orange')
plt.title('Voz Andrea Separada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

# Calcular la FFT de la señal separada Voz Andrea
N = len(voz1_separada)
freqs = np.fft.fftfreq(N, 1/fs_ruido)
X_voz1 = np.abs(np.fft.fft(voz1_separada))

# Graficar el espectro en escala semilogarítmica para Voz Andrea
plt.subplot(3, 2, 2)
plt.semilogx(freqs[:N//2], X_voz1[:N//2], label='Espectro Voz Andrea', color='orange')
plt.title('Espectro de Voz Andrea (Escala Semilogarítmica)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True)

# Graficar la señal separada Voz Laura
plt.subplot(3, 2, 3)
plt.plot(np.arange(len(voz2_separada)) / fs_ruido, voz2_separada, label='Voz Laura Separada', color='pink')
plt.title('Voz Laura Separada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

# Calcular la FFT de la señal separada Voz Laura
X_voz2 = np.abs(np.fft.fft(voz2_separada))

# Graficar el espectro en escala semilogarítmica para Voz Laura
plt.subplot(3, 2, 4)
plt.semilogx(freqs[:N//2], X_voz2[:N//2], label='Espectro Voz Laura', color='pink')
plt.title('Espectro de Voz Laura (Escala Semilogarítmica)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True)

# Graficar el ruido original
plt.subplot(3, 2, 5)
plt.plot(np.arange(len(ruido)) / fs_ruido, ruido, label='Ruido', color='yellow', alpha=0.5)
plt.title('Ruido Original')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

# Calcular la FFT del ruido
X_ruido = np.abs(np.fft.fft(ruido))

# Graficar el espectro en escala semilogarítmica para el Ruido
plt.subplot(3, 2, 6)
plt.semilogx(freqs[:N//2], X_ruido[:N//2], label='Espectro Ruido', color='yellow')
plt.title('Espectro de Ruido (Escala Semilogarítmica)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True)

plt.tight_layout()
plt.show()

# Si quieres guardar las señales separadas en archivos WAV
def guardar_audio(archivo, fs, señal):
    señal = np.int16(señal * 32767)  # Convertir de vuelta a 16-bit PCM
    wavfile.write(archivo, fs, señal)

# Guardar las señales separadas como archivos WAV
guardar_audio('VozAndreaSeparada.wav', fs_ruido, voz1_separada)
guardar_audio('VozLauraSeparada.wav', fs_ruido, voz2_separada)

# Reproducir la Voz de Andrea
sd.play(voz1_separada, fs_ruido)  # Reproduce la señal separada de Voz Andrea
sd.wait()  # Espera a que termine la reproducción
