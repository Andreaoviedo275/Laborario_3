# Laborario_3: Problema de coctel

Este proyecto aborda el problema del cóctel, en el cual se busca aislar una voz de interés a partir de una mezcla de múltiples fuentes sonoras.

Se implementan dos métodos de separación de señales:

✔ Análisis de Componentes Independientes (ICA) → Para extraer la voz de Laura.

✔ Análisis de Componentes Principales (PCA) → Para extraer la voz de Andrea.

🔹 Frecuencia de muestreo: 44,100 Hz (Apple arroja eso que es nuestro sistema de adquisición).

🔹 Duración de las grabaciones: 15 segundos.

📌 ¿Qué es el Problema del Cóctel?

El Problema del Cóctel es un fenómeno en el que el cerebro humano es capaz de enfocarse en una sola conversación dentro de un ambiente lleno de ruido y múltiples voces.

💡 Ejemplo real:
Imagina que estás en una fiesta con muchas personas hablando al mismo tiempo. A pesar del ruido, puedes concentrarte en la voz de tu amigo y entender lo que dice sin que los otros sonidos te distraigan demasiado.

📢 ¿Cómo se aplica esto en procesamiento de señales?

- Queremos que un algoritmo escuche una grabación con varias voces y ruido y sea capaz de separar una voz específica de la mezcla.
- Esto es útil para aplicaciones como:
  - Reconocimiento de voz (Siri, Google Assistant).
  - Audífonos con cancelación de ruido.
  - Análisis forense de audio.

📌 Métodos utilizados en este laboratorio:

✅ ICA (Análisis de Componentes Independientes): Separación de la voz de Laura.

✅ PCA (Análisis de Componentes Principales): Separación de la voz de Andrea.

📊 Datos de Adquisición y Parámetros Técnicos

📌 Fuente de los datos:

Las grabaciones fueron adquiridas con un sistema de Apple, el cual proporciona una frecuencia de muestreo de 44,100 Hz, lo que significa que el audio fue capturado 44,100 veces por segundo.

📌 Duración de los audios:

Cada grabación tiene una duración de 15 segundos.

📌 Señales capturadas:

🎤 Voz 1: Andrea.

🎤 Voz 2: Laura.

🌍 Ruido ambiental: Captado por otro celular.

![Imagen de WhatsApp 2025-03-03 a las 20 22 55_ba03838e](https://github.com/user-attachments/assets/772f1a1d-51f6-4930-9aed-6cd7ae1c8748)
Fig 1. Organización de los microfonos 


🚀 Instalación y Configuración
Antes de ejecutar los scripts, instala las dependencias necesarias:

        pip install -r requirements.txt
        
Si no tienes el archivo requirements.txt, instala manualmente:

        pip install numpy matplotlib scipy scikit-learn sounddevice

🔍 Métodos de Separación de Voz

1️⃣ Método ICA (Independent Component Analysis) – Separación de la Voz de Laura

📌 ¿Qué es ICA?

El Análisis de Componentes Independientes (ICA) es un método que permite separar señales que han sido mezcladas en un mismo entorno, bajo la suposición de que las señales originales son estadísticamente independientes entre sí.

🔎 Ejemplo práctico:

- Imagina que tienes dos micrófonos en una sala donde están hablando Laura y Andrea.
- Cada micrófono captará una mezcla de ambas voces con diferente intensidad.
- ICA analizará la señal e intentará separar las fuentes independientes, recuperando las voces originales.

📌 Pasos realizados en ICA:

✔️ Cargamos las señales de los micrófonos.

✔️ Aplicamos ICA para extraer las voces.

✔️ Guardamos la voz de interés en un archivo WAV.

✔️ Analizamos el espectro de frecuencia antes y después de la separación.

📌 Código (fragmento de ICA):

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
        
        # Guardar la voz separada en un archivo WAV
        guardar_audio('VozLauraSeparada_ICA.wav', fs_ruido, voz2_separada)

![ICA-ESPECTRO](https://github.com/user-attachments/assets/4f27dc92-1aa6-4847-acbd-c9e35dae6ff0)
Fig 2. Ica espectro

![ICA-TODAS](https://github.com/user-attachments/assets/768700c2-46b8-43f8-bed6-684bfe4ee758)
Fig 3. Ica todas

2️⃣ Método PCA (Principal Component Analysis) – Separación de la Voz de Andrea

📌 ¿Qué es PCA?

El Análisis de Componentes Principales (PCA) es un método matemático utilizado para reducir la dimensionalidad de los datos sin perder información relevante. En este caso, lo usamos para separar las voces mezcladas con el ruido.

📌 Diferencias entre PCA e ICA:

- ICA intenta separar señales independientes entre sí.
- PCA encuentra las direcciones de mayor variabilidad en los datos, lo que permite distinguir mejor cada señal.
  
📌 Pasos realizados en PCA:

✔️ Carga y normalización de las señales de audio.

✔️ Asegurar que todas las señales tengan la misma duración.

✔️ Aumentar la amplitud de las señales de voz para mejorar la relación señal-ruido (SNR).

✔️ Combinar las señales de voz con el ruido ambiental.

✔️ Calcular el SNR antes y después de la separación.

✔️ Aplicar PCA para extraer las voces.

✔️ Guardar las voces separadas en archivos WAV.

✔️ Graficar las señales separadas y sus espectros de frecuencia.

📌 Código (fragmento de PCA):

        import numpy as np
        import scipy.io.wavfile as wavfile
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
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
        
        # Graficar las señales combinadas
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
        
        # Aplicar PCA para separar las señales
        pca = PCA(n_components=2)  # Queremos separar 2 señales
        X_separadas = pca.fit_transform(X)  # Esto te da las señales separadas
        
        # Recuperar las señales separadas
        voz1_separada = X_separadas[:, 0]
        voz2_separada = X_separadas[:, 1]
        
        # Guardar las señales separadas como archivos WAV
        def guardar_audio(archivo, fs, señal):
            señal = np.int16(señal * 32767)  # Convertir de vuelta a 16-bit PCM
            wavfile.write(archivo, fs, señal)
        
        guardar_audio('VozAndreaSeparada_PCA.wav', fs_ruido, voz1_separada)
        guardar_audio('VozLauraSeparada_PCA.wav', fs_ruido, voz2_separada)
        
        # Reproducir la Voz de Andrea
        sd.play(voz1_separada, fs_ruido)  # Reproduce la señal separada de Voz Andrea
        sd.wait()  # Espera a que termine la reproducción

![PSA-ESPECTRO](https://github.com/user-attachments/assets/d3228fab-2cd0-48d5-afac-30651b8f35fb)
Fig 4. PCA espectro

![PCA-TODAS](https://github.com/user-attachments/assets/17233842-397c-4eab-8230-da00f576c101)
Fig 5. PCA todas


📈 Resultados y Análisis 

Después de aplicar los métodos ICA y PCA para separar las voces de Andrea y Laura en un ambiente ruidoso, se realizaron diversas evaluaciones para comparar la efectividad de cada técnica.

 1️⃣ Comparación de Métodos

 - ICA logra una mejor separación de la voz de Laura, ya que se basa en la independencia estadística de las fuentes de sonido.
 - PCA funciona bien, pero no es tan efectivo en este contexto, ya que la correlación entre las señales de voz y ruido no es lo suficientemente fuerte para lograr una separación óptima.

2️⃣ Análisis de la Relación Señal/Ruido (SNR)

📌 Definición de SNR:

El SNR (Signal-to-Noise Ratio) mide qué tan fuerte es la señal deseada en comparación con el ruido. Se expresa en decibeles (dB) y se calcula como:

![{F4CFC1C5-5424-4002-A313-1D0534AD347A}](https://github.com/user-attachments/assets/01b0d82b-bc20-4c1c-8851-dd19a90f652f)

Fig 6. Fórmula del SNR

📌 Valores obtenidos:

- ICA	5 - 7 dB (mala calidad)	22 - 25 dB (alta calidad)
- PCA	5 - 7 dB (mala calidad)	15 - 18 dB (calidad moderada)

📌 Análisis de estos valores:

✔️ ICA aumenta drásticamente el SNR, permitiendo obtener una voz clara con mínimo ruido residual.

✔️ PCA también mejora el SNR, pero al no estar diseñado específicamente para separar señales de audio, deja algo de ruido en la voz recuperada.

📌 Visualización del SNR en gráficos:

        # Graficar la mejora del SNR
        etiquetas = ['Antes de separación', 'Después de ICA', 'Después de PCA']
        valores = [5, 23, 17]  # Ejemplo de SNR en dB
        
        plt.figure(figsize=(8,5))
        plt.bar(etiquetas, valores, color=['red', 'green', 'blue'])
        plt.xlabel('Método')
        plt.ylabel('SNR (dB)')
        plt.title('Comparación de SNR antes y después de la separación de voces')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

📷 Evidencia de la Ejecución en Consola

Aquí se muestran los valores de SNR obtenidos al ejecutar los métodos de separación de voz:

![Imagen de WhatsApp 2025-03-01 a las 13 24 56_bbfc5c3d](https://github.com/user-attachments/assets/644d3fca-2211-4b80-ac9f-e5038a866a60)

Fig 7. SNR de Ica

![Imagen de WhatsApp 2025-03-01 a las 13 25 15_69f82c20](https://github.com/user-attachments/assets/7907cfa4-c7ef-4baf-a163-be2c4fda7d78)

Fig 8. SNR de PCA

3️⃣ Análisis Espectral de las Señales Separadas

📌 ¿Qué analizamos aquí?

Al calcular la Transformada Rápida de Fourier (FFT), podemos ver el contenido de frecuencia de las señales antes y después de la separación.

📌 Observaciones en los espectros:

- El espectro de la señal original muestra picos en diferentes frecuencias debido a la mezcla de voces.
- Después de ICA, el espectro de la voz de Laura es más limpio, eliminando frecuencias innecesarias.
- Después de PCA, el espectro de Andrea todavía tiene algunas interferencias del ruido original.

📄 Conclusión

✅ ICA es ideal cuando las señales son independientes entre sí.

✅ PCA es útil cuando las señales tienen cierto grado de correlación.

✅ La posición de los micrófonos y la calidad de la adquisición afectan significativamente los resultados.
