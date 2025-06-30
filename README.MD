# Proyecto de Análisis de Señales Fisiológicas


Desarrrollado durante el trabajo final de la asignatura Programación Digital Avanzada, este proyecto tiene como objetivo el diseño, desarrollo e implementación de una librería para el procesamiento, análisis y visualización de señales fisiológicas como EEG, EMG y ECG, utilizando Programación Orientada a Objetos (POO).

Autrores:
- Juan Acosta
- Florencia Sosa
- Joaquín Sánchez

Docentes de la asignatura:
- MSc. Bioing. BALDEZZARI Lucas
- Ing. Biomédica ÁLVAREZ Emiliano

---

## Clases principales

### `Info`
Almacena información relevante sobre el registro de datos:
- Nombres y tipos de canales
- Frecuencia de muestreo
- Canales malos
- Descripción del experimento
- Vecinos de canales (para EEG)

### `Anotaciones`
Gestión de eventos asociados a la señal:
- Carga y guarda anotaciones desde/hacia CSV
- Agrega, elimina y busca anotaciones
- Recorta anotaciones fuera de un rango temporal

### `RawSignal`
Clase base para señales fisiológicas:
- Almacena los datos de la señal y metadatos
- Métodos para recorte (`crop`), filtrado (`filter`), selección de canales (`pick`, `drop_channels`)
- Visualización básica (`plot`)
- Acceso flexible a subconjuntos de datos

### `EEGSignal` (hereda de `RawSignal`)
Herramientas específicas para señales EEG:
- Referenciación (promedio, canal, laplaciana)
- Espectro de frecuencias y transformada de Fourier
- Análisis tiempo-frecuencia
- Transformada de Hilbert

### `EMGSignal` (hereda de `RawSignal`)
Herramientas específicas para señales EMG:
- Detección de activaciones musculares por umbral
- Visualización de activaciones
- Espectrograma y transformada de Hilbert

### `ECGSignal` (hereda de `RawSignal`)
Herramientas específicas para señales ECG:
- Detección de picos R
- Cálculo de frecuencia cardíaca (HR)
- Visualización de picos R y análisis tiempo-frecuencia

---

## Requisitos

- Python 3.x
- numpy
- pandas
- matplotlib
- scipy
- copy

---

## Notas

- El código está pensado para ser flexible y extensible a otros tipos de señales fisiológicas.
- La documentación de cada clase y método está disponible en los docstrings del código fuente.
  
---

## Ejemplo de uso básico

```python
from TIF import Info, Anotaciones, EEGSignal, EMGSignal, ECGSignal

# Crear objeto Info
info = Info(
    experimenter="Investigador",
    subject_info={"edad": 25, "sexo": "M"},
    ch_names=["Fz", "Cz", "Pz"],
    ch_types=["eeg", "eeg", "eeg"],
    bads=[],
    description="Registro EEG de ejemplo",
    fm=512
)

# Crear objeto Anotaciones
anot = Anotaciones(
    onset=[1.0, 5.0],
    duration=[0.5, 0.5],
    description=["evento1", "evento2"]
)

# Crear y analizar una señal EEG
import numpy as np
data = np.random.randn(3, 5000)
eeg = EEGSignal(data, sfreq=512, info=info, anotaciones=anot)
eeg.plot(picks=["Fz"], start=0, duration=2)

# Crear y analizar una señal EMG
emg = EMGSignal(data[:1], sfreq=1000, info=info, umbral_microv=5)
emg.plot_activaciones(canal="Fz", start_time=0, end_time=2)

# Crear y analizar una señal ECG
ecg = ECGSignal(data[:1], sfreq=1000, info=info)
ecg.detectar_r_peaks(canales=["Fz"])
ecg.plot_r_peaks(picks=["Fz"], start=0, end=2)
```

---