import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.signal
from scipy.signal import find_peaks, spectrogram, hilbert

class Info:
    """ Clase para almacenar información acerca del registro de datos"""

    def __init__(self, experimenter:str, subject_info:dict, ch_names:list | str, ch_types:list | str, bads:list | str, description:str, fm:float = 512, neighbors:dict = None):
        """
            Genera un objeto Info()
            Args:
                experimenter : Nombre del experimentador.
                subject_info : Información adicional del sujeto.
                ch_names : Lista con los nombres de los canales.
                ch_types : Tipo de cada canal ('emg', 'eeg', 'ecg') o un único tipo para todos.
                bads : Lista de canales marcados como "malos ".
                fm : Frecuencia de muestreo en Hz (por defecto 512).
                description : Descripción del registro de datos.
                neighbors : Diccionario con los vecinos de cada canal (solo para señales de EEG).
        """
        
        if len(ch_names) != len(ch_types):
            raise ValueError ("La cantidad de canales y los tipos de canales deben tener la misma longuitud")
        
        self.data = {"Experimentador": experimenter,
                     "Sujeto":subject_info,
                     "Nombre canales": ch_names,
                     "Tipo canales": ch_types,
                     "Canales malos": bads,
                     "Descripción": description,
                     "Frecuencia muestreo": fm}  

        if (all(i.lower() == "eeg" for i in ch_types)):
            if neighbors is not None:
                self.data["Vecinos"] = neighbors      

    def __contains__(self, clave):
        """
        Permite verificar si una clave esta presente en el objeto.
            Args:
                clave : Nombre de la clave que se quiere verificar.
            Returns:
                True: La clave se encuentra en el objeto.
                False: La clave no se encuentra en el objeto.
        """
        
        if clave in self.data:
            return True
        else:
            return False
        
    def __getitem__(self, clave):
        """
        Permite acceder a elementos como un diccionario.
            Args:
                clave : Nombre de la clave a la que se quiere acceder.

            Returns:
                Devuelve el valor asociado a la clave ingrsada, o
                False: Cuando la clave ingresada no existe.
        """
        
        if self.__contains__(clave) == True:
            return self.data[clave]
        else:
            return False
        
    def __len__(self) -> int:
        """
        Devuelve la cantidad de elementos almacenados.
        """

        return len(self.data)
    
    def keys(self):
        """
        Devuelve las claves del objeto
        """

        return [clave for clave in self.data]
    
    def get(self, clave):
        """
        Obtiene solo el valor de una clave específica.
            Args:
                clave : Clave de la cual se quiere conocer su valor.

            Returns:
                Devuleve el valor de la clave si la misma existe, o False en caso de que no exista.
        """
        
        if self.__contains__(clave) == True:
            return self.data[clave]
        
        else:
            return False

    def item(self, elemento):
        """
        Devuelve los elementos como pares clave-valor de una clave en específica.
            Args:
                elemento : Elemento del que se quiere obtener la clave y su valor.
            Returns:
                tuple : Tupla que contiene la clave y el valor del elemento.
                False : Si no exite la clave en el diccionario.
        """
        
        valor = self.__getitem__(elemento)
        if valor == False:
            return False
        
        else:
            return (elemento, valor)
        
    def _check_channel(self, channel_name):
        """
        Verifica si un canal se encuentra entre los nombres de los canales.
            Args:
                channel_name : Canal a verificar si se encuentra en la lista de canales
            
            Returns:
                True : Si el canal existe.
                False : Si el canal no existe.
        """
        
        if channel_name in self.data["Nombre canales"]:
            return True
        
        else:
            return False
        
    def rename_channels(self, nombre_canal, nuevo_nombre):
        """
        Permite renombrar canales de forma segura.
            Args:
                nombre_canal : Nombre del canal que se quiere cambiar.
                nuevo_nombre : Nuevo nombre que va a tener el canal.

            Returs:
                True : Si se cambio el nombre del canal correctamente.
                False: Si no se pudo modificar el nombre del canal.
        """
        
        if self._check_channel(nombre_canal) == True:
            indice = self.data["Nombre canales"].index(nombre_canal)
            self.data["Nombre canales"][indice] = nuevo_nombre
            return True
        
        else:
            return False  
    
    def eliminar_elementos(self, key:str, elementos:str|list):
        """
        Elimina uno o varios elementos de la lista asociada a una clave específica.

        Args:
            clave : Clave del diccionario donde se eliminarán los elementos.
            elementos : Elemento o lista de elementos a eliminar.

        Raises:
            KeyError: Si la clave no existe en el diccionario.
            ValueError: Si uno o más elementos a eliminar no están presentes en la lista.
        """
        
        if self.__contains__(key) == False:
            raise KeyError(f"La clave '{key}' no existe en Info.")
        
        for elemento in elementos:
            if elemento not in self.data[key]:
                raise ValueError(f"El elemento '{elemento}' no se encuentra en la lista asociada a '{key}'.")
            self.data[key].remove(elemento)         

class Anotaciones:
    """
    Almacena y gestiona información relacionada con eventos en registros fisiológicos. 
    Permite la adición, eliminación y modificación de eventos.
    """
    
    def __init__(self, onset:np.ndarray=None, duration:np.ndarray=None, description=None, file=None):
        """
        Inicializa la clase con los datos de las anotaciones de forma manual o mediante un archivo csv.
        
        Args:
            onset : Inicio del evento en segundos.
            duration : Duración del evento en segundos.
            description : Descripción del evento.
            
            file : Nombre del archivo csv con las atonaciones.
                   Se espera una estructura de columnas [Inicio, Duracion, Descripcion].
        """
        
        self.onset = onset
        self.duration = duration
        self.description = description

        if file is not None:
            self.anotaciones = self.load(archivo=file)
        
        elif onset and duration and description:
            if len(onset) != len(duration):
                raise ValueError ("Onset y duration deben tener la misma cantidad de elementos")
            self.anotaciones = pd.DataFrame({"Inicio": self.onset, "Duracion": self.duration, "Descripcion": self.description})
        
        else:
            raise ValueError ("Debe ingresar las anotaciones manualmente o cargarlas a partir de un archivo")

    def get_annotations(self):
        """
        Devuelve una DataFrame con todas las anotaciones que recibe el constructor.
        """

        return self.anotaciones
    
    def add(self, anotacion:list|tuple):
        """
        Agrega una nueva anotación
            Args:
                anotacion : Nueva anotación a agregar (se espera la forma [Inicio, Duracion, Descripcion])
        """

        if len(anotacion) != 3:
            return False 
        else:
            self.anotaciones.loc[len(self.anotaciones)] = anotacion  # Agrega un fila
            return True 
    
    def remove(self, anotacion_eliminar):
        """
        Elimina una anotación específica
            Args:
                anotacion_eliminar : Anotación que se quiere eliminar
            Returns:
                True : Selimino correctamente la anotación
                False : No se pudo eliminar la anotación
        """
        
        if len(anotacion_eliminar) != 3:
            return False
        
        else:
            eliminar = (self.anotaciones["Inicio"] == anotacion_eliminar[0]) & (self.anotaciones["Duracion"] == anotacion_eliminar[1]) & (self.anotaciones["Descripcion"] == anotacion_eliminar[2])
            indice = self.anotaciones[eliminar].index
            self.anotaciones = self.anotaciones.drop(indice)
            return True
    
    def recorte(self, tmin:float, tmax:float):
        """
        Obtiene un df con las anotaciones a aliminar en base a la columna "Inicio".
        Usa el metodo remove() para eliminar ese las filas de ese df de las anotaciones.

        Args:
            tmin : Tiempo en segundos. Los "Inicio" menores a este son eliminados.
            tmax : Tiempo en segundos. Los "Inicio" mayores a este son eliminados.        
        """
        anotaciones = self.get_annotations()
        minimo = anotaciones["Inicio"] < tmin
        maximo = anotaciones["Inicio"] > tmax
        eliminar = anotaciones[minimo | maximo]

        for _, row in eliminar.iterrows():
            self.remove((row["Inicio"], row["Duracion"], row["Descripcion"]))

    def find(self, buscar_anotacion):
        """
        Busca y devuelve una anotación específica
            Args:
                buscar_anotacion : Anotación que se quiere buscar entre los datos 
            Returs:
                Devuelve la anotación o False si la longuitud de la anotación no coincide con la estructura de los datos
        """
        
        if len(buscar_anotacion) != 3:
            return False
        else:
            buscar = (self.anotaciones["Inicio"] == buscar_anotacion[0]) & (self.anotaciones["Duracion"] == buscar_anotacion[1]) & (self.anotaciones["Descripcion"] == buscar_anotacion[2])
            return self.anotaciones[buscar]

    def save(self, nombre):
        """
        Guarda las anotaciones en un archivo .csv
            Args:
                nombre : Nombre con el que se guardara el archivo
        """

        return self.anotaciones.to_csv(f"{nombre}.csv")
    
    def load(self, archivo):
        """
        Carga las anotaciones desde un archivo .csv
            Args:
                archivo : Nombre del archivo que se quiere cargar
            Devuelve el dataframe con los datos del archivo csv.
        """
        
        anotacion = pd.read_csv(archivo)
        return anotacion
    
class RawSignal:
    """
    Clase para manejar señales fisiológicas en formato NumPy.
    Este constructor permite inicializar el objeto 'RawSignal' a partir de un array de datos ,
    con información adicional de los canales y el índice de la primera muestra."""
    
    def __init__(self, data:np.ndarray, sfreq:float, info:Info=None, anotaciones:Anotaciones=None, first_samp:int=0):
        """
        Inicializa una instancia de la clase RawSignal.
        
        Args:
            data : Array de datos con forma '(n_canales , n_muestras)'.
            sfreq : Frecuencia de muestreo de la señal en Hz.
            info : Por defecto es None. Información adicional sobre la señal. El diccionario contiene info relevante de la señal
            anotaciones : Objeto de tipo Anotaciones que almacena eventos asociados a la señal y al experimento.
            first_samp : Indica el tiempo de inicio del segmento, pero no recorta los datos. (Por defecto es 0)
            
        Raises:
            ValueError : Si el array 'data' no tiene la forma '(n_canales , n_muestras)'.
            ValueError : Si el índice 'first_samp' está fuera del rango de la señal. """
        
        if not isinstance(data, np.ndarray):     # Que data sea un array de NumPy
            raise ValueError("El parámetro 'data' debe ser un array de NumPy (np.ndarray).")

        if data.ndim != 2:                       # Que data tenga dos dimensiones 
            raise ValueError("El array 'data' debe tener dos dimensiones: (n_canales, n_muestras).")

        n_muestras = data.shape[1]               # Número de muestras es la segunda dimensión
        if ((first_samp < 0) or (first_samp > n_muestras / sfreq)):   # Que first_samp sea un entero positivo y menor que la duración total de la señal
            raise ValueError("El índice 'first_samp' está fuera del rango de duración total de la señal")

        if info is not None and info["Nombre canales"] is not None:
            if len(info["Nombre canales"]) != data.shape[0]:
                raise ValueError(f"La cantidad de canales de la señal ({data.shape[0]}) no coincide con la cantidad ingresada en el objeto Info ({len(info['Nombre canales'])})")
        
        self.data = data
        self.sfreq = sfreq
        self.info = info
        self.first_samp = first_samp

        if anotaciones is not None:
            self.validar_anotaciones(anotaciones)    # Validación estructurada
            self.anotaciones = anotaciones
    
    def validar_anotaciones(self, anotaciones:Anotaciones):
        """
        Verifica que las anotaciones estén dentro del rango temporal de la señal.

        Args:
            anotaciones :  Objeto de la clase 'Anotaciones' que contiene la información de los eventos.

        Raises:
            TypeError : Si el parámetro 'anotaciones' no es una instancia de la clase 'Anotaciones'.
            ValueError : Si alguna anotación está fuera del rango temporal de la señal. """

        if not isinstance(anotaciones, Anotaciones):
            raise TypeError("El parámetro 'anotaciones' debe ser una instancia de la clase 'Anotaciones'.")

        df = anotaciones.get_annotations()                # Método de la clase Anotaciones

        duracion_total = (self.data.shape[1] / self.sfreq) + self.first_samp  # Duración total de la señal

        for i, fila in df.iterrows():                     # Validar que todas las anotaciones estén dentro del rango
            inicio = fila["Inicio"]
            duracion = fila["Duracion"]
            if not (0 <= inicio <= duracion_total):
                raise ValueError(f"Anotación con inicio en {inicio}s fuera del rango (0 - {duracion_total:.2f}s).")
            if inicio + duracion > duracion_total:
                raise ValueError(f"Anotación desde {inicio}s con duración {duracion}s excede el final de la señal.")

        self.anotaciones = anotaciones
    
    def set_anotaciones(self, anotaciones: Anotaciones):
        """
        Asocia o reemplaza el objeto 'Anotaciones' de la señal, validando que esté en rango.

        Args:
            anotaciones : Objeto de la clase 'Anotaciones' que contiene la información de los eventos.

        Raises:
            TypeError : Si el parámetro no es de tipo 'Anotaciones'.
            ValueError : Si alguna anotación está fuera del rango temporal de la señal.
        """
        self.validar_anotaciones(anotaciones)
        self.anotaciones = anotaciones
        
    def get_data(self, picks=None, start:float|int=0, stop:float|int=0, reject:float=None, times:bool=False):
        """
        Obtiene muestras de la señal en un rango dado.
        Args:
            picks (str o array_like) : Canales o índices a extraer. Si es 'None', se seleccionan todos los canales.
            start : Tiempo inicial (en segundos) para extraer muestras (por defecto 0).
            stop : Tiempo final (en segundos) para extraer muestras (por defecto 0, que significa hasta el final de la señal).
            reject : Valor pico a pico de umbral para rechazar canales. Si una muestra supera este umbral, el canal se descarta (por defecto 'None').
            times : Si es 'True', se retorna también el vector de tiempos asociado a las muestras.

        Returns:
            np.ndarray : Matriz con los datos seleccionados (n_canales x n_muestras).
            np.ndarray (opcional) : Vector de tiempos (solo si 'times=True').

        Raises
            ValueError : Si los índices seleccionados están fuera de rango. """
        
        n_canales, n_muestras = self.data.shape                       # Número de canales y muestras  
    
        start_idx = int(start * self.sfreq)                           # Convertir start de segundos a número de muestra
        stop_idx = int(stop * self.sfreq) if stop > 0 else n_muestras # Si stop es mayor que 0, convertir a indice de muestra, sino, tomar hasta el final

        if not (0 <= start_idx < stop_idx <= n_muestras):             # Si no se cumple que start_idx es mayor o igual a 0 y menor que stop_idx y ademas stop_idx es menor o igual al numero de muestras, lleva al error
            raise ValueError("Índices de tiempo fuera de rango.")

        # Seleccionar canales
        if picks is None:                             # Si picks queda por defecto, seleccionar todos los canales
            canales_idx = np.arange(n_canales)

        elif isinstance(picks, (list, np.ndarray)):
            if all(isinstance(pick, int) for pick in picks):   # Verificar que los canales ingresados sean todos enteros
                if np.isin(picks, np.arange(n_canales)).all(): # Verificar si los canales ingresados existen
                    canales_idx = np.array(picks)
                else:
                    raise ValueError("Error: algunos canales ingresados no existen")
            
            elif all(isinstance(pick, str) for pick in picks):   # Verificar que los canales ingresados sean todos strings
                if self.info is None:
                    raise ValueError("Debe ingresar el objeto Info")
                
                nombres = self.info.get("Nombre canales")       # Aplicamos el médoto de Info para obtener los canales del diccionario
                try:
                    canales_idx = [nombres.index(canal) for canal in picks] # Si "canal" esta en los canels pasados desde Info, obtiene el indice y se agrega a canales_idx
                except ValueError:
                    raise ValueError(f"Error: uno o más canales no se encuentran en los canales Ingresados desde el objeto Info")
            else:
                raise ValueError("Error: los canales ingresados no tienen el mismo formato")

        else:
            raise ValueError("Formato de 'picks' inválido. Deben ser strings o enteros.")

        datos = self.data[np.array(canales_idx), start_idx : stop_idx]  # Extraer datos de los canales

        if reject is not None:                 # Aplicar umbral de rechazo si se especifica
            pico_pico = np.ptp(datos, axis=1)  # Toma el max y min del canal y los resta, obteniendo el valor pico_pico de cada canal (axis=1 filas)
            filtro = pico_pico < reject        # Filtro para quedarme solo con los valores pico_pico menores a reject
            datos = datos[filtro]

        if times is True:
            tiempo_vector = (np.arange(start_idx, stop_idx) / self.sfreq) + self.first_samp
            
            return datos, tiempo_vector

        return datos
        
    def crop(self, tmin:int|float=0.0, tmax:int|float=None) -> "RawSignal":
        """
        Obtiene un trozo de RawSignal. Limita los datos dentro de RawSignal
        para obtener un nuevo objeto RawSignal pero con una cantidad de muestras recortadas.
        
        Args:
            tmin : Tiempo inicial en segundos para iniciar el recorte (por defecto es 0.0).
            tmax : Tiempo final en segundos para finalizar el recorte (por defecto es None).
        
        Returns:
            RawSignal : Nueva instancia de 'RawSignal' que contiene el segmento temporal recortado.
        
        Raises
            Value Error : Si los tiempos 'tmin' o 'tmax' están fuera del rango de la señal. """
        
        n_canales, n_muestras = self.data.shape
        duracion_total = n_muestras / self.sfreq

        anotaciones_copia = copy.deepcopy(self.anotaciones)
        anotaciones_copia.recorte(tmin=tmin, tmax=tmax)               # Elimina las anotaciones que superan el tmax del recorte

        # Validaciones
        if not isinstance(tmin, (int, float)) or tmin < 0:             # Si tmin no es entero, flotante o positivo
            raise ValueError("'tmin' debe ser un número positivo.")
        
        if tmax is not None:
            if not isinstance(tmax, (int, float)) or tmax <= tmin:     # Si tmax no es un número o es menor a tmin
                raise ValueError("'tmax' debe ser mayor que 'tmin'.")
            elif tmax > duracion_total:                                # Si tmax es mayor que la duracion total de la señal
                raise ValueError(f"'tmax' está fuera del rango de la señal ({duracion_total:.2f} s).")

        if tmin > duracion_total:                                      #Chequeo si tmin es mayor que la duracion total de la señal 
            raise ValueError(f"'tmin' está fuera del rango de la señal ({duracion_total:.2f} s).")

        start_idx = int(tmin * self.sfreq)                             # Convertir a índices
        end_idx = int(tmax * self.sfreq) if tmax is not None else n_muestras

        datos_crop = self.data[:, start_idx:end_idx]                   # De todos los canales (:), extraer el segmento de la señal (start_idx:end_idx)

        return RawSignal(data=datos_crop, sfreq=self.sfreq, info=self.info, anotaciones=anotaciones_copia, first_samp= self.first_samp + tmin) # El segmento recortado empieza en 0 (first_samp)
    
    def drop_channels(self, ch_names:list|np.ndarray) -> "RawSignal":
        """ 
        Elimina uno o más canales a partir de ch_names.

        Args:
            ch_names : Nombres de canales a eliminar

        Returns:
            Objeto RawSignal """

        # Validaciones
        if self.info is None or self.info.get("Nombre canales") is None:
            raise ValueError("No se puede eliminar canales sin el objeto Info con los nombres de los canales")
        
        canales_actuales = self.info["Nombre canales"]
        datos_actuales = self.data
        info_copia = copy.deepcopy(self.info)

        if all(isinstance(ch, str) for ch in ch_names):    # Si todos los canales son str
            try:
                indices_eliminar = [canales_actuales.index(nombre) for nombre in ch_names]
            except ValueError as e:
                raise ValueError(f"Uno o más nombres de canal no se encuentran en la lista: {e}")
            
        elif all(isinstance(ch, int) for ch in ch_names):  # Si todos los canales son int
            indices_eliminar = ch_names
        
        else:
            raise ValueError("Todos los elementos de 'ch_names' deben ser del mismo tipo (str o int).")
        
        datos_filtrados = np.delete(datos_actuales, indices_eliminar, axis=0)   # Elimina los indices (canales) de la señal

        info_copia.eliminar_elementos(key="Nombre canales", elementos=ch_names)

        return RawSignal(data=datos_filtrados, sfreq=self.sfreq, info=info_copia, anotaciones=self.anotaciones, first_samp=self.first_samp)

    def describe(self, archivo_salida):
        """
        Genera un DataFrame con estadísticas descriptivas para cada canal de la señal.

        Para cada canal se obtiene:
        - Nombre: Nombre del canal.
        - Tipo: Tipo de canal (eeg, ecg, emg).
        - Min: Valor mínimo del canal.
        - Q1: Primer cuartil (percentil 25%).
        - Mediana: Mediana (percentil 50%).
        - Q3: Tercer cuartil (percentil 75%).
        - Max: Valor máximo del canal.

        Args:
            archivo_salida : Nombre del archivo csv que se genera

        Returns:
            pd.DataFrame : Tabla con una fila por estadístico y una columna por canal. """
        
        n_canales = 0

        if self.info is not None:
            if "Nombre canales" in self.info:
                nombres = self.info.get("Nombre canales")
                n_canales = len(nombres)

            if "Tipo canales" in self.info:
                tipos = self.info.get("Tipo canales")
        
        if self.info is None: 
            n_canales = self.data.shape[0]
            nombres = [f"Canal_{i+1}" for i in range(n_canales)]
            tipos = ["Desconocido"] * n_canales
        
        tabla = {}

        for i in nombres:
            indice = nombres.index(i)
            canal = self.data[indice]
            resumen = {
                "Nombre canal": i,
                "Tipo canal": tipos[indice],
                "Min": np.min(canal),
                "Q1": np.percentile(canal, 25),
                "Mediana": np.median(canal),
                "Q3": np.percentile(canal, 75),
                "Max": np.max(canal) }
            tabla[i] = resumen

        datos = pd.DataFrame(tabla)
        datos.to_csv(archivo_salida, index=False)

        return datos

    def filter(self, l_freq:float, h_freq:float, notch_freq:float = 50.0, order:int = 4) -> "RawSignal":
        """
        Aplica un filtro pasabanda (Butterworth) y un filtro notch a la señal fisiológica.
        El filtro notch permite eliminar una frecuencia fija (por defecto 50 Hz).
        Filtro pasabanda Butterworth que permite mantener solo las frecuencias entre l_freq y h_freq.

        Args:
            l_freq : Frecuencia de corte baja del filtro pasabanda (en Hz).
            h_freq : Frecuencia de corte alta del filtro pasabanda (en Hz).
            notch_freq : Frecuencia del filtro notch para eliminar ruido (por defecto 50 Hz).
            order : Orden del filtro Butterworth (por defecto 4).

        Returns:
            RawSignal : Nueva instancia de RawSignal con los datos filtrados.

        Raises:
            ValueError : Si las frecuencias de corte no son válidas. """
        
        if l_freq >= h_freq:
            raise ValueError("La frecuencia de corte mínima debe ser menor que la de corte máxima.")
        
        if notch_freq <= 0 or l_freq <= 0 or h_freq <= 0:
            raise ValueError("Las frecuencias deben ser mayores que cero.")

        datos_filtrados = np.empty_like(self.data)   # Crear un nuevo array para almacenar los datos filtrados. empty_like: Return a new array with the same shape and type as a given array.

        # Diseño del filtro notch
        Q = 30.0  # Factor de calidad del notch (más alto = más estrecho)
        b_notch, a_notch = scipy.signal.iirnotch(notch_freq, Q, self.sfreq)

        # Diseño del filtro pasabanda Butterworth
        b_band, a_band = scipy.signal.butter(order, [l_freq, h_freq], btype='band', fs=self.sfreq)

        for i in range(self.data.shape[0]):                                           # Aplica los filtros a cada canal de la señal
            canal = self.data[i]                                                      # Extraer un canal
            canal_filtrado = scipy.signal.filtfilt(b_notch, a_notch, canal)           # Aplica notch
            canal_filtrado = scipy.signal.filtfilt(b_band, a_band, canal_filtrado)    # Aplica pasabanda
            datos_filtrados[i] = canal_filtrado                                       # Guarda resultado

        # return RawSignal(data=datos_filtrados, sfreq=self.sfreq, info=self.info, anotaciones=self.anotaciones, first_samp=self.first_samp)
        return RawSignal(data=datos_filtrados, sfreq=self.sfreq, info=self.info, anotaciones=self.anotaciones, first_samp=self.first_samp)

    def pick(self, canales=None, slice:list=None) -> "RawSignal":
        """
        Retorna un subset de canales seleccionados.

        Args:
            canales : Lista con los canales para armar el subset. Por defecto es None.
                Puede ser:
                    - list[str] : lista de nombres de canales
                    - list[int] : lista de índices de canales
            
            slice : Lista con los extremos para hacer slicing. Por defecto es None.
            
        Returns:
            RawSignal : Nueva instancia con los canales seleccionados.

        Raises: ValueError:
                    Si el canal especificado no existe.
                    Si el índice está fuera del rango de canales. """
        

        if canales is not None and slice is None:
            indices = canales

        elif canales is None and slice is not None:
            indices = list(range(slice[0], slice[1]))

        else:
            raise ValueError ("Error: debe ingresar canales o slice. No se permiten los dos al mismo tiempo")
        
        if self.info is not None:
            subset = self.get_data(picks=indices)
            info_copia = copy.deepcopy(self.info)
            canales_originales = info_copia.get("Nombre canales")           

            canales_eliminar = []
            for canal in canales_originales:
                if canal not in indices:
                    canales_eliminar.append(canal)
            info_copia.eliminar_elementos(key="Nombre canales", elementos= canales_eliminar)
        
        else:
            subset = self.get_data(picks=indices)
            info_copia = None

        return RawSignal(data=subset, sfreq=self.sfreq, info=info_copia, anotaciones=self.anotaciones, first_samp=self.first_samp)    

    def plot(self, picks=None, start:float=0, duration:float=None, show_anotaciones:bool=True):

        """
        Grafica un segmento de la señal fisiológica.

        Args:
            Canal o canales a visualizar. Puede ser:
                - list[str]: lista de nombres de canales,
                - list[int]: lista de índices de canales.
                - Si es None, se grafican todos los canales.

            start : Tiempo inicial (en segundos) desde donde comenzar la visualización (por defecto 0.0).
            duration : Duración del segmento de señal a mostrar en segundos, por defecto None, gráfica toda la señal).

            show_anotaciones (por defecto True): 
                    True : Se muestran las anotaciones sobre la señal.
                    False : No se muestran las anotaciones.
        """  
        
        if duration is None:
            n_muestras = self.data.shape[1]
            duration = n_muestras / self.sfreq

        intervalo_inicio = start
        intervalo_fin = start + duration

        canales, tiempo = self.get_data(picks=picks, start=intervalo_inicio, stop=intervalo_fin, times=True)
        n_canales = canales.shape[0]

        fig, axes = plt.subplots(n_canales, 1, figsize=(10, 3 * n_canales), sharex=True)
        if n_canales == 1:
            axes = [axes]

        for i in range(n_canales):
            axes[i].plot(tiempo, canales[i])
            if self.info is not None and "Nombre canales" in self.info: # Nombre real del canal si hay Info y picks
                if picks is None:
                    nombre = self.info["Nombre canales"][i]
                else:
                    # Obtener los índices reales de los canales seleccionados
                    if all(isinstance(pick, int) for pick in picks):
                        idx = picks[i]
                        nombre = self.info["Nombre canales"][idx]
                    elif all(isinstance(p, str) for p in picks):
                        nombre = picks[i]
                    else:
                        nombre = f"Canal {i}"
                axes[i].set_ylabel("Amplitud (uV)")
                axes[i].set_title(f"Canal: {nombre}")
            else:
                axes[i].set_ylabel("Amplitud (uV)")
                axes[i].set_title(f"Canal {i}")
            axes[i].set_xlabel("Tiempo [s]")
            axes[i].grid(True)

        # El eje x debe mostrar los tiempos absolutos (+ first_samp)
        x_min = self.first_samp + intervalo_inicio
        x_max = self.first_samp + intervalo_fin

        if show_anotaciones and self.anotaciones is not None:
            df = self.anotaciones.get_annotations()
            eventos = df["Descripcion"].unique()
            colormap = plt.colormaps.get_cmap('Set1')  # 'rainbow', 'Set3', cool, plasma

            # Asignar un color único a cada tipo de evento
            colores_eventos = {evento: colormap(i / len(eventos)) for i, evento in enumerate(eventos)}

            for evento, color in colores_eventos.items():
                df_filtrado = df[df["Descripcion"] == evento]

                for _, fila in df_filtrado.iterrows():
                    inicio = fila["Inicio"]
                    duracion = fila["Duracion"]
                    fin = inicio + duracion
                    if (inicio < x_max) and (fin > x_min):
                        sombra_inicio = max(inicio, x_min)
                        sombra_fin = min(fin, x_max)
                        for ax in axes:
                            ax.axvspan(sombra_inicio, sombra_fin, color=color, alpha=0.3)

            handles = [Patch(color=color, label=evento) for evento, color in colores_eventos.items()]
            fig.legend(handles=handles, loc='upper right')
            
        plt.xlim(x_min, x_max)
        plt.tight_layout()
        plt.show()
    
    def __getitem__(self, item:str|list|tuple):
        """
        Permite acceder a subconjuntos de la señal utilizando indexación personalizada.

        El método soporta múltiples formas de indexación:
        
        - Si se pasa un string (str), devuelve todos los datos del canal correspondiente a ese nombre.
        - Si se pasa una lista de strings (list[str]), devuelve los datos de los canales correspondientes.
        - Si se pasa un slice, devuelve todas las muestras del intervalo especificado para todos los canales.
        - Si se pasa una tupla ([list[str], slice]), devuelve las muestras indicadas por el slice para los canales especificados por nombre.

        Args:
            item (str | list[str] | slice | tuple): 
                Clave de acceso para extraer subconjuntos de la señal.
                - str: nombre del canal.
                - list[str]: lista de nombres de canales.
                - slice: intervalo de muestras para todos los canales.
                - tuple: combinación (canales, slice).

        Returns:
            np.ndarray: Submatriz de datos (n_canales x n_muestras) correspondiente a la selección realizada.
        """

        if self.info is None:
            raise ValueError("No se ha proporcionado el objeto Info.")
        
        nombre_canales = self.info.get("Nombre canales")   # Canales que se pasan desde el objeto Info

        if isinstance(item, str):
            canales_idx = nombre_canales.index(item)             
            return self.data[canales_idx,:]

        elif isinstance(item, list):
            canales_idx = [nombre_canales.index(canal) for canal in item]
            return self.data[canales_idx,:]            

        elif isinstance(item, slice):
            return self.data[:,item.start:item.stop:item.step]
        
        elif isinstance(item, tuple):
            canales, slicing = item[0], item[1]
            canales_idx = [nombre_canales.index(canal) for canal in canales]
            return self.data[canales_idx,slicing.start:slicing.stop:slicing.step]

class EEGSignal(RawSignal):
    """
    Clase para representar y analizar una señal de Electroencefalografía (EEG).

    Proporciona herramientas básicas para manipular y visualizar señales EEG, aplica filtrado espacial,
    y realiza análisis espectrales y temporales.

    """

    def __init__(self, data: np.ndarray, sfreq: float, info: Info = None, anotaciones: Anotaciones = None, first_samp: int = 0,
                referencia: str = "promedio", canal: str = None):
        """

        Extiende la clase base RawSignal e incorpora herramientas específicas para el análisis EEG,

        Args:
            data : Matriz 2D con forma (n_canales, n_muestras) que contiene la señal EEG cruda.

            sfreq : Frecuencia de muestreo de la señal (en Hz).

            info : Objeto con datos de la señal, como nombres de canales, tipo de canales, frecuencia de muestreo,
                   y otros datos del sujeto o experimento.

            anotaciones : Objeto que almacena las anotaciones temporales asociadas a eventos.

            first_samp : first_samp : Indica el tiempo de inicio del segmento, pero no recorta los datos. (Por defecto es 0)

            referencia : Tipo de referencia a aplicar a la señal EEG. Puede ser 'canal', 'promedio' o 'laplaciano'.
                         Por defecto se utiliza 'promedio'.

            canal : Nombre del canal de referencia si se elige referencia tipo 'canal'.

        Atributos:
            self.data_ref : Señal EEG con la referencia aplicada (se asigna desde set_reference()).

            self.referencia : Tipo de referencia actualmente aplicada.

        Raises:
            ValueError : Si el tipo de referencia especificado no es válido.

        """
        super().__init__(data, sfreq, info, anotaciones, first_samp)

        self.referencia = referencia
        self.canal_ref = canal
        self.set_reference(reference=referencia, channel=canal)

    def set_reference(self, reference: str, channel: str|int = None):
        """
        Cambia la referencia de la señal EEG. Opciones disponibles: 'canal', 'promedio' , 'laplaciano'.".

        """

        self.referencia = reference

        if self.referencia == "canal":
            self.data_ref = self._aplicar_referencia_a_canal(channel=channel)
        
        elif self.referencia == "promedio":
            self.data_ref = self._aplicar_referencia_promedio()

        elif self.referencia == "laplaciano":
            self._aplicar_referencia_laplaciana()

        else:
            raise ValueError("Tipo de referencia no reconocida. Debe usar 'canal', 'promedio' o 'laplaciano'.")
        
        return self.data_ref

    def _aplicar_referencia_a_canal(self, channel: str|int):
        """
        Cambia la referencia de todos los canales a un canal específico.

        Este método resta, muestra a muestra, la señal de un canal de referencia seleccionado al resto de los canales EEG. 

        """

        canales_disponibles = self.info.get("Nombre canales")

        if channel in canales_disponibles:
            channel_values = self.get_data([channel])
            reference = self.data - channel_values

        else:
            raise ValueError (f"El canal {channel} no existe. Canales disponibles: {canales_disponibles}")
        
        return reference

    def _aplicar_referencia_promedio(self):
        """
        Aplica una referencia promedio a la señal EEG.

        Este método calcula el promedio de todos los canales en cada muestra de tiempo
        y lo resta a cada canal individual, generando así una señal referenciada al promedio.
        """

        channel_values = self.get_data() 
        mean_reference = np.mean(channel_values, axis=0)
        reference = self.data - mean_reference
        return reference

    def _aplicar_referencia_laplaciana(self):
        """
        Aplica un filtro espacial Laplaciano para resaltar la actividad local de cada canal.

        Return:
            Señal con el filtro aplicado a todos los canales.
        """

        neighbors = self.info.get("Vecinos")
        canales = self.info.get("Nombre canales")

        if neighbors is None:
            raise ValueError("Faltan los campos 'Vecinos' en el objeto Info.")

        laplacian_data = np.zeros_like(self.data)

        for i, canal in enumerate(canales):
            vecinos = neighbors.get(canal, [])
            
            if not vecinos:
                laplacian_data[i, :] = self.data[i, :]               # Si no tiene vecinos, lo deja igual
                continue

            indices_vecinos = [canales.index(v) for v in vecinos]

            promedio = np.mean(self.data[indices_vecinos, :], axis=0)
            laplacian_data[i, :] = self.data[i, :] - promedio

        self.data_ref = laplacian_data
        return self.data_ref

    def espectro_frecuencias(self, picks : list, plot : bool = False, fmin : float = None, fmax :float = None):
        """
        Calcula el espectro de potencia de uno o más canales.

        Args:
            picks : Lista de canales a calcular la FFT. Si es None se calcula la de todos los canales
            plot : 
                - True : Muestra el gráfico del espectro de frecuencias
                - False : Por defecto, no muestra el gráfico
            fmin : Si se decide mostrar el gráfico, define desde que frecuencia mostrarlo (eje x del gráfico)
            fmax : Si se decide mostrar el gráfico, define hasta que frecuencia máxima mostrar (eje x del gráfico)
            
        Returns:
            frequencies : array de frecuencias (Hz)
            spectrum : array (n_canales, n_frecuencias)
        """

        n_canales, n_muestras = self.data_ref.shape

        if picks is None:                                    # Si picks queda por defecto, seleccionar todos los canales
            canales_idx = np.arange(n_canales)
        
        else:
            channels = self.info.get("Nombre canales")
            try:
                canales_idx = [channels.index(canal) for canal in picks]
            except:
                raise ValueError(f"Error: uno o más canales no se encuentran en los canales Ingresados desde el objeto Info")
        
        datos = self.data_ref[canales_idx]
        datos = np.atleast_2d(datos)                               # asegura que siempre sea 2D

        frequencies, spectrum = scipy.signal.welch(datos, fs=self.sfreq, nperseg=1024)
        spectrum = 10 * np.log10(spectrum)                         # dB re: 1 μV²/Hz

        fr_max_ejex = frequencies.max()                            # valor maximo de frecuencia
        fr_min_ejex = frequencies.min()

        if fmin:
            if isinstance (fmin, (int, float)):
                if fmin >= fr_min_ejex and fmin < fr_max_ejex:
                    fr_min_ejex = fmin
                else:
                    raise ValueError(f"La frecuencia mínima '{fmin}' debe estar entre {fr_min_ejex:.2f} y {fr_max_ejex:.2f} Hz.")
            else:
                raise ValueError (f"La frecuencia mínima ingresada ´{fmin}´ no es valida")
      
        if fmax:
            if isinstance (fmax, (int, float)):
                if fmax <= fr_max_ejex and fmax > fr_min_ejex:
                    fr_max_ejex = fmax
                else:
                    raise ValueError (f"La frecuencia máxima '{fmax}' debe ser mayor a la frecuencia minima establecida {fr_min_ejex:.2f} Hz.")
            else:
                raise ValueError (f"La frecuencia máxima ingresada '{fmax}' no es valida")

        if plot:
            self._plot_spectrum(frecuencias=frequencies, espectro=spectrum, indices=canales_idx, fmin=fr_min_ejex, fmax=fr_max_ejex)
    
        return frequencies, spectrum
    
    def fourier_spectrum(self, picks : list, plot : bool = False, fmin : float = None, fmax :float = None):
        """
        Calcula la FFT (transformada de Fourier) de uno o más canales.

        Args:
            picks : Lista de canales a calcular la FFT. Si es None se calcula la de todos los canales
            plot : 
                - True : Muestra el gráfico del espectro de frecuencias
                - False : Por defecto, no muestra el gráfico
            fmin : Si se decide mostrar el gráfico, define desde que frecuencia mostrarlo (eje x del gráfico)
            fmax : Si se decide mostrar el gráfico, define hasta que frecuencia máxima mostrar (eje x del gráfico)
            
        Returns:
            freqs : array de frecuencias (Hz)
            espectro : array (n_canales, n_frecuencias)
        """

        n_canales, n_muestras = self.data_ref.shape

        if picks is None:                                    # Si picks queda por defecto, seleccionar todos los canales
            canales_idx = np.arange(n_canales)
        
        else:
            channels = self.info.get("Nombre canales")
            try:
                canales_idx = [channels.index(canal) for canal in picks]
            except:
                raise ValueError(f"Error: uno o más canales no se encuentran en los canales Ingresados desde el objeto Info")
        
        datos = self.data_ref[canales_idx]
        datos = np.atleast_2d(datos)                         # asegura que siempre sea 2D

        fft_data = np.fft.rfft(datos, axis=1)                # calcula la Transformada Rápida de Fourier (FFT) real para cada canal (por filas).
        freqs = np.fft.rfftfreq(n_muestras, d=1/self.sfreq)  # calcula el vector de frecuencias correspondientes a la FFT.
        espectro = np.abs(fft_data)

        fr_max_ejex = freqs.max()                            # valor maximo de frecuencia
        fr_min_ejex = freqs.min()

        espectro = 10 * np.log10(espectro)       # dB re: 1 μV²/Hz

        if fmin:
            if isinstance (fmin, (int, float)):
                if fmin >= fr_min_ejex and fmin < fr_max_ejex:
                    fr_min_ejex = fmin
                else:
                    raise ValueError(f"La frecuencia mínima '{fmin}' debe estar entre {fr_min_ejex:.2f} y {fr_max_ejex:.2f} Hz.")
            else:
                raise ValueError (f"La frecuencia mínima ingresada '{fmin}' no es valida")
      
        if fmax:
            if isinstance (fmax, (int, float)):
                if fmax <= fr_max_ejex and fmax > fr_min_ejex:
                    fr_max_ejex = fmax
                else:
                    raise ValueError (f"La frecuencia máxima '{fmax}' debe ser mayor a la frecuencia minima establecida {fr_min_ejex:.2f} Hz.")
            else:
                raise ValueError (f"La frecuencia máxima ingresada '{fmax}' no es valida")

        if plot:
            self._plot_spectrum(frecuencias=freqs, espectro=espectro, indices=canales_idx, fmin=fr_min_ejex, fmax=fr_max_ejex)
    
        return freqs, espectro
    
    def _plot_spectrum(self, frecuencias:np.ndarray, espectro:np.ndarray, indices, fmin, fmax):
        """
        Genera un gráfico individual del espectro de Fourier para cada canal especificado.

        Args:
            frecuencias : Array unidimensional con las frecuencias correspondientes a los valores del espectro (en Hz).
            espectro : Matriz 2D de amplitudes espectrales con forma (n_canales, n_frecuencias), resultado de la transformada de Fourier.
            indices : Lista de índices de los canales que fueron seleccionados para análisis.
            fmin : Frecuencia mínima a mostrar.
            fmax : Frecuencia máxima a mostrar.

        """

        canales = self.info.get("Nombre canales")

        mask = (frecuencias >= fmin) & (frecuencias <= fmax)

        for i, idx in enumerate(indices):
            plt.figure(figsize=(8, 4))
            plt.plot(frecuencias[mask], espectro[i][mask], color='royalblue')
            plt.title(f"Espectro de Fourier - Canal {canales[idx]}")
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Amplitud")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    def plot_time_frequency(self, canal: str|int = 0, fmin=None, fmax=None, start_time=None, end_time=None, nperseg=256, absoluto=False):
        """
        Calcula y grafica una representación tiempo-frecuencia para un canal específico de EEG.

        Args:
            canal : índice o nombre del canal.
            fmin, fmax : límites del eje de frecuencia (Hz).
            start_time : Tiempo (en segundos) de inicio del gráfico
            end_time : Tiempo (en segundos) final del gráfico
            nperseg : tamaño de ventana para la STFT.
            absoluto : Ajusta el eje temporal de grafico (recomendado cuando se intenta usar el metodo en una señal previamente recortada)

        """
        
        if canal is None:
            canal_idx = 0

        elif isinstance(canal, int):
            canal_idx = canal

        elif isinstance(canal, str):
            if self.info is None or "Nombre canales" not in self.info:
                raise ValueError("No se puede identificar el canal por nombre: falta el objeto Info.")
            try:
                canal_idx = self.info["Nombre canales"].index(canal)
            except ValueError:
                raise ValueError(f"Canal '{canal}' no encontrado en Info.")
        else:
            raise ValueError("El parámetro 'canal' debe ser int, str o None.")
        
        if canal_idx < 0 or canal_idx >= self.data.shape[0]:
            raise ValueError("Índice de canal fuera de rango.")
        
        if absoluto:   # Restar el desplazamiento del segmento (usar solo para cuando tengo una señal ya recortada)
            if start_time is not None:
                start_time = start_time - self.first_samp
            if end_time is not None:
                end_time = end_time - self.first_samp
        
        start_idx = int(start_time * self.sfreq) if start_time is not None else 0
        end_idx = int(end_time * self.sfreq) if end_time is not None else self.data.shape[1]
        
        señal = self.data_ref[canal_idx, start_idx:end_idx].flatten()

        f, t, Sxx = spectrogram(señal, fs=self.sfreq, nperseg=nperseg)
        t_real = t + self.first_samp + (start_time if start_time is not None else 0)

        if fmin is not None: 
            f_mask = f >= fmin
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]

        if fmax is not None: 
            f_mask = f <= fmax
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]

        plt.figure(figsize=(10, 5))
        # plt.pcolormesh(t_real, f, Sxx, shading='gouraud', cmap='magma')
        plt.pcolormesh(t_real, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
        plt.title(f"EEG Tiempo-Frecuencia - Canal {canal}")
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.colorbar(label='Potencia')
        plt.tight_layout()
        plt.show()

    def plot_hilbert_transform(self, picks=None, tmin:int|float=None, tmax:int|float=None):
        """
        Calcula y grafica la transformada de Hilbert de uno o varios canales, mostrando la envolvente.

        Args:
            picks : Lista de nombres de canales o None para gráficar todos los canales.
            tmin, tmax : Tiempo de inicio y fin del segmento a graficar (en segundos).

        Returns:
            hilbert_transform : Matriz compleja con la transformada de Hilbert aplicada.
        """
        if self.info is None or "Nombre canales" not in self.info:
            raise ValueError("Se requiere el objeto Info con los nombres de los canales para graficar la transformada de Hilbert.")
        
        channels = self.info.get("Nombre canales")         # Falta verificar si el objeto Info existe

        n_canales, n_muestras = self.data_ref.shape

        if picks is None:
            canales_idx = np.arange(n_canales)
        else:
            try:
                canales_idx = [channels.index(canal) for canal in picks]
            except:
                raise ValueError("Error: uno o más canales no se encuentran en el objeto Info.")

        start = 0
        end = n_muestras

        if tmin is not None:
            start = int(tmin * self.sfreq)

        if tmax is not None:
            end = int(tmax * self.sfreq)

        if start < 0 or end > n_muestras or start >= end:
            raise ValueError("Rango temporal inválido.")

        data_segment = self.data_ref[canales_idx, start:end]    # Recorto de señal

        hilbert_transform = scipy.signal.hilbert(data_segment)  # Aplico transformada de Hilbert
        envolvente = np.abs(hilbert_transform)

        tiempo = np.arange(start, end) / self.sfreq + self.first_samp           # Vector de tiempo

        for i, canal_idx in enumerate(canales_idx):
            plt.figure(figsize=(12, 4))
            plt.plot(tiempo, data_segment[i], label='Señal original')                       # Señal original
            plt.plot(tiempo, envolvente[i], linestyle='--', label='Envolvente (Hilbert)')   # Envolvente
            plt.title(f"Transformada de Hilbert - Canal: {channels[canal_idx]}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud (uV)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return hilbert_transform
    
    def filter(self, l_freq, h_freq, notch_freq = 50, order = 4)-> "EEGSignal":
        """
        Sobreescribe el metodo filter de RawSignal para que devuelva un objeto EEGSignal

        Aplica un filtro pasabanda (Butterworth) y un filtro notch a la señal fisiológica.
        El filtro notch permite eliminar una frecuencia fija (por defecto 50 Hz).
        Filtro pasabanda Butterworth que permite mantener solo las frecuencias entre l_freq y h_freq.

        Args:
            l_freq : Frecuencia de corte baja del filtro pasabanda (en Hz).
            h_freq : Frecuencia de corte alta del filtro pasabanda (en Hz).
            notch_freq : Frecuencia del filtro notch para eliminar ruido (por defecto 50 Hz).
            order : Orden del filtro Butterworth (por defecto 4).

        Returns:
            EEGSignal : Nueva instancia de RawSignal con los datos filtrados.

        Raises:
            ValueError : Si las frecuencias de corte no son válidas.
        """

        filtrada = super().filter(l_freq, h_freq, notch_freq, order)
        
        return EEGSignal(filtrada.data, sfreq=self.sfreq, info=filtrada.info, anotaciones=filtrada.anotaciones, first_samp=filtrada.first_samp,
                         referencia=self.referencia, canal=self.canal_ref)
    
    def crop(self, tmin:int|float=0.0, tmax:int|float=None) -> "EEGSignal":
        """
        Sobreescribe el metodo crop de RawSignal para que devuelva un objeto EEGSignal

        Limita los datos dentro de EEGSignal para obtener un nuevo objeto EEGSignal pero con una cantidad de muestras recortadas.
        
        Args:
            tmin : Tiempo inicial en segundos para iniciar el recorte (por defecto es 0.0).
            tmax : Tiempo final en segundos para finalizar el recorte (por defecto es None).
        
        Return:
            EEGSignal : Nueva instancia de 'EEGSignal' que contiene el segmento temporal recortado.
        
        Raises
            Value Error : Si los tiempos 'tmin' o 'tmax' están fuera del rango de la señal. 
        
        """

        recorte = super().crop(tmin=tmin, tmax=tmax)
        
        return EEGSignal(recorte.data, sfreq=self.sfreq, info=recorte.info, anotaciones=recorte.anotaciones, first_samp=recorte.first_samp,
                         referencia=self.referencia, canal=self.canal_ref)
    
    def drop_channels(self, ch_names:list|np.ndarray) -> "EEGSignal":
        """ 
        Sobreescribe el metodo drop_channels de RawSignal para que devuelva un objeto EEGSignal

        Elimina uno o más canales a partir de ch_names.

        Args:
            ch_names : Nombres de canales a eliminar

        Return:
            Objeto EEGSignal """
        
        drop = super().drop_channels(ch_names=ch_names)
        return EEGSignal(drop.data, sfreq=self.sfreq, info=drop.info, anotaciones=drop.anotaciones, first_samp=drop.first_samp,
                         referencia=self.referencia, canal=self.canal_ref)
    
    def pick(self, canales=None, slice:list=None) -> "EEGSignal":
        """
        Sobreescribe el metodo pick de RawSignal para que devuelva un objeto EEGSignal

        Retorna un subset de canales seleccionados.

        Args:
            canales : Lista con los canales para armar el subset. Por defecto es None.
                Puede ser:
                    - list[str] : lista de nombres de canales
                    - list[int] : lista de índices de canales
            
            slice : Lista con los extremos para hacer slicing. Por defecto es None.
            
        Returns:
            EEGSignal : Nueva instancia con los canales seleccionados.

        Raises: ValueError:
                    Si el canal especificado no existe.
                    Si el índice está fuera del rango de canales. """
        
        picks = super().pick(canales=canales, slice=slice)
        return EEGSignal(picks.data, sfreq=self.sfreq, info=picks.info, anotaciones=picks.anotaciones, first_samp=picks.first_samp,
                         referencia=self.referencia, canal=self.canal_ref)
    
class EMGSignal(RawSignal):
    def __init__(self, data:np.ndarray, sfreq:float, info:Info=None, anotaciones:Anotaciones=None, first_samp: int = 0, umbral_microv=5):
        """
        Clase para representar una señal de electromiografía (EMG).

        Args:
            data : Array 2D con la señal EMG. Forma: (n_canales, n_muestras)
            sfreq : Frecuencia de muestreo en Hz.
            umbral_microv : Umbral de detección de activación en microvoltios (por defecto 5).

        """

        data = data if data.ndim == 2 else data[np.newaxis, :]

        super().__init__(data, sfreq, info, anotaciones, first_samp)

        self.umbral = umbral_microv

        self.activaciones = self.detectar_activaciones()

    def detectar_activaciones(self):
        """
        Detecta eventos de activación muscular en la señal EMG basados en un umbral.

        Recorre cada canal de la señal EMG y determina los índices de muestra 
        en los que la amplitud de la señal supera el umbral especificado (en microvoltios). 
        Se considera que en esos puntos ocurre una activación muscular.

        Returns:
            activaciones : lista de arrays.
                Para cada canal, contiene un array con los índices de muestra
                donde se detectó una activación 
        """
        activaciones = []

        for canal in range(self.data.shape[0]):
            canal_data = self.data[canal]
            indices = np.where(canal_data > self.umbral)[0]
            activaciones.append(indices)

        return activaciones

    def plot_activaciones(self, canal=None, start_time=None, end_time=None, umbral=None):
        """
        Grafica la señal EMG de un canal específico junto con las activaciones detectadas.

        Permite visualizar la señal EMG resaltando los puntos donde la amplitud de la señal 
        supera el umbral de activación previamente definido. 

        Args:
            canal : Índice del canal a graficar. Por defecto es 0.
            start_time : Tiempo (en segundos) de inicio del gráfico.
            end_time : Tiempo (en segundos) final del gráfico.
            umbral_microv : Umbral de detección de activación en microvoltios.

        Raises:
            ValueError: Si el índice de canal proporcionado está fuera del rango de canales disponibles.

        """

        if canal is None:
            canal_idx = 0

        elif isinstance(canal, str):
            if self.info is None or "Nombre canales" not in self.info:
                raise ValueError("No se puede identificar el canal por nombre: falta el objeto Info.")
            try:
                canal_idx = self.info["Nombre canales"].index(canal)
            except ValueError:
                raise ValueError(f"Canal '{canal}' no encontrado en Info.")
            canal_idx = self.info["Nombre canales"].index(canal)

        else:
            canal_idx = canal
            if canal_idx > self.data.shape[0]:
                raise ValueError(f" El canal de indice '{canal}' no se encuentra en la señal")


        start_idx = int(start_time * self.sfreq) if start_time is not None else 0
        end_idx = int(end_time * self.sfreq) if end_time is not None else self.data.shape[1]

        señal_recorte = self.data[canal_idx, start_idx:end_idx]
        tiempo = np.arange(start_idx, end_idx) / self.sfreq

        umbral = umbral if umbral is not None else self.umbral

        activaciones_idx = np.where(señal_recorte > umbral)[0]

        plt.figure(figsize=(12, 4))
        plt.plot(tiempo, señal_recorte, label='EMG')
        if len(activaciones_idx) > 0:
            plt.plot(tiempo[activaciones_idx], señal_recorte[activaciones_idx], 'ro', label='Activaciones detectadas')
        plt.title(f'Señal EMG - Canal {canal}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [µV]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_spectrogram(self, canal: str|int = None, fmin=None, fmax=None, start_time=None, end_time=None, nperseg=256, absoluto=False):
        """
        Grafica el espectrograma de un canal específico de la señal EMG.

        El espectrograma muestra cómo varía la distribución de potencia en frecuencia 
        a lo largo del tiempo. Se aplica una escala logarítmica (decibeles) para visualizar mejor los componentes 
        de baja amplitud.

        Args:
            canal : int, opcional
                Índice del canal a analizar. Por defecto es 0.
                fmin, fmax : límites del eje de frecuencia (Hz).
                start_time : Tiempo (en segundos) de inicio del gráfico.
                end_time : Tiempo (en segundos) final del gráfico.
                nperseg : tamaño de ventana para la STFT.
                absoluto : Ajusta el eje temporal de grafico (recomendado cuando se intenta usar el metodo en una señal previamente recortada)

        Raises:
            ValueError: Si el índice del canal está fuera del rango de canales disponibles.

        """

        if canal is None:
            canal_idx = 0

        elif isinstance(canal, int):
            canal_idx = canal

        elif isinstance(canal, str):
            if self.info is None or "Nombre canales" not in self.info:
                raise ValueError("No se puede identificar el canal por nombre: falta el objeto Info.")
            try:
                canal_idx = self.info["Nombre canales"].index(canal)
            except ValueError:
                raise ValueError(f"Canal '{canal}' no encontrado en Info.")
        else:
            raise ValueError("El parámetro 'canal' debe ser int, str o None.")
        
        if canal_idx < 0 or canal_idx >= self.data.shape[0]:
            raise ValueError("Índice de canal fuera de rango.")
        
        start_idx = int(start_time * self.sfreq) if start_time is not None else 0
        end_idx = int(end_time * self.sfreq) if end_time is not None else self.data.shape[1]
        
        señal = self.data[canal_idx, start_idx:end_idx].flatten()
        f, t, Sxx = spectrogram(señal, fs=self.sfreq)
        t_real = t + self.first_samp + (start_time if start_time is not None else 0)

        if absoluto:   # Restar el desplazamiento del segmento (usar solo para cuando tengo una señal ya recortada)
            if start_time is not None:
                start_time = start_time - self.first_samp
            if end_time is not None:
                end_time = end_time - self.first_samp

        if fmin is not None: 
            f_mask = f >= fmin
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]

        if fmax is not None: 
            f_mask = f <= fmax
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t_real, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.title(f'Espectrograma - Canal {canal}')
        plt.colorbar(label='Potencia [dB]')
        plt.tight_layout()
        plt.show()

    def plot_hilbert(self, picks=None, tmin:int|float=None, tmax:int|float=None):
        """
        Calcula y grafica la transformada de Hilbert de uno o varios canales, mostrando la envolvente.

        Args:
            picks : Lista de nombres de canales o None para gráficar todos los canales.
            tmin, tmax : Tiempo de inicio y fin del segmento a graficar (en segundos).

        Returns:
            hilbert_transform : Matriz compleja con la transformada de Hilbert aplicada.
        """

        if self.info is None or "Nombre canales" not in self.info:
            raise ValueError("Se requiere el objeto Info con los nombres de los canales para graficar la transformada de Hilbert.")
        
        channels = self.info.get("Nombre canales")         # Falta verificar si el objeto Info existe

        n_canales, n_muestras = self.data.shape

        if picks is None:
            canales_idx = np.arange(n_canales)
        else:
            try:
                canales_idx = [channels.index(canal) for canal in picks]
            except:
                raise ValueError("Error: uno o más canales no se encuentran en el objeto Info.")

        start = 0
        end = n_muestras

        if tmin is not None:
            start = int(tmin * self.sfreq)

        if tmax is not None:
            end = int(tmax * self.sfreq)

        if start < 0 or end > n_muestras or start >= end:
            raise ValueError("Rango temporal inválido.")

        data_segment = self.data[canales_idx, start:end]    # Recorto de señal

        hilbert_transform = hilbert(data_segment)  # Aplico transformada de Hilbert
        envolvente = np.abs(hilbert_transform)

        tiempo = np.arange(start, end) / self.sfreq + self.first_samp           # Vector de tiempo

        for i, canal_idx in enumerate(canales_idx):
            plt.figure(figsize=(12, 4))
            plt.plot(tiempo, data_segment[i], label='Señal original')                       # Señal original
            plt.plot(tiempo, envolvente[i], linestyle='--', label='Envolvente (Hilbert)')   # Envolvente
            plt.title(f"Transformada de Hilbert - Canal: {channels[canal_idx]}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud (uV)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return hilbert_transform
    
    def filter(self, l_freq, h_freq, notch_freq = 50, order = 4)-> "EMGSignal":
        """
        Sobreescribe el metodo filter de RawSignal para que devuelva un objeto EMGSignal

        Aplica un filtro pasabanda (Butterworth) y un filtro notch a la señal fisiológica.
        El filtro notch permite eliminar una frecuencia fija (por defecto 50 Hz).
        Filtro pasabanda Butterworth que permite mantener solo las frecuencias entre l_freq y h_freq.

        Args:
            l_freq : Frecuencia de corte baja del filtro pasabanda (en Hz).
            h_freq : Frecuencia de corte alta del filtro pasabanda (en Hz).
            notch_freq : Frecuencia del filtro notch para eliminar ruido (por defecto 50 Hz).
            order : Orden del filtro Butterworth (por defecto 4).

        Returns:
            EMGSignal : Nueva instancia de RawSignal con los datos filtrados.

        Raises:
            ValueError : Si las frecuencias de corte no son válidas.
        """

        filtrada = super().filter(l_freq, h_freq, notch_freq, order)
        
        return EMGSignal(data=filtrada.data, sfreq=self.sfreq, info=filtrada.info, anotaciones=filtrada.anotaciones, first_samp=filtrada.first_samp,
                         umbral_microv=self.umbral)
    
    def crop(self, tmin:int|float=0.0, tmax:int|float=None) -> "EMGSignal":
        """
        Sobreescribe el metodo crop de RawSignal para que devuelva un objeto EMGSignal

        Limita los datos dentro de EMGSignal para obtener un nuevo objeto EMGSignal pero con una cantidad de muestras recortadas.
        
        Args:
            tmin : Tiempo inicial en segundos para iniciar el recorte (por defecto es 0.0).
            tmax : Tiempo final en segundos para finalizar el recorte (por defecto es None).
        
        Return:
            EMGSignal : Nueva instancia de 'EEGSignal' que contiene el segmento temporal recortado.
        
        Raises
            Value Error : Si los tiempos 'tmin' o 'tmax' están fuera del rango de la señal. 
        
        """

        recorte = super().crop(tmin=tmin, tmax=tmax)
        
        return EMGSignal(recorte.data, sfreq=self.sfreq, info=recorte.info, anotaciones=recorte.anotaciones, first_samp=recorte.first_samp,
                         umbral_microv=self.umbral)
    
    def drop_channels(self, ch_names:list|np.ndarray) -> "EMGSignal":
        """ 
        Sobreescribe el metodo drop_channels de RawSignal para que devuelva un objeto EMGSignal

        Elimina uno o más canales a partir de ch_names.

        Args:
            ch_names : Nombres de canales a eliminar

        Return:
            Objeto EMGSignal """
        
        drop = super().drop_channels(ch_names=ch_names)
        return EMGSignal(drop.data, sfreq=self.sfreq, info=drop.info, anotaciones=drop.anotaciones, first_samp=drop.first_samp,
                         umbral_microv=self.umbral)
    
    def pick(self, canales=None, slice:list=None) -> "EMGSignal":
        """
        Sobreescribe el metodo pick de RawSignal para que devuelva un objeto EMGSignal

        Retorna un subset de canales seleccionados.

        Args:
            canales : Lista con los canales para armar el subset. Por defecto es None.
                Puede ser:
                    - list[str] : lista de nombres de canales
                    - list[int] : lista de índices de canales
            
            slice : Lista con los extremos para hacer slicing. Por defecto es None.
            
        Returns:
            EMGSignal : Nueva instancia con los canales seleccionados.

        Raises: ValueError:
                    Si el canal especificado no existe.
                    Si el índice está fuera del rango de canales. """
        
        picks = super().pick(canales=canales, slice=slice)
        return EMGSignal(picks.data, sfreq=self.sfreq, info=picks.info, anotaciones=picks.anotaciones, first_samp=picks.first_samp,
                         umbral_microv=self.umbral)

class ECGSignal(RawSignal):
    def __init__(self, data, sfreq, info=None, anotaciones=None, first_samp:int = 0):
        """
        Clase especializada para el análisis de señales ECG.

        Args:
            - data (np.ndarray): Señal ECG, forma esperada (n_canales, n_muestras)
            - sfreq : frecuencia de muestreo
            - info : Objeto Info con la información de la señal.
            - anotaciones Objeto Anotaciones con los eventos asociados.
            - first_samp : Indica el tiempo de inicio del segmento, pero no recorta los datos. (Por defecto es 0)

        """

        data = data if data.ndim == 2 else data[np.newaxis, :]

        super().__init__(data, sfreq, info, anotaciones)

        self.r_peaks = []
        self.hr = 0.0

    def detectar_r_peaks(self, canales:list=None, start:float=None, end:float=None, height:float=None, distance:float=None):
        """
        Detecta los picos R en uno o más canales de una señal ECG.

        Args:
            canales : list[str] Lista con los nombres de los canales a analizar. Si es None, se usan todos.
            start : Tiempo de inicio en segundos para el análisis.
            end : Tiempo de fin en segundos para el análisis.
            height : Umbral mínimo de altura para considerar un pico R.
            distance : Distancia mínima entre picos R, en número de muestras.

        Returns:
            list[np.ndarray]: Lista con los índices de los picos R para cada canal.
        """

        if self.info is None or "Nombre canales" not in self.info:
            raise ValueError("El objeto Info con los nombres de los canales es requerido.")
        
        nombres_canales = self.info.get("Nombre canales")
        n_canales, n_muestras = self.data.shape

        if canales is None:
            canales = nombres_canales
        
        if not all(c in nombres_canales for c in canales):
            raise ValueError("Uno o más nombres de canales no existen en el objeto Info.")
        
        canales_idx = [nombres_canales.index(c) for c in canales]

        start_idx = int(start * self.sfreq) if start is not None else 0
        end_idx = int(end * self.sfreq) if end is not None else n_muestras

        r_peaks_list = []

        for idx in canales_idx:
            señal = self.data[idx, start_idx:end_idx]

            _height = np.mean(señal) if height is None else height
            _distance = self.sfreq / 2.5 if distance is None else distance

            peaks, _ = find_peaks(señal, height=_height, distance=_distance)
            r_peaks_list.append(peaks + start_idx)

        self.r_peaks = r_peaks_list
        return r_peaks_list 
    
    def calcular_hr(self):
        """
        Calcula la frecuencia cardíaca (heart rate) a partir de los picos R detectados.

        Returns:
            list[float] : Devuelve una lista con las frecuancias cardiacas.
        """
        if not self.r_peaks:
            self.r_peaks = self.detectar_r_peaks()
        
        picos = self.r_peaks

        if isinstance(picos, np.ndarray):
            rr_intervals = np.diff(picos) / self.sfreq
            self.hr = 60. / np.mean(rr_intervals)
            return self.hr
        
        elif isinstance(picos, list):
            frec_card = []
            for pico in picos:
                rr_intervals = np.diff(pico) / self.sfreq
                hr = 60. / np.mean(rr_intervals)
                frec_card.append(hr)
            self.hr = frec_card
            return frec_card

    def plot_time_frequency(self, canal: str|int = 0, fmin=None, fmax=None, start_time=None, end_time=None, nperseg=256, absoluto=False):
        """
        Calcula y grafica una representación tiempo-frecuencia para un canal específico de EEG.

        Args:
            canal : índice o nombre del canal.
            fmin, fmax : límites del eje de frecuencia (Hz).
            start_time : Tiempo (en segundos) de inicio del gráfico
            end_time : Tiempo (en segundos) final del gráfico
            nperseg : tamaño de ventana para la STFT.
            absoluto : Ajusta el eje temporal de grafico (recomendado cuando se intenta usar el metodo en una señal previamente recortada)

        """
        
        if canal is None:
            canal_idx = 0

        elif isinstance(canal, int):
            canal_idx = canal

        elif isinstance(canal, str):
            if self.info is None or "Nombre canales" not in self.info:
                raise ValueError("No se puede identificar el canal por nombre: falta el objeto Info.")
            try:
                canal_idx = self.info["Nombre canales"].index(canal)
            except ValueError:
                raise ValueError(f"Canal '{canal}' no encontrado en Info.")
        else:
            raise ValueError("El parámetro 'canal' debe ser int, str o None.")
        
        if canal_idx < 0 or canal_idx >= self.data.shape[0]:
            raise ValueError("Índice de canal fuera de rango.")
        
        if absoluto:   # Restar el desplazamiento del segmento (usar solo para cuando tengo una señal ya recortada)
            if start_time is not None:
                start_time = start_time - self.first_samp
            if end_time is not None:
                end_time = end_time - self.first_samp
        
        start_idx = int(start_time * self.sfreq) if start_time is not None else 0
        end_idx = int(end_time * self.sfreq) if end_time is not None else self.data.shape[1]
        
        señal = self.data[canal_idx, start_idx:end_idx].flatten()

        f, t, Sxx = spectrogram(señal, fs=self.sfreq, nperseg=nperseg)
        t_real = t + self.first_samp + (start_time if start_time is not None else 0)

        if fmin is not None: 
            f_mask = f >= fmin
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]

        if fmax is not None: 
            f_mask = f <= fmax
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]

        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t_real, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
        plt.title(f"EEG Tiempo-Frecuencia - Canal {canal}")
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.colorbar(label='Potencia')
        plt.tight_layout()
        plt.show()

    def plot_r_peaks(self, picks:list=None, start:float=None, end:float=None):

        """
        Grafica la señal de ECG con los picos R detectados superpuestos.

        Args:
            picks : lista con nombres de los canales a graficar (si None, grafica todos).
            start : tiempo inicial en segundos para la ventana a graficar.
            end   : tiempo final en segundos para la ventana a graficar.
        """

        n_canales, n_muestras = self.data.shape

        if picks is None:                                    # Si picks queda por defecto, seleccionar todos los canales
            channels = self.info.get("Nombre canales")
            canales_idx = np.arange(n_canales)

        else:
            channels = self.info.get("Nombre canales")
            try:
                canales_idx = [channels.index(canal) for canal in picks]
            except:
                raise ValueError(f"Error: uno o más canales no se encuentran en los canales Ingresados desde el objeto Info")
        
        start_idx = int(start * self.sfreq) if start is not None else 0
        end_idx = int(end * self.sfreq) if end is not None else n_muestras

        señal = self.data[canales_idx, start_idx:end_idx]

        tiempo = np.arange(start_idx, end_idx) / self.sfreq

        if len(self.r_peaks) == 0:
            self.detectar_r_peaks()
        
        r_peaks = self.r_peaks
        if isinstance(r_peaks, np.ndarray):
            r_peaks = [r_peaks]
        
        for i, idx in enumerate(canales_idx):

            señal = self.data[idx, start_idx:end_idx]
            picos_rel = np.array(r_peaks[i]) - start_idx

            picos_rel = picos_rel[(picos_rel >= 0) & (picos_rel < señal.shape[0])]

            plt.figure(figsize=(12, 4))
            plt.plot(tiempo, señal, label=f"ECG canal {channels[idx]}")
            plt.plot(tiempo[picos_rel], señal[picos_rel], 'ro', label="Picos R")
            plt.title(f"Señal ECG con Picos R - Canal {channels[idx]}")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Amplitud [mV]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    def filter(self, l_freq, h_freq, notch_freq = 50, order = 4)-> "ECGSignal":
        """
        Sobreescribe el metodo filter de RawSignal para que devuelva un objeto ECGSignal

        Aplica un filtro pasabanda (Butterworth) y un filtro notch a la señal fisiológica.
        El filtro notch permite eliminar una frecuencia fija (por defecto 50 Hz).
        Filtro pasabanda Butterworth que permite mantener solo las frecuencias entre l_freq y h_freq.

        Args:
            l_freq : Frecuencia de corte baja del filtro pasabanda (en Hz).
            h_freq : Frecuencia de corte alta del filtro pasabanda (en Hz).
            notch_freq : Frecuencia del filtro notch para eliminar ruido (por defecto 50 Hz).
            order : Orden del filtro Butterworth (por defecto 4).

        Returns:
            ECGSignal : Nueva instancia de RawSignal con los datos filtrados.

        Raises:
            ValueError : Si las frecuencias de corte no son válidas.
        """

        filtrada = super().filter(l_freq, h_freq, notch_freq, order)
        
        return ECGSignal(data=filtrada.data, sfreq=self.sfreq, info=filtrada.info, anotaciones=filtrada.anotaciones, first_samp=filtrada.first_samp)
    
    def crop(self, tmin:int|float=0.0, tmax:int|float=None) -> "ECGSignal":
        """
        Sobreescribe el metodo crop de RawSignal para que devuelva un objeto ECGSignal

        Limita los datos dentro de ECGSignal para obtener un nuevo objeto ECGSignal pero con una cantidad de muestras recortadas.
        
        Args:
            tmin : Tiempo inicial en segundos para iniciar el recorte (por defecto es 0.0).
            tmax : Tiempo final en segundos para finalizar el recorte (por defecto es None).
        
        Return:
            EMGSignal : Nueva instancia de 'EEGSignal' que contiene el segmento temporal recortado.
        
        Raises
            Value Error : Si los tiempos 'tmin' o 'tmax' están fuera del rango de la señal. 
        
        """

        recorte = super().crop(tmin=tmin, tmax=tmax)
        
        return ECGSignal(recorte.data, sfreq=self.sfreq, info=recorte.info, anotaciones=recorte.anotaciones, first_samp=recorte.first_samp)
    
    def drop_channels(self, ch_names:list|np.ndarray) -> "ECGSignal":
        """ 
        Sobreescribe el metodo drop_channels de RawSignal para que devuelva un objeto ECGSignal

        Elimina uno o más canales a partir de ch_names.

        Args:
            ch_names : Nombres de canales a eliminar

        Return:
            Objeto ECGSignal """
        
        drop = super().drop_channels(ch_names=ch_names)
        return ECGSignal(drop.data, sfreq=self.sfreq, info=drop.info, anotaciones=drop.anotaciones, first_samp=drop.first_samp)
    
    def pick(self, canales=None, slice:list=None) -> "ECGSignal":
        """
        Sobreescribe el metodo pick de RawSignal para que devuelva un objeto ECGSignal

        Retorna un subset de canales seleccionados.

        Args:
            canales : Lista con los canales para armar el subset. Por defecto es None.
                Puede ser:
                    - list[str] : lista de nombres de canales
                    - list[int] : lista de índices de canales
            
            slice : Lista con los extremos para hacer slicing. Por defecto es None.
            
        Returns:
            EMGSignal : Nueva instancia con los canales seleccionados.

        Raises: ValueError:
                    Si el canal especificado no existe.
                    Si el índice está fuera del rango de canales. """
        
        picks = super().pick(canales=canales, slice=slice)
        return ECGSignal(picks.data, sfreq=self.sfreq, info=picks.info, anotaciones=picks.anotaciones, first_samp=picks.first_samp,
                         umbral_microv=self.umbral)