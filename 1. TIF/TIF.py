import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import scipy.signal
from matplotlib.patches import Patch

class Info:
    """
    Clase para almacenar información acerca del registro de datos
    """

    def __init__(self, experimenter:str, subject_info:dict, ch_names:list | str, ch_types:list | str, bads:list | str, description:str, fm:float = 512):
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
        """
        
        if len(ch_names) != len(ch_types):
            raise ValueError ("La cantidad de canales y los tipos de canales deben ser la misma")
        
        self.data = {"Experimentador": experimenter,
                     "Sujeto":subject_info,
                     "Nombre canales": ch_names,
                     "Tipo canales": ch_types,
                     "Canales malos": bads,
                     "Descripción": description,
                     "Frecuencia muestreo": fm}        

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
            first_samp : Índice del primer muestreo a utilizar (default es 0).
            
        Raises:
            ValueError : Si el array 'data' no tiene la forma '(n_canales , n_muestras)'.
            ValueError : Si el índice 'first_samp' está fuera del rango de la señal. """
        
        if not isinstance(data, np.ndarray):     # Que data sea un array de NumPy
            raise ValueError("El parámetro 'data' debe ser un array de NumPy (np.ndarray).")

        if data.ndim != 2:                       # Que data tenga dos dimensiones 
            raise ValueError("El array 'data' debe tener dos dimensiones: (n_canales, n_muestras).")

        n_muestras = data.shape[1]               # Número de muestras es la segunda dimensión
        if not (0 <= first_samp < n_muestras):   # Que first_samp sea un entero positivo y menor que el numero de muestras
            raise ValueError("El índice 'first_samp' está fuera del rango de muestras disponibles.")


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

        return RawSignal(data=datos_crop, sfreq=self.sfreq, info=self.info, anotaciones=anotaciones_copia, first_samp=tmin) # El segmento recortado empieza en 0 (first_samp)
    
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

        return RawSignal(data=datos_filtrados, sfreq=self.sfreq, info=info_copia, anotaciones=self.anotaciones)

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

        # El tiempo relativo en el objeto recortado siempre empieza en 0
        intervalo_inicio = start
        intervalo_fin = start + duration

        canales, tiempo = self.get_data(picks=picks, start=intervalo_inicio, stop=intervalo_fin, times=True)
        n_canales = canales.shape[0]

        fig, axes = plt.subplots(n_canales, 1, figsize=(10, 3 * n_canales), sharex=True)
        if n_canales == 1:
            axes = [axes]

        for i in range(n_canales):
            axes[i].plot(tiempo, canales[i])
            axes[i].set_ylabel(f"Canal {i}")
            axes[i].set_xlabel("Tiempo [s]")
            axes[i].grid(True)

        # El eje x debe mostrar los tiempos absolutos (first_samp + tiempo relativo)
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

            first_samp : Índice del primer punto de la señal. Por defecto es 0.

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
            pass

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
        Aplica una referencia Laplaciana espacial a la señal EEG.

        Este método calcula la diferencia entre la señal de un canal central y la media de sus canales vecinos.

        """
        # Corregir despues
        kernel_size = 3
        laplacian_kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        laplacian_kernel[center, :] = -1
        laplacian_kernel[:, center] = -1
        laplacian_kernel[center, center] = 4

        filtered_signal = scipy.signal.convolve2d(self.data, laplacian_kernel, mode='same', boundary='symm') # mode= same: el resultado tiene el mismo tamaño que la matriz original. # boundary = symm: los bordes se tratan con reflexión simétrica, para evitar que el borde se pierda.
        return filtered_signal

    def apply_laplacian_filter(self):
        """
        Aplica un filtro espacial Laplaciano para resaltar la actividad local de cada canal.

        """

        pass

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
        datos = np.atleast_2d(datos)                         # asegura que siempre sea 2D

        frequencies, spectrum = scipy.signal.welch(datos, fs=self.sfreq, nperseg=1024)
        spectrum = 10 * np.log10(spectrum)       # dB re: 1 μV²/Hz

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
                raise ValueError (f"La frecuencia máxima ingresada ´{fmax}´ no es valida")

        if plot:
            self._plot_fourier_spectrum(frecuencias=frequencies, espectro=spectrum, indices=canales_idx, fmin=fr_min_ejex, fmax=fr_max_ejex)
    
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
                raise ValueError (f"La frecuencia mínima ingresada ´{fmin}´ no es valida")
      
        if fmax:
            if isinstance (fmax, (int, float)):
                if fmax <= fr_max_ejex and fmax > fr_min_ejex:
                    fr_max_ejex = fmax
                else:
                    raise ValueError (f"La frecuencia máxima '{fmax}' debe ser mayor a la frecuencia minima establecida {fr_min_ejex:.2f} Hz.")
            else:
                raise ValueError (f"La frecuencia máxima ingresada ´{fmax}´ no es valida")

        if plot:
            self._plot_fourier_spectrum(frecuencias=freqs, espectro=espectro, indices=canales_idx, fmin=fr_min_ejex, fmax=fr_max_ejex)
    
        return freqs, espectro
    
    def _plot_fourier_spectrum(self, frecuencias:np.ndarray, espectro:np.ndarray, indices, fmin, fmax):
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
    
    def plot_time_frequency(self):
        """
        Calcula y grafica una representación tiempo-frecuencia de un canal específico

        """

        pass

    def plot_hilbert_transform(self, picks):
        """
        Calcula y grafica la transformada de Hilbert de uno o varios canales, mostrando la envolvente.

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
            
        hilbert_transform = scipy.signal.hilbert(self.data_ref[canales_idx])

        envolvente = np.abs(hilbert_transform)

        for i, canal_idx in enumerate(canales_idx):
            plt.figure(figsize=(12, 4))
            plt.plot(self.data_ref[canal_idx], label='Señal original')
            plt.plot(envolvente[i], linestyle='--', label='Envolvente (Hilbert)')
            plt.title(f"Transformada de Hilbert - Canal: {channels[canal_idx]}")
            plt.xlabel("Tiempo (muestras)")
            plt.ylabel("Amplitud (uV)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        return hilbert_transform
    
    def filter(self, l_freq, h_freq, notch_freq = 50, order = 4):
        """
        Sobreescribimos el metodo filter de RawSignal para que devuelva un objeto EEGSignal"""

        filtrada = super().filter(l_freq, h_freq, notch_freq, order)
        
        return EEGSignal(filtrada.data, sfreq=self.sfreq, info=filtrada.info, anotaciones=filtrada.anotaciones, first_samp=filtrada.first_samp,
                         referencia=self.referencia, canal=self.canal_ref )
    
    def crop(self, tmin:int|float=0.0, tmax:int|float=None) -> "EEGSignal":
        """
        Sobreescribimos el metodo crop de RawSignal para que devuelva un objeto EEGSignal"""

        recorte = super().crop(tmin, tmax)
        
        return EEGSignal(recorte.data, sfreq=self.sfreq, info=recorte.info, anotaciones=recorte.anotaciones, first_samp=recorte.first_samp,
                         referencia=self.referencia, canal=self.canal_ref )