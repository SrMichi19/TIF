import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Info:
    """ Clase para almacenar información acerca del registro de datos"""

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
                description : Descripción del registro de datos."""
        
        if len(ch_names) != len(ch_types):
            raise ValueError ("La cantidad de canales y los tipos de canales deben tener la misma longuitud")
        
        self.data = {"Experimentador": experimenter,
                     "Sujeto":subject_info,
                     "Nombre canales": ch_names,
                     "Tipo canales": ch_types,
                     "Canales malos": bads,
                     "Descripción": description,
                     "Frecuencia muestreo": fm}        

    def __contains__(self, clave):
        """ Permite verificar si una clave esta presente en el objeto
            Args:
                clave : Nombre de la clave que se quiere verificar
            Returns:
                True: La clave se encuentra en el objeto
                False: La clave no se encuentra en el objeto"""
        
        if clave in self.data:
            return True
        else:
            return False
        
    def __getitem__(self, clave):
        """ Permite acceder a elementos como un diccionario
            Args
                clave: Nombre de la clave a la que se quiere acceder
            Returns
                Devuelve el valor asociado a la clave ingrsada, o
                False: Cuando la clave ingresada no existe """
        
        if self.__contains__(clave) == True:
            return self.data[clave]
        else:
            return False
        
    def __len__(self) -> int:
        """ Devuelve la cantidad de elementos almacenados"""
        return len(self.data)
    
    def keys(self):
        """ Devuelve las claves del objeto"""
        return [clave for clave in self.data]
    
    def get(self, clave):
        """ Obtiene solo el valor de una clave específica
            Args:
                clave : Clave de la cual se quiere conocer su valor
            Returns:
                Devuleve el valor de la clave si la misma existe, o false en caso de que no exista"""
        
        if self.__contains__(clave) == True:
            return self.data[clave]
        else:
            return False

    def item(self, elemento):
        """ Devuelve los elementos como pares clave-valor de una clave en específica
            Args
                elemento : Elemento del que se quiere obtener la clave y su valor
            Returns:
                tuple : Tupla que contiene la clave y el valor del elemento
                False : Si no exite la clave en el diccionario"""
        
        valor = self.__getitem__(elemento)
        if valor == False:
            return False
        else:
            return (elemento, valor)
        
    def _check_channel(self, channel_name):
        """ Verifica si un canal se encuentra entre los nombres de los canales
            Args:
                channel_name : Canal a verificar si se encuentra en la lista de canales"""
        
        if channel_name in self.data["Nombre canales"]:
            return True
        else:
            return False
        
    def rename_channels(self, nombre_canal, nuevo_nombre):
        """Permite renombrar canales de forma segura
            Args:
                nombre_canal : Nombre del canal que se quiere cambiar
                nuevo_nombre : Nuevo nombre que va a tener el canal
            Returs:
                True : Si se cambio el nombre del canal correctamente
                False: Si no se pudo modificar el nombre del canal"""
        
        if self._check_channel(nombre_canal) == True:
            indice = self.data["Nombre canales"].index(nombre_canal)
            self.data["Nombre canales"][indice] = nuevo_nombre
            return True
        else:
            return False           

class Anotaciones:
    """ Almacena y gestiona información relacionada con eventos en registros fisiológicos. 
        Permite la adición, eliminación y modificación de eventos."""
    
    def __init__(self, onset:np.ndarray, duration:np.ndarray, description):
        """Inicializa la clase con los datos de las anotaciones"""
        self.onset = onset
        self.duration = duration
        self.description = description

        if len(onset) != len(duration):
            raise ValueError ("Onset y duration deben tener la misma cantidad de elementos")
        
        self.anotaciones = pd.DataFrame({"Inicio (s)": self.onset, "Duración (s)": self.duration, "Descripcion": self.description})

    def get_annotations(self):
        """Devuelve una DataFrame con todas las anotaciones que recibe el constructor"""
        return self.anotaciones
    
    def add(self, anotacion:list|tuple):
        """Agrega una nueva anotación
            Args:
                anotacion : Nueva anotación a agregar (se espera la forma [inicio, duración, descripción])"""
        if len(anotacion) != 3:
            return False 
        else:
            self.anotaciones.loc[len(self.anotaciones)] = anotacion  # Agrega un fila
            return True 
    
    def remove(self, anotacion_eliminar):
        """Elimina una anotación específica
            Args:
                anotacion_eliminar : Anotación que se quiere eliminar
            Returns:
                True : Selimino correctamente la anotación
                False : No se pudo eliminar la anotación """
        
        if len(anotacion_eliminar) != 3:
            return False
        else:
            eliminar = (self.anotaciones["Inicio (s)"] == anotacion_eliminar[0]) & (self.anotaciones["Duración (s)"] == anotacion_eliminar[1]) & (self.anotaciones["Descripcion"] == anotacion_eliminar[2])
            indice = self.anotaciones[eliminar].index
            self.anotaciones = self.anotaciones.drop(indice)
            return True

    def find(self, buscar_anotacion):
        """Busca y devuelve una anotación específica
            Args:
                buscar_anotacion : Anotación que se quiere buscar entre los datos 
            Returs:
                Devuelve la anotación o False si la longuitud de la anotación no coincide con la estructura de los datos"""
        
        if len(buscar_anotacion) != 3:
            return False
        else:
            buscar = (self.anotaciones["Inicio (s)"] == buscar_anotacion[0]) & (self.anotaciones["Duración (s)"] == buscar_anotacion[1]) & (self.anotaciones["Descripcion"] == buscar_anotacion[2])
            return self.anotaciones[buscar]

    def save(self, nombre):
        """Guarda las anotaciones en un archivo .csv
            Args:
                nombre : Nombre con el que se guardara el archivo"""
        return self.anotaciones.to_csv(f"{nombre}.csv")
    
    def load(self, archivo):
        """Carga las anotaciones desde un archivo .csv
            Args:
                archivo : Nombre del archivo que se quiere cargar
            Devuelve el dataframe con los datos del archivo csv"""
        
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
            data : Matriz de datos con forma '(n_canales , n_muestras)'.
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

        self.data = data
        self.sfreq = sfreq
        self.info = info
        self.anotaciones = anotaciones
        self.first_samp = first_samp
        
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
            tiempo_vector = np.arange(start_idx, stop_idx) / self.sfreq # t = muestra/fm
            return datos, tiempo_vector

        return datos