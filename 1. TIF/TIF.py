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
        Permitir la adición, eliminación y modificación de eventos."""
    
    def __init__(self):
        """Inicializa la clase con los datos de las anotaciones"""
        pass
    def add(self):
        """Agrega una nueva anotación
            Args:"""
        pass
    def remove(self):
        """Elimina una anotación específica
            Args:"""
        pass
    def get_annotations(self):
        """Devuelve una DataFrame con todas las anotaciones
            Args:"""
        pass
    def find(self):
        """Busca y devuelve las anotaciones que coincidan con una descripción específica
            Args:"""
        pass
    def save(self):
        """Guarda las anotaciones en un archivo .csv
            Args:"""
        pass
    def load(self):
        """Carga las anotaciones desde un archivo .csv
            Args:"""
        pass