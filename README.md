Este GitHub es creado con el fin de compartir con toda la comunidad estudiantil de la EIA
y del mundo, los conocimientos adquiridos en el desarrollo del trabajo de grado llamado
"SISTEMA DE RETROALIMENTACIÓN EN EL ESPACIO DE TRABAJO DEL ROBOT ABB POR MEDIO DE TECNOLOGÍAS DE VISIÓN ARTIFICIAL"

A continuación se explicarán los diferentes archivos que se pueden encontrar aquí:

Data: Datos usados para el entrenamiento del modelo de visión artificial usando YOLOv8.

imagenes: Carpeta de salida de la aplicación de etiquetado manual. Esta carpeta tiene todas las
imagenes etiquetadas con su respectivo archivo de texto. De esta carpeta fue de donde se hizo
manualmente la separación de datos para la carpeta "Data"

runs/detect: Carpeta de salida creada automaticamente al ejecutar el archivo "ImageValidation.py".
Esta carpeta contiene la información estadistica del comportamiento del modelo y como se entrenó para
llegar al output.

Codigo RAPID: Codigo RAPID usado en la aplicación RoboStudio de ABB para poder ejecutar los 
diferentes comandos proporcionados desde MATLAB.

ImageValidation.py: Codigo python usado para validar el modelo entrenado con una sola imagen especifica.
image_90.png fue la imagen usada en este codigo.

classes.txt: Archivo en el que se especifican las clases o objetos a diferenciar. Es importante el orden

dataset.yaml: Archivo de configuración para el entrenamiento del modelo. En este archivo se especifica la ruta
de los datos de entrenamiento y evaluación, además de las clases a clasificar.

main.py: Archivo de python en el que está cargado el modelo de vision artificial, junto con toda la interfaz de usuario
que habilita la comunicación con matlab igualmente. Este archivo y el archivo de MATLAB trabajan juntos,
ya que uno analiza y procesa los datos obtenidos de la camara, y la envía a MATLAB que es el mediador
entre OPC y Python.

mainMatlab.m: Archivo de MATLAB encargado de la recepción y envío de información de python y OPC. Este archivo recibe 
información de ambos lados para permitir un flujo más organizado y sencillo de analizar.

