# Sistema de Retroalimentación en el Espacio de Trabajo del Robot ABB por Medio de Tecnologías de Visión Artificial.
# Autores: Andrés Urrutia y Kevin Varón. Profesor: David Rozo Osorio

Este repositorio de GitHub ha sido creado para compartir con la comunidad estudiantil de la EIA y del mundo el conocimiento adquirido en el desarrollo del trabajo de grado titulado:

**"Sistema de Retroalimentación en el Espacio de Trabajo del Robot ABB por Medio de Tecnologías de Visión Artificial"**

A continuación, se explica el propósito y contenido de cada archivo y carpeta incluidos en este repositorio.

## Contenido del Repositorio

### Carpetas

- **Data**: Contiene los datos utilizados para el entrenamiento del modelo de visión artificial con YOLOv8.

- **imagenes**: Carpeta de salida de la aplicación de etiquetado manual. Aquí se encuentran las imágenes etiquetadas junto con sus archivos de texto correspondientes. Esta carpeta fue la base para la separación manual de los datos que se encuentran en la carpeta "Data".

- **runs/detect**: Carpeta de salida generada automáticamente al ejecutar el archivo `ImageValidation.py`. Contiene la información estadística sobre el rendimiento del modelo y detalles del proceso de entrenamiento.

### Archivos

- **Codigo RAPID**: Código RAPID usado en la aplicación RoboStudio de ABB. Este código permite la ejecución de comandos desde MATLAB para el robot ABB.

- **ImageValidation.py**: Código en Python para validar el modelo de visión artificial con una imagen específica (`image_90.png`). Este archivo se utiliza para pruebas puntuales del modelo.

- **classes.txt**: Archivo que especifica las clases u objetos a diferenciar. Es fundamental mantener el orden de las clases en este archivo.

- **dataset.yaml**: Archivo de configuración para el entrenamiento del modelo. Define las rutas de los datos de entrenamiento y evaluación, así como las clases que el modelo debe clasificar.

- **main.py**: Código en Python que carga el modelo de visión artificial y la interfaz de usuario para la comunicación con MATLAB. Este archivo trabaja en conjunto con el archivo de MATLAB (`mainMatlab.m`), facilitando el análisis y procesamiento de datos de la cámara y su envío a MATLAB, que actúa como intermediario entre OPC y Python.

- **mainMatlab.m**: Archivo en MATLAB encargado de recibir y enviar información entre Python y OPC. Este archivo organiza el flujo de datos para que sea más sencillo de gestionar y analizar.

## Contribuciones

Si deseas contribuir a este proyecto, por favor, sigue los lineamientos establecidos para mantener la integridad del código y la estructura del repositorio.

## Licencia

Este proyecto se comparte bajo la licencia [especificar licencia aquí]. 

¡Gracias por tu interés en este proyecto y por contribuir al desarrollo del conocimiento en el campo de la visión artificial aplicada a la robótica!


