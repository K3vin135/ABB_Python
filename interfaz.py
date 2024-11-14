import socket
import time
import cv2
from ultralytics import YOLO
import depthai as dai
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.font as font
from PIL import Image, ImageTk
from tkinter import messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# Variables para almacenar las coordenadas filtradas
# Variables para almacenar las coordenadas filtradas, inicializadas en 0
filtered_x = deque([0.0], maxlen=10)
filtered_y = deque([0.0], maxlen=10)
filtered_z = deque([0.0], maxlen=10)


# Inicializa las listas para almacenar los datos de detecciones por clase
boxes_data = []
abb_data = []
abb_base_data = []

anguloFovMed = 18.3
anguloFovMed2 = 30.11

# Carga el modelo YOLOv8
model = YOLO('runs/detect/train/weights/best.pt')

# Crear pipeline para la cámara y la profundidad
pipeline = dai.Pipeline()

# Crear nodos de cámara y profundidad
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
color = pipeline.create(dai.node.ColorCamera)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutColor = pipeline.create(dai.node.XLinkOut)

# Configurar cámaras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
color.setPreviewSize(640, 360)
color.setBoardSocket(dai.CameraBoardSocket.RGB)

# Configurar nodo de profundidad
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setSubpixel(True)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Configurar XLinkOut
xoutDepth.setStreamName("depth")
xoutColor.setStreamName("color")
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDisparity.setStreamName("disparity")

# Enlazar nodos
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xoutDisparity.input)
stereo.depth.link(xoutDepth.input)
color.preview.link(xoutColor.input)

running = False
device = None
conn = None

# Configurar la interfaz de tkinter
root = tk.Tk()
root.title("YOLOv8 Inference with Depth")
width = 1320  # Ancho de la ventana
height = 800
root.geometry("%dx%d" % (width, height))

main_frame = ttk.Frame(root)
main_frame.place(x=0, y=0, width=1280, height=900)

lblVideo = None
lblVideoDepth = None
lbl1 = None
lblCoordMatlab = None
canvas = None
fig = None
ax = None

def get_filtered_value(values):
    return sum(values) / len(values)

def createFrameZeros():
    global lblVideo, lblVideoDepth, lbl1, lblCoordMatlab, canvas, fig, ax
    lblVideo = tk.Label(main_frame, borderwidth=2, relief="solid")
    lblVideo.place(x=20, y=50)

    # Frame original 
    frame = np.zeros([360, 640, 3], dtype=np.uint8)  # Tamaño ajustado a 640x360
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgArray = Image.fromarray(frame)
    imgTk = ImageTk.PhotoImage(image=imgArray)
    lblVideo.configure(image=imgTk)
    lblVideo.image = imgTk
        # Frame original 
    frame2 = np.zeros([180, 360, 3], dtype=np.uint8)  # Tamaño ajustado a 640x360
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    imgArray2 = Image.fromarray(frame2)
    imgTk2 = ImageTk.PhotoImage(image=imgArray2)


    # Frame Video Profundidad
    lblVideoDepth = tk.Label(main_frame, borderwidth=2, relief="solid")
    lblVideoDepth.place(x=720, y=50)
    lblVideoDepth.configure(image=imgTk2)  # Le paso las propiedades de imgTk
    lblVideoDepth.image = imgTk

    # Frame ROI 1 - ROI ABB
    lbl1 = tk.Label(main_frame, borderwidth=2, relief="solid")
    lbl1.place(x=720, y=270)
    lbl1.configure(image=imgTk2)
    lbl1.image = imgTk
    
    # Etiqueta para mostrar coordenadas recibidas desde MATLAB
    lblCoordMatlab = tk.Label(root, text="Coordenadas recibidas del ABB:", font=('Helvetica', 9, 'bold'))
    lblCoordMatlab.place(x=720, y=600)
    lblCoordMatlab.config(text="Coordenadas recibidas del ABB: No recibidas")

    # Crear figura y eje para el gráfico 3D
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(x=100, y=430)

def createWidgets():
    global lblVideo, lblVideoDepth, lbl1, lblCajas1, lblCajas2, lblCoordABB
    fontText = font.Font(family='Helvetica', size=10, weight='bold')
    fontText1 = font.Font(family='Helvetica', size=12, weight='bold')
    fontButton = font.Font(family='Helvetica', size=12, weight='bold')

    style = ttk.Style()
    style.configure('TLabel', foreground='#000000', background='#F0F0F0', font=fontText)

    # Configurar estilos para botones
    style.configure('Green.TButton', font=fontButton, padding=6, foreground='#FFFFFF', background='#45B39D')
    style.map('Green.TButton',
              foreground=[('!disabled', '#45B39D')],
              background=[('!disabled', '#D5F5E3'), ('pressed', '#229954'), ('active', '#52BE80')])

    style.configure('Blue.TButton', font=fontButton, padding=6, foreground='#FFFFFF', background='#5DADE2')
    style.map('Blue.TButton',
              foreground=[('!disabled', '#5DADE2')],
              background=[('!disabled', '#D6EAF8'), ('pressed', '#21618C'), ('active', '#3498DB')])

    style.configure('Red.TButton', font=fontButton, padding=6, foreground='#FFFFFF', background='#C0392B')
    style.map('Red.TButton',
              foreground=[('!disabled', '#C0392B')],
              background=[('!disabled', '#FADBD8'), ('pressed', '#A93226'), ('active', '#E74C3C')])

    lblNameCamera = ttk.Label(root, text="Video en Tiempo real")
    lblNameCamera.place(x=20, y=20)

    lblNameDepth = ttk.Label(root, text="Video Profundidad")
    lblNameDepth.place(x=720, y=20)

    lblRoi1 = ttk.Label(root, text="ABB")
    lblRoi1.place(x=720, y=240)

    lblContador = ttk.Label(root, text="Información Sistema de retroalimentación", font=fontText1)
    lblContador.place(x=720, y=480)

    lblCajas1 = ttk.Label(root, text="Caja 1: No detectada")
    lblCajas1.place(x=720, y=510)

    lblCajas2 = ttk.Label(root, text="Caja 2: No detectada")
    lblCajas2.place(x=720, y=540)

    lblCoordABB = ttk.Label(root, text="Coordenadas del ABB:")
    lblCoordABB.place(x=720, y=570)

    # Crear un botón para iniciar el video y el servidor
    btn_start = ttk.Button(root,
                           text="Iniciar",
                           style='Green.TButton',
                           command=start_all)
    btn_start.place(x=750, y=700, width=120, height=40)

    # Crear un botón para enviar la información
    btn_send = ttk.Button(root,
                          text="Enviar Información",
                          style='Blue.TButton',
                          command=send_message)
    btn_send.place(x=890, y=700, width=170, height=40)

    btn_close = ttk.Button(root,
                           text="Cerrar",
                           style='Red.TButton',
                           command=exit)
    btn_close.place(x=1080, y=700, width=120, height=40)


def send_message():
    global boxes_data, conn

    if len(boxes_data) >= 1:
        # Usar las coordenadas de la primera caja detectada
        box1 = boxes_data[0]
        x1, y1, z1 = box1['x']*1000, box1['y']*1000, 160
    else:
        # Si no hay cajas detectadas, enviar (0, 0, 0)
        x1, y1, z1 = 0, 0, 0

    Moverobot = 1  # 1 para True, 0 para False  
    numBox = len(boxes_data)

    # Crear mensaje con las coordenadas actuales y variables adicionales
    message = f"{x1},{y1},{z1},{Moverobot},{numBox}\n"

    try:
        # Enviar coordenadas y variables adicionales al cliente MATLAB
        conn.send(message.encode('utf-8'))
        print('Enviado:', message.strip())
    except Exception as e:
        print("Error al enviar el mensaje:", e)

def start_all():
    global running, device, conn, x, y, z, Moverobot, numBox, depthQueue, colorQueue, disparityQueue

    if running:  # Evita que se inicie si ya está corriendo
        return

    running = True

    # Inicializar servidor TCP
    host = 'localhost'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print('Servidor Python esperando conexiones en el puerto', port)

    try:
        conn, addr = server_socket.accept()
        conn.settimeout(1.0)  # Establecer un tiempo de espera para la recepción de datos
        print('Conectado a:', addr)
        threading.Thread(target=receive_data, daemon=True).start()  # Iniciar recepción de datos en un hilo separado
    except Exception as e:
        print('Error al conectar:', e)
        running = False
        return

    # Coordenadas iniciales y variables adicionales
    x, y, z = 0, 0, 0
    Moverobot = 0  # 1 para True, 0 para False
    numBox = 0

    # Iniciar dispositivo de cámara
    try:
        device = dai.Device(pipeline, usb2Mode=True)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        update_frame()
    except RuntimeError as e:
        print(f"Error starting device: {e}")
        running = False

# Función para detener la captura de video y cerrar la conexión
def stop_all():
    global running, device, conn
    running = False
    if device:
        device.close()
        device = None
    if conn:
        conn.close()
    print("Video and server stopped")

def update_frame():
    global running, lblVideo, lblVideoDepth, lbl1, lblCajas1, lblCajas2, lblCoordABB, boxes_data, abb_data, abb_base_data, ax, canvas,filtered_x, filtered_y, filtered_z
    if not running:
        return

    try:
        depthFrame = depthQueue.get().getFrame()  # Obtener el frame de profundidad
        colorFrame = colorQueue.get().getCvFrame()  # Obtener el frame de color
        disparityFrame = disparityQueue.get().getFrame()  # Obtener el frame de disparidad

        # Realizar predicciones con YOLOv8 en el frame de color
        results = model.predict(colorFrame, conf=0.65)
        annotated_frame = results[0].plot()

        # Asegurar que el frame tenga las dimensiones correctas
        annotated_frame = cv2.resize(annotated_frame, (640, 360))

        # Resetear las listas de detecciones por clase en cada frame
        boxes_data.clear()
        abb_data.clear()
        abb_base_data.clear()

        # Procesar detecciones
        for detection in results[0].boxes:
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]
            x1, y1, x2 = int(detection.xyxy[0][0]), int(detection.xyxy[0][1]), int(detection.xyxy[0][2])
            y2 = int(detection.xyxy[0][3])
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            theta_y = (anguloFovMed / 180) * (centroid_y - 180)
            theta_x = (anguloFovMed2 / 320) * (centroid_x - 320)

            # Obtener el valor de profundidad en el centroide
            if 0 <= centroid_x < depthFrame.shape[1] and 0 <= centroid_y < depthFrame.shape[0]:
                z_value = depthFrame[centroid_y, centroid_x] / 1000
                if class_name == 'ABB':
                    if z_value > 2.5:
                        z_value = 2.5
                    h = z_value
                else:
                    h = 2.5
                x = (h * np.tan(np.deg2rad(theta_x)) * (1 / 0.8887)) - 0.0102
                x = -x
                y = h * np.tan(np.deg2rad(theta_y))
                # Dibujar un círculo en el centroide y mostrar la distancia en Z
                cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

                # Almacenar la detección en la lista adecuada
                detection_info = {"class": class_name, "x": x, "y": y, "z": z_value, "bbox": (x1, y1, x2, y2)}
                if class_name == 'BOX':
                    boxes_data.append(detection_info)
                elif class_name == 'ABB':
                    abb_data.append(detection_info)
                elif class_name == 'ABB_BASE':
                    abb_base_data.append(detection_info)

        # Calcular las coordenadas absolutas del ABB y las cajas
        if abb_base_data:
            abb_base = abb_base_data[0]
            abb_base["x"] = abb_base["x"]+0.39
            abb_base["y"] = abb_base["y"]-0.025
        else:
            abb_base = {"x": 0, "y": 0, "z": 0}
            

        for abb in abb_data:
            abb["x"] -= abb_base["x"]
            abb["y"] -= abb_base["y"]
            abb["z"] = (abb_base["z"] - abb["z"])-0.2
            #necesito hallar el angulo de abb y obtener las coordenadas de la herramienta que esta al lado
            theta=np.arctan2(abb["y"],abb["x"])
            abb["x"]=abb["x"]+0.07*np.sin(theta)
            abb["y"]=abb["y"]-0.14*np.cos(theta)
        
        for box in boxes_data:
            box["x"] -= abb_base["x"]
            box["y"] -= abb_base["y"]
            box["z"] -= abb_base["z"]
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Actualizar las coordenadas de las cajas en la interfaz
        if len(boxes_data) > 0:
            box1_coords = f"Caja 1: X: {boxes_data[0]['x']*1000:.2f}, Y: {boxes_data[0]['y']*1000:.2f}, Z: {0:.2f}"
            lblCajas1.config(text=box1_coords)
            ax.scatter(boxes_data[0]['x'], boxes_data[0]['y'], 0, c='b', marker='o')
        else:
            lblCajas1.config(text="Caja 1: No detectada")

        if len(boxes_data) > 1:
            box2_coords = f"Caja 2: X: {boxes_data[1]['x']*1000:.2f}, Y: {boxes_data[1]['y']*1000:.2f}, Z: {0:.2f}"
            lblCajas2.config(text=box2_coords)
            ax.scatter(boxes_data[1]['x'], boxes_data[1]['y'], 0, c='g', marker='o')
        else:
            lblCajas2.config(text="Caja 2: No detectada")

        # Actualizar las coordenadas del ABB en la interfaz
        if abb_data:
            abb_coords = f"ABB: X: {abb_data[0]['x']*1000:.2f}, Y: {abb_data[0]['y']*1000:.2f}, Z: {abb_data[0]['z']*1000:.2f}"
        else:
            abb_coords = "ABB: X: 0, Y: 0, Z: 0"
        lblCoordABB.config(text=abb_coords)

        # Actualizar gráfico 3D


        if abb_data:
            filtered_x.append(abb_data[0]['x'])
            filtered_y.append(abb_data[0]['y'])
            filtered_z.append(abb_data[0]['z'])

        filtered_x_value = get_filtered_value(filtered_x)
        filtered_y_value = get_filtered_value(filtered_y)
        filtered_z_value = get_filtered_value(filtered_z)
        ax.scatter(filtered_x_value, filtered_y_value, filtered_z_value, c='r', marker='o')
        canvas.draw()

        # Mostrar el frame anotado en la interfaz de tkinter
        annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=annotated_image)
        lblVideo.imgtk = imgtk
        lblVideo.configure(image=imgtk)

        # Normalizar y colorear el frame de disparidad para mejor visualización
        disparityFrame = cv2.normalize(disparityFrame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparityFrame = disparityFrame.astype(np.uint8)
        disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)
        disparityFrame = cv2.resize(disparityFrame, (320, 180))

        # Mostrar el frame de disparidad en la interfaz de tkinter
        disparity_image = Image.fromarray(disparityFrame)
        imgtk_disparity = ImageTk.PhotoImage(image=disparity_image)
        lblVideoDepth.imgtk = imgtk_disparity
        lblVideoDepth.configure(image=imgtk_disparity)

        # Asegurar que el frame de ABB tenga las dimensiones correctas
        try:
            if abb_data:
                x1, y1, x2, y2 = abb_data[0]["bbox"]
                abb_frame = colorFrame[y1 -70:y2+70, x1-100:x2+100]
                abb_frame = cv2.resize(abb_frame, (320, 180))
            else:
                abb_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        except:
            abb_frame = np.zeros((180, 320, 3), dtype=np.uint8)

        # Mostrar el frame de ABB en la interfaz de tkinter
        abb_image = Image.fromarray(cv2.cvtColor(abb_frame, cv2.COLOR_BGR2RGB))
        imgtk_abb = ImageTk.PhotoImage(image=abb_image)
        lbl1.imgtk = imgtk_abb
        lbl1.configure(image=imgtk_abb)

        if running:
            root.after(10, update_frame)
    except RuntimeError as e:
        print(f"Error during frame update: {e}")
        stop_all()

def receive_data():
    global conn, lblCoordMatlab,forma
    while running:
        try:
            data = conn.recv(1024).decode('utf-8').strip()
            if data:
                coords = data.split(',')
                if len(coords) == 3:
                    x, y, z = coords
                    formatted_coords = f"X: {x}, Y: {y}, Z: {z}"
                    lblCoordMatlab.config(text=f"Coordenadas recibidas del ABB: {formatted_coords}")
                    print(f"Coordenadas recibidas del ABB: {formatted_coords}")
        except socket.timeout:
            continue  # Intentar nuevamente
        except Exception as e:
            print(f"Error al recibir datos: {e}")
            lblCoordMatlab.config(text="Coordenadas recibidas del ABB: No disponibles")
            break

def exit():
    respuesta = messagebox.askyesno("Confirmar salida", "¿Está seguro de que desea salir?")
    if respuesta:
        stop_all()
        root.destroy()

# Llamar a las funciones para crear los widgets
createFrameZeros()
createWidgets()

# Iniciar el bucle principal de tkinter
root.mainloop()
