import cv2
from ultralytics import YOLO
import depthai as dai
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.font as font
from PIL import Image, ImageTk
from tkinter import messagebox

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

def createFrameZeros():
    global lblVideo, lblVideoDepth, lbl1
    lblVideo = tk.Label(main_frame, borderwidth=2, relief="solid")
    lblVideo.place(x=20, y=50)

    # Frame original 
    frame = np.zeros([360, 640, 3], dtype=np.uint8)  # Tamaño ajustado a 640x360
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgArray = Image.fromarray(frame)
    imgTk = ImageTk.PhotoImage(image=imgArray)
    lblVideo.configure(image=imgTk)
    lblVideo.image = imgTk

    # Frame Video Profundidad
    lblVideoDepth = tk.Label(main_frame, borderwidth=2, relief="solid")
    lblVideoDepth.place(x=720, y=50)
    lblVideoDepth.configure(image=imgTk)  # Le paso las propiedades de imgTk
    lblVideoDepth.image = imgTk

    # Frame ROI 1 - ROI ABB
    lbl1 = tk.Label(main_frame, borderwidth=2, relief="solid")
    lbl1.place(x=20, y=450)
    lbl1.configure(image=imgTk)
    lbl1.image = imgTk

def createWidgets():
    global lblVideo, lblVideoDepth, lbl1, lblCajas1, lblObjetos1, lblCoordABB
    fontText = font.Font(family='Helvetica', size=10, weight='bold')
    fontText1 = font.Font(family='Helvetica', size=12, weight='bold')

    lblNameCamera = tk.Label(root, text="Video en Tiempo real", fg="#000000")
    lblNameCamera['font'] = fontText
    lblNameCamera.place(x=20, y=20)

    lblNameDepth = tk.Label(root, text="Video Profundidad", fg="#000000")
    lblNameDepth['font'] = fontText
    lblNameDepth.place(x=720, y=20)

    lblRoi1 = tk.Label(root, text="ABB", fg="#000000")
    lblRoi1['font'] = fontText
    lblRoi1.place(x=20, y=420)

    lblContador = tk.Label(root, text="Contador de cajas", fg="#000000")
    lblContador['font'] = fontText1
    lblContador.place(x=720, y=450)

    lblCajas = tk.Label(root, text="Total cajas: ", fg="#000000")
    lblCajas['font'] = fontText
    lblCajas.place(x=720, y=480)

    lblCajas1 = tk.Label(root, text="0", fg="#000000")
    lblCajas1['font'] = fontText
    lblCajas1.place(x=820, y=480)

    lblObjetos = tk.Label(root, text="Total objetos: ", fg="#000000")
    lblObjetos['font'] = fontText
    lblObjetos.place(x=720, y=510)

    lblObjetos1 = tk.Label(root, text="0", fg="#000000")
    lblObjetos1['font'] = fontText
    lblObjetos1.place(x=820, y=510)

    lblCoordABB = tk.Label(root, text="Coordenadas ABB: ", fg="#000000")
    lblCoordABB['font'] = fontText
    lblCoordABB.place(x=720, y=540)

    # Crear un botón para iniciar el video
    btn_start = tk.Button(root,
                          text="Iniciar",
                          bg='#45B39D',
                          fg='#FFFFFF',
                          width=12,
                          command=start_video)
    btn_start.place(x=720, y=600)

    # Crear un botón para detener el video
    btn_stop = tk.Button(root,
                         text="Parar",
                         bg='#5DADE2',
                         fg='#FFFFFF',
                         width=12,
                         command=stop_video)
    btn_stop.place(x=820, y=600)

    btn_close = tk.Button(root,
                          text="Cerrar",
                          bg='#C0392B',
                          fg='#FFFFFF',
                          width=12,
                          command=exit)
    btn_close.place(x=920, y=600)

# Función para iniciar la captura de video y mostrar las anotaciones
def start_video():
    global running, device, depthQueue, colorQueue, disparityQueue
    if running:  # Evita que se inicie si ya está corriendo
        return

    running = True
    try:
        device = dai.Device(pipeline, usb2Mode=True)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        update_frame()
    except RuntimeError as e:
        print(f"Error starting device: {e}")
        running = False

# Función para detener la captura de video
def stop_video():
    global running, device
    running = False
    if device:
        device.close()
        device = None
    print("Video stopped")

def update_frame():
    global running, lblVideo, lblVideoDepth, lbl1, boxes_data, abb_data, abb_base_data
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
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            x1 = x1
            x2 = x2
            centroid_x =(x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            theta_y = (anguloFovMed / 180) * (centroid_y - 180)
            theta_x = (anguloFovMed2 / 320) * (centroid_x - 320)

            # Obtener el valor de profundidad en el centroide
            if 0 <= centroid_x < depthFrame.shape[1] and 0 <= centroid_y < depthFrame.shape[0]:
                z_value = depthFrame[centroid_y, centroid_x] / 1000
                if class_name == 'ABB':
                    if z_value > 2.4:
                        z_value = 2.4
                    h = z_value
                else:
                    h = 2.4
                x = (h * np.tan(np.deg2rad(theta_x)) * (1 / 0.8887)) - 0.0102
                x = -x
                y = h * np.tan(np.deg2rad(theta_y))
                # Dibujar un círculo en el centroide y mostrar la distancia en Z
                cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"Z: {round(z_value, 2)} m", (centroid_x + 10, centroid_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.putText(annotated_frame, f"X: {round(x, 2)} m", (centroid_x + 10, centroid_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.putText(annotated_frame, f"Y: {round(y, 2)} m", (centroid_x + 10, centroid_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

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
            abb["z"] -= abb_base["z"]
        
        for box in boxes_data:
            box["x"] -= abb_base["x"]
            box["y"] -= abb_base["y"]
            box["z"] -= abb_base["z"]

        # Actualizar los contadores en la interfaz
        lblCajas1.config(text=str(len(boxes_data)))
        total_objects = len(boxes_data) + len(abb_data) + len(abb_base_data)
        lblObjetos1.config(text=str(total_objects))

        # Actualizar las coordenadas de la caja en la interfaz
        if boxes_data:
            abb_coords = f"X: {boxes_data[0]['x']:.2f}, Y: {boxes_data[0]['y']:.2f}, Z: {boxes_data[0]['z']:.2f}"
            lblCoordABB.config(text=f"Coordenadas de la caja: {abb_coords}")
        else:
            lblCoordABB.config(text="Coordenadas Caja: No detectado")

        # Mostrar el frame anotado en la interfaz de tkinter
        annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=annotated_image)
        lblVideo.imgtk = imgtk
        lblVideo.configure(image=imgtk)

        # Normalizar y colorear el frame de disparidad para mejor visualización
        disparityFrame = (disparityFrame * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)

        # Asegurar que el frame de disparidad tenga las dimensiones correctas
        disparityFrame = cv2.resize(disparityFrame, (640, 360))

        # Mostrar el frame de disparidad en la interfaz de tkinter
        disparity_image = Image.fromarray(disparityFrame)
        imgtk_disparity = ImageTk.PhotoImage(image=disparity_image)
        lblVideoDepth.imgtk = imgtk_disparity
        lblVideoDepth.configure(image=imgtk_disparity)

        # Asegurar que el frame de ABB tenga las dimensiones correctas
        if abb_data:
            x1, y1, x2, y2 = abb_data[0]["bbox"]
            abb_frame = colorFrame[y1 -70:y2+70, x1-100:x2+100]
            abb_frame = cv2.resize(abb_frame, (640, 330))
        else:
            abb_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Mostrar el frame de ABB en la interfaz de tkinter
        abb_image = Image.fromarray(cv2.cvtColor(abb_frame, cv2.COLOR_BGR2RGB))
        imgtk_abb = ImageTk.PhotoImage(image=abb_image)
        lbl1.imgtk = imgtk_abb
        lbl1.configure(image=imgtk_abb)

        if running:
            root.after(10, update_frame)
    except RuntimeError as e:
        print(f"Error during frame update: {e}")
        stop_video()

def exit():
    respuesta = messagebox.askyesno("Confirmar salida", "¿Está seguro de que desea salir?")
    if respuesta:
        root.destroy()

# Llamar a las funciones para crear los widgets
createFrameZeros()
createWidgets()

# Iniciar el bucle principal de tkinter
root.mainloop()
