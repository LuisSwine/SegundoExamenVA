#Bibliotecas
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from tkinter import *
from tkinter import messagebox as MessageBox

def openImage(ruta):
    return cv2.imread(ruta)

def showImage(imagen, titulo):
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    pass

#k-means
def kmeans(imagen):
    
    #Extraemos los valores de cada canal
    valores_pixeles = imagen.reshape((-1, 3))
    valores_pixeles = np.float32(valores_pixeles)
    
    #Establecemos el criterio de parada
    criterio_stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    #establecemos el numero de iteraciones
    n_intentos = 30
    
    #Establecemos una inicializacion de valores para los centroides de manera aleatoria
    ini_centroides = cv2.KMEANS_RANDOM_CENTERS
    
    #Ejecutamos kmeans
    ret, labels, centers = cv2.kmeans(valores_pixeles, 4, None, criterio_stop, n_intentos, ini_centroides)
    
    centers = np.uint8(centers)
    info_segmentada = centers[labels.flatten()]
    
    img_segmentada = info_segmentada.reshape(imagen.shape)
    
    return img_segmentada
    

#Procesos para Watershed
#Grayscale
def color_to_grayscale(imagen):
    filas, columnas, profundidad = imagen.shape

    img_salida = np.zeros((filas, columnas), np.uint8)

    for i in range(filas):
        for j in range(columnas):
            pixel = imagen[i][j]
            azul = pixel[0]
            verde = pixel[1]
            rojo = pixel[2]

            img_salida[i][j] = 0.299 * azul + 0.587 * verde + 0.11 * rojo
            
    return img_salida

def binarizarImg(imagen, valor):
    filas, columnas = imagen.shape
    img_salida = np.zeros((filas, columnas), np.uint8)

    for i in range(filas):
        for j in range(columnas):
            if imagen[i][j] <= valor:
                img_salida[i][j] = 255
            else:
                img_salida[i][j] = 0
    return img_salida

def distancia_2_puntos(p1, p2):
    dif1 = pow(p2[0]-p1[0],2)
    dif2 = pow(p2[1]-p1[1],2)
    result = math.sqrt(dif1 + dif2)
    print(result)
    return result

def display(img, cmap='gray'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

def get_red_chanel(imagen):
    R, _, _ = cv2.split(imagen)
    return R

def coords_2_vector_objeto_4(coordenadas, imagen):
    cant_coordenadas = len(coordenadas)
    mitad = cant_coordenadas // 2
    octavo = cant_coordenadas // 8
    
    #Tomamos 4 puntos
    a1, b1 = coordenadas[cant_coordenadas - octavo][0]
    a2, b2 = coordenadas[mitad-octavo][0]

    cv2.line(imagen, (a1,b1), (a2,b2), (255,0,0), 3)
    
    #Distancia 1
    dist = distancia_2_puntos((a1,b1), (a2,b2))
    
    return (imagen, dist, (a1,b1), (a2,b2))

def coords_2_vector_objeto_2(coordenadas, imagen):
    cant_coordenadas = len(coordenadas)
    mitad = cant_coordenadas // 2
    octavo = cant_coordenadas // 8
    doceavo = cant_coordenadas // 11
    
    
    #Tomamos 4 puntos
    a1, b1 = coordenadas[cant_coordenadas-octavo - doceavo][0]
    a2, b2 = coordenadas[mitad-octavo - doceavo][0]

    cv2.line(imagen, (a1,b1), (a2,b2), (255,0,0), 3)
    
    #Distancia 2
    dist = distancia_2_puntos((a1,b1), (a2,b2))
    
    return (imagen, dist, (a1,b1), (a2,b2))

#Find Couontour
if __name__ == '__main__':
    print("SEGUNDA EVALUACION PRACTICA")

    #Primero abrimos la imagen
    img_original = openImage('Jit1.JPG')

    #Primero segmentamos por kmeans
    img_blur_color = cv2.medianBlur(img_original, 13)
    
    #Segmentamos por K-means
    img_kmeans = kmeans(img_blur_color)
    
    #Segmentamos por color
    img_rojo = get_red_chanel(img_kmeans) 
    
    #Desenfocamos la imagen
    img_blur = cv2.medianBlur(img_rojo, 15)
    
    #guardamos el valor de cada cluster
    valores_clusters = np.unique(img_rojo)
    
    #Binarizamos
    img_bin = binarizarImg(img_blur, valores_clusters[0])
    
    #Extraemos las coordenadas de los contornos con la funcion de OpenCV
    #La funcion recibe como paramentros una imagen binarizada y extrae los contronos de la misma devolviendo un arreglo con todos los puntos
    #del contorno para posteriormente poderlos analizar de manera más detenida
    contornos,hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    indexes = []
    for i in range(len(contornos)):
        if len(contornos[i]) < 700:
            indexes.append(i)  
    contornos = np.delete(contornos, indexes)
    
    #Creamos tres copiar de la imagen original para trazar en ellas toda la informacion de resultados
    img_copy = np.copy(img_original)
    img_copy_1 = np.copy(img_original)
    img_copy_2 = np.copy(img_original)
    
    #Trazamos los contornos detectados en la primera copia de la imagen
    cv2.drawContours(img_copy, contornos, -1, (0, 255, 0), 1)
    
    #Extraemos los puntos, las distancias y los trazamos en la copia correspondiente
    img_painted1, dist1, punto1, punto2 = coords_2_vector_objeto_4(contornos[0], img_copy_1)
    img_painted2, dist2, punto3, punto4 = coords_2_vector_objeto_2(contornos[2], img_copy_2)
    
    #Imprmimos todos los resultados de acuerdo al procesamiento realizado
    #1. Imagen Original abierta con OpenCv
    display(img_original)
    
    #2. Realizamos un suavizado o desenfoque de la imagen apoyandonos del suavizado de OpenCV con un kernel de 13
    #   Este paso se realiza con el objetivo de disminuir la prescencia de bordes en la imagen, puede realizarse con cualquier metodo de suavizado
    #   pero esta funcion ofrece los resultados más optimos en un tiempo considerable. 
    display(img_blur_color)
    
    #3. Aplicamos K-means a la imagen suavizada
    #   Dado que la precscencia de bordes esta suavizada, para k-means es más sencillo diferenciar los jitomates de las rocas de la imagen
    #   Estamos utilizando 4 clusters. Se ha optado por realizar esta segmentación como primer paso importante pues es alto el contraste del rojo
    #   de los tomates contra el blanco de las piedras. Al usar 4 clusters y después del suavizado se segmentan bien los jitomates
    display(img_kmeans)
    
    #4. Ya que tenemos esta segmentación obtenemos una imagen con tres tonos de color de los cuales uno de se encuentra con gran saturacion de rojo
    #   Por lo que se decide realizar otra segmentación, esta vez por canales de color, extrayendo el canal rojo, al extraerlo, obtenemos una imagen
    #   en escala de grises
    display(img_rojo)
    
    #5. Procedemos a realizar un suavizado de la imagen de canal rojo, de nuevo utilizamos la funcion de opencv
    display(img_blur)
    
    #6. Binarizamos la imagen, la imagen suavizada tiene bien remarcado el valor del rojo, y contiene 4 intensidades de gris, como el rojo es el valor de gris más bajo
    #   pues es el predominante en su canal de color, procedemos a extraer con np.unique() los cuatro valores y seleccionamos el menor y con ese
    #   valor binarizamos, de esta manera nos quedan bien marcados los jitomates
    display(img_bin)
    
    #7. Extraemos los contornos, para este paso podría haberse utilizado un algoritmo como watershed, o LoG para obterner los bordes, despues iterar
    #   la imagen para encontrar todas las coordenadas de los bordes de cada uno de los elementos encontrados con el procesamiento en la imagen
    #   sin embargo, después de intentarlo un tiempo, encontre la funcion findContours() de opencv que realiza un trabajo excelente sobre una imagen binarizada
    #   y que retorna directamente en un array todos los contornos de la imagen y sus coordenadas, por lo que opté por utilizar la función
    #   despues dibujamos los bordes sobre la imagen y la mostramos
    display(img_copy)
    
    #8. Ya teniendo los contornos y todas las coordenadas lo siguiente era poder encontrar los puntos de interes de la imagen.
    #   No fue una tarea sencilla, pues a decir verdad, no encontraba una manera analítica de hacerlo, así de que después de un timepo 
    #   opte por identificar el punto de partida de la identificación de contornos de opencv y sobre ese contorno encontrar los puntos de interes
    #   para después operarlos y calcular la distancia enntre los puntos
    display(img_copy_2)
    MessageBox.showinfo("Informacion Jitomate 2", (f'Puntos del jitomate: \n'+
                        f'A: x={punto3[0]}, y={punto3[1]} \n' +
                        f'B: x={punto4[0]}, y={punto4[1]} \n' +
                        f'Distancia entre los puntos: {dist2}'))
    
    display(img_copy_1)
    MessageBox.showinfo("Informacion Jitomate 4", (f'Puntos del jitomate: \n'+
                        f'A: x={punto1[0]}, y={punto1[1]} \n' +
                        f'B: x={punto2[0]}, y={punto2[1]} \n' +
                        f'Distancia entre los puntos: {dist1}'))
    
    #8. Por ultimo hacemos un OR entre las dos imagenes para obetener la imagen final
    img_final = cv2.bitwise_or(img_copy_1, img_copy_2)
    display(img_final)