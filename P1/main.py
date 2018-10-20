import cv2 as cv
import sys
from matplotlib import pyplot as plt
import copy
import numpy as np
import math

#FUNCIONES EJERCICIO 1
def ejercicio1a(src, ksize, sigma):
	#Aplicar filtro gaussiano con tamano de kernel 'ksize' y sigma 'sigma'
	img =  cv.GaussianBlur(src,(ksize, ksize),sigma)

	#Mostrar imagen
	cv.imshow('Ejercicio1a',img)
	
	#Esperar pulsacion de tecla para cerrar imagen mostrada
	cv.waitKey(0)
	cv.destroyAllWindows()

def ejercicio1b(ksize, dx, dy):
	#Hallar mascaras de derivadas
	kx, ky = cv.getDerivKernels(dx, dy, ksize)

	print(kx)
	print(ky)

def ejercicio1c(src, ksize, sigma, type_border):
	#Hacer desenfcado gaussiano para suavizar la imagen (eliminar ruido)
	img =  cv.GaussianBlur(src,(ksize,ksize),sigma, borderType = type_border)

	#Aplico una laplaciana para resaltar las aristas de la imagen
	img =  cv.Laplacian(img, 2, delta = 50, borderType = type_border)

	plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
	plt.title('Ejercicio1c'), plt.xticks([]), plt.yticks([])

	plt.show()


#FUNCIONES EJERCICIO 2

def ejercicio2a(src, kx, ky, border_type):
	#Aplico a src las mascaras kx (sobre el eje X) y ky (sobre el eje Y)
	res = apply_separable_mask(src, kx, ky, border_type)

	plt.subplot(1,1,1),plt.imshow(res,cmap = 'gray')
	plt.title('Ejercicio2a'), plt.xticks([]), plt.yticks([])

	plt.show()


def ejercicio2bc(src, ksize, dx, dy, border_type):
	#Hallo las mascaras de derivadas kx y ky
	kx, ky = cv.getDerivKernels(dx, dy, ksize)

	#Aplico las mascaras kx y ky
	img = apply_separable_mask(src, kx, ky, border_type)

	plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
	plt.title('Ejercicio2bc'), plt.xticks([]), plt.yticks([])
	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()

def ejercicio2d(src, sigma, n_levels):
	img_per_row = 3

	#Aplica sobre la imagen src una piramide gaussiana con sigma sigma de n_levels niveles
	gaussianPyramid = gaussian_pyramid(src,sigma, n_levels)

	n_rows = n_levels // img_per_row
	row = n_levels % img_per_row

	for i in range(0,len(gaussianPyramid)):
		plt.subplot(n_rows + row,img_per_row,i + 1),plt.imshow(gaussianPyramid[i],cmap = 'gray')	
		plt.title('Level ' + str(i)), plt.xticks([]), plt.yticks([])

	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()

def ejercicio2e(img, sigma, n_levels):
	img_per_row = 3

	#Aplica sobre img una piramide gaussiana de n_levels + 1. Esto es debido a que la piramide laplaciana tiene un nivel menos que la gaussiana
	gaussianPyramid = gaussian_pyramid(img,sigma, n_levels + 1)

	#A partir de la piramide gausiana obtenemos la piramide laplaciana de n_levels
	laplacianPyramid = laplacian_pyramid(gaussianPyramid)

	n_rows = n_levels // img_per_row
	row = n_levels % img_per_row

	for i in range(0,len(laplacianPyramid)):
		plt.subplot(n_rows + n_rows,img_per_row,i + 1),plt.imshow(laplacianPyramid[i],cmap = 'gray')	
		plt.title('Level ' + str(i)), plt.xticks([]), plt.yticks([])

	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()

#FUNCIONES AUXILIARES
#Devuelve el tamano de kernel optimo para un valor dado de sigma
def getOptSKernelSigma(sigma):
	#Tamano de kernel optimo segun la estadistica
	size = (math.floor(6 * sigma + 1))
	
	if(size % 2 == 0):
		size = size - 1

	return (int(size), int(size))

#Aplica el kernel 'kx', sobre la imagen 'img'
def apply_kernel_x(img,kx):
	dim_img = img.shape
	dim_kernel = len(kx)

	#Dimension del borde necesaria para que el tamano de la imagen no se vea alterada
	kernel_border = (dim_kernel - 1) // 2
	
	#Inicializamos res a 0
	res = np.zeros(dim_img)
	
	#Aplicamos la formula de convolucion teniendo en cuenta solo las columnas de la imagen
	for i in range(0, dim_img[0]):
		for j in range(kernel_border, dim_img[1] - kernel_border):
			for v in range(-kernel_border, kernel_border+1):	
				res[i,j] = res[i,j] + kx[v] * img[i, j - v]


	return res

#Aplica el kernel 'ky' sobre la imagen 'img'
def apply_kernel_y(img,ky):
	dim_img = img.shape
	dim_kernel = len(ky)

        #Dimension del borde necesaria para que el tamano de la imagen no se vea alterada
	kernel_border = (dim_kernel - 1) // 2

        #Inicializamos res a 0
	res = np.zeros(dim_img)

        #Aplicamos la formula de convolucion teniendo en cuenta solo las filas de la imagen
	for i in range(kernel_border, dim_img[0] - kernel_border):
		for j in range(0, dim_img[1]):
			for u in range(-kernel_border, kernel_border+1):
				res[i,j] = res[i,j] + ky[u] * img[i - u,j]

	return res

def apply_separable_mask(img, kx, ky, borderType = cv.BORDER_DEFAULT):
	tmp = copy.deepcopy(img)
	ksize = (len(kx),len(ky))
	tmp = make_borders(tmp, ksize, borderType)
	tmp = apply_kernel_x(tmp, kx)
	tmp = apply_kernel_y(tmp, ky)

	res = np.delete(tmp, range(0,(len(ky) - 1)//2),0)
	res = np.delete(res, range(res.shape[0] - (len(ky) - 1)//2,res.shape[0]),0)
	res = np.delete(res, range(0,(len(kx) - 1)//2),1)
	res = np.delete(res, range(res.shape[1] - (len(kx) - 1)//2,res.shape[1]),1)

	
	return res


#BORDERS
def make_borders(img, ksize, type_border = cv.BORDER_DEFAULT):
	if(type_border == cv.BORDER_DEFAULT):
		res = make_default_borders(img, ksize)

	elif(type_border == cv.BORDER_REFLECT):
		res = make_reflex_borders(img, ksize)

	

	return res



#Anade bordes con valor 0 a la imagen
def make_default_borders(img, ksize):
	shape = img.shape

	#Calculamos una columna de 0 con igual dimension a una columna de la imagen
	newCol = np.zeros((shape[0], (ksize[0] - 1) // 2))

	#Calculamos una fila de 0 de igual dimension a la imagen, teniendo en cuenta la columna anteriormente calculada
	newRow = np.zeros(((ksize[0] - 1) // 2, shape[1] + newCol.shape[1]))

	#Anadimos la columna a la imagen en la parte izda
	res = np.column_stack((newCol,img))

	#Anadimos la fila a la imagen en la parte superior
	res = np.row_stack((newRow, res))

	shape = res.shape

	#Volvemos a calcular la columna con los cambios hechos
	newCol = np.zeros((shape[0], (ksize[0] - 1) // 2))

	#Volvemos a calcular la fila con los cambios hechos anteriormente y teniendo en cuenta la nueva columna
	newRow = np.zeros(((ksize[0] - 1) // 2, shape[1] + newCol.shape[1]))

	#Anadimos la columna a la imagen en la parte derecha
	res = np.column_stack((res, newCol))

	#Anadimos la fila a la imagen en la parte inferior
	res = np.row_stack((res, newRow))

	return res

#Anade bordes en espejo
def make_reflex_borders(img, ksize):
	res = copy.deepcopy(img)

	shape = res.shape

	#Calculamos las columnas reflejadas por la parte izquierda
	newCol = res[0:shape[0], 0:(ksize[1] - 1) // 2]

	#Anadimos las columnas a la parte izquierda
	res = np.column_stack((newCol[:,::-1], res))

	shape = res.shape
	
	#Calculamos las columnas reflejadas por la parte derecha
	newCol = res[0:shape[0], shape[1] - (ksize[1] - 1) // 2:shape[1]]

	#Anadimos las columnas a la parte derecha
	res = np.column_stack((res, newCol[:,::-1]))

	shape = res.shape

	#Teniendo en cuenta las columnas anadidas, calculamos las filas reflejaas por la parte superior 
	newRow = res[0:(ksize[0] - 1) // 2,0:shape[1]]
	
	#Anadimos las filas a la parte superior
	res = np.row_stack((newRow[::-1], res))

	shape = res.shape

	#Teniendo en cuenta las columnas anadidas, calculamos las filas reflejaas por la parte inferior
	newRow = res[shape[0] - (ksize[0] - 1) // 2:shape[0],0:shape[1]]

	#Anadimos las filas a la parte inferior
	res = np.row_stack((res, newRow[::-1]))

	return res
		
#PIRAMIDE GAUSSIANA
def gaussian_pyramid(src, sigma, n_levels):
	#Tamano optimo del kernel en funcion del sigma
	ksize = getOptSKernelSigma(sigma)

	#Copamos src en res ya que los objetos mutables en python se pasan por referencia y operar con src directamente lo alteraria fuera del ambito de la funcion
	res = copy.deepcopy(src)

	#Guardamos la imagen original dentro de res (No cuenta como nivel de la piramide)
	gaussianPyramid = [res]

	#Empezamos a rellenar la piramide
	for i in range(0, n_levels):
		#Sacamos un kernel gaussiano
		kernel = cv.getGaussianKernel(ksize[0], sigma)

		#Aplicamos el kernel sobre la imagen resultante
		res = apply_separable_mask(res, kernel, kernel, cv.BORDER_REFLECT)

		#Eliminamos las filas pares de la imagen
		res = cv.pyrDown(res)
		
		#Anadimos el resultado a la piramide
		gaussianPyramid.append(res)
	
	return gaussianPyramid

#PIRAMIDE LAPLACIANA
def laplacian_pyramid(gaussianPyramid):
	laplacianPyramid = []

	#Empezamos a rellenar la piramide
	for i in range(1, len(gaussianPyramid)):
		#Anade filas intermedias e interpola en el nivel i de la piramide gaussiana
		res = cv.pyrUp(gaussianPyramid[i])

		#Tratamiento de columnas pares
		if res.shape[0] != gaussianPyramid[i-1].shape[0]:
			res = np.delete(res, res.shape[0]-1,0)

		#Tratamiento de filas pares
		if res.shape[1] != gaussianPyramid[i-1].shape[1]:
			res = np.delete(res, res.shape[1]-1,1)

		#Resta la imagen resultante con el nivel anterior de la gaussiana
		res = gaussianPyramid[i-1] - res

		#Anade a la piramide
		laplacianPyramid.append(res)

	#Invierte la piramide
	laplacianPyramid.reverse()

	return laplacianPyramid

#EJERCICIO 3

#FUNCIONES AUXILIARES
def hibridar(img1, img2, sigma_low, sigma_high): 
	size =  getOptSizeKernel(sigma_high)
	
	#Le quito a la imagen las frecuencias bajas
	img_high_frequency = img1 - cv.GaussianBlur(img1,size[0],sigma_high)

	size =  getOptSizeKernel(sigma_low)

	#Le quito a la imagen las frecuencias altas
	img_low_frequency = cv.GaussianBlur(img2,size[0],sigma_low)

	#Sumo ambas imagenes para obtener la imagen hibrida
	hybrid = img_high_frequency + img_low_frequency

	plt.subplot(1,3,1),plt.imshow(img_low_frequency,cmap = 'gray')
	plt.title('Baja'), plt.xticks([]), plt.yticks([])

	plt.subplot(1,3,2),plt.imshow(img_high_frequency,cmap = 'gray')
	plt.title('Alta'), plt.xticks([]), plt.yticks([])

	plt.subplot(1,3,3),plt.imshow(hybrid,cmap = 'gray')
	plt.title('Hibrida'), plt.xticks([]), plt.yticks([])
	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()

##############################################MAIN################################################################
def main():

	#COMIENZA EJERCICIO 1
	print('Ejercicio 1')
	#COMIENZA APARTADO A
	ruta_imagen = 'imagenes/bird.bmp'
	print('Apartado A')
	#Leer la imagen de la ruta definida
	img = cv.imread(ruta_imagen, cv.IMREAD_GRAYSCALE) 

	print('Primera parte del experimento')
	print('ksize = 3, sigma = 3')
	ejercicio1a(img, 3, 3)
	
	print('ksize = 3, sigma = 4')
	ejercicio1a(img, 3, 4)

	print('ksize = 3, sigma = 5')
	ejercicio1a(img, 3, 5)


	print('Segunda parte del experimento')
	print('ksize = 31, sigma = 5')
	ejercicio1a(img, 31, 5)

	print('ksize = 31, sigma = 3')
	ejercicio1a(img, 31, 3)

	print('ksize = 31, sigma = 1')
	ejercicio1a(img, 31, 1)


	#COMIENZA APARTADO B
	print('Apartado B')
	print('ksize: 3, dx: 1,  dy: 1')
	ejercicio1b(3,1,1)

	print('ksize: 3, dx: 1,  dy: 2')
	ejercicio1b(3,1,2)


	print('ksize: 3, dx: 2,  dy: 1')
	ejercicio1b(3,2,1)

	print('ksize: 3, dx: 2,  dy: 2')
	ejercicio1b(3,2,2)


	#COMIENZO APARTADO C
	print('Apartado C')
	ruta_imagen = 'imagenes/dog.bmp'
	img = cv.imread(ruta_imagen, cv.IMREAD_GRAYSCALE) 

	print('ksize = 7, sigma = 1, borde = default')
	ejercicio1c(img,7,1,cv.BORDER_DEFAULT)

	print('ksize = 19, sigma = 3, borde = default')
	ejercicio1c(img,19,3,cv.BORDER_DEFAULT)

	print('ksize = 7, sigma = 1, borde = replicate')
	ejercicio1c(img,7,1,cv.BORDER_REPLICATE)

	print('ksize = 19, sigma = 3, borde = replicate')
	ejercicio1c(img,19,3,cv.BORDER_REPLICATE )



	#EJERCICIO 2
	print('Ejercicio 2')
	ruta_imagen = 'imagenes/cat.bmp'
	img = cv.imread(ruta_imagen, cv.IMREAD_GRAYSCALE)
	
	#APARTADO A
	print('Apartado A')
	print('kx = [-1,0,1], ky = [1,-2,1], borde = reflect')
	kx = np.array([-1,0,1])
	ky = np.array([1,-2,1])

	ejercicio2a(img, kx, ky, cv.BORDER_REFLECT)


	#APARTADO B
	print('Apartado B')
	print('ksize = 3, dx = 1, dy = 1, borde = default')
	ejercicio2bc(img, 3, 1, 1, cv.BORDER_DEFAULT)

	#APARTADO C
	print('Apartado C')
	print('ksize = 3, dx = 2, dy = 2, borde = default')
	ejercicio2bc(img, 3, 2, 2, cv.BORDER_DEFAULT)

	#APARTADO D
	print('Apartado D')
	print('NLevels: 4, sigma: 3')
	ejercicio2d(img, 3, 4)

	#APARTADO E
	print('Apartado E')
	print('NLevels: 4, sigma: 3')
	ejercicio2e(img, 3, 4)

	#EJERCICIO 3
	print('Ejercicio 3')

	print('Hibrida: Submarine + Fish')
	img1 = cv.imread("imagenes/submarine.bmp", cv.IMREAD_GRAYSCALE)
	img2 = cv.imread("imagenes/fish.bmp", cv.IMREAD_GRAYSCALE)
	sigma_high = 3
	sigma_low = 8
	hibridar(img1, img2, sigma_low, sigma_high)

	print('Hibrida: Plane + Bird')
	img1 = cv.imread("imagenes/bird.bmp", cv.IMREAD_GRAYSCALE)
	img2 = cv.imread("imagenes/plane.bmp", cv.IMREAD_GRAYSCALE)
	sigma_high = 2
	sigma_low = 6
	hibridar(img1, img2, sigma_low, sigma_high) 

	print('Hibrida: Dog + Cat')
	img1 = cv.imread("imagenes/cat.bmp", cv.IMREAD_GRAYSCALE)
	img2 = cv.imread("imagenes/dog.bmp", cv.IMREAD_GRAYSCALE)
	sigma_high = 3
	sigma_low = 5
	hibridar(img1, img2, sigma_low, sigma_high) 

	print('Hibrida: Einstein + Marilyn')
	img1 = cv.imread("imagenes/einstein.bmp", cv.IMREAD_GRAYSCALE)
	img2 = cv.imread("imagenes/marilyn.bmp", cv.IMREAD_GRAYSCALE)
	sigma_high = 2
	sigma_low = 4
	hibridar(img1, img2, sigma_low, sigma_high)
	
	print('Hibrida: Motorcycle + Bicycle')
	img1 = cv.imread("imagenes/bicycle.bmp", cv.IMREAD_GRAYSCALE)
	img2 = cv.imread("imagenes/motorcycle.bmp", cv.IMREAD_GRAYSCALE)
	sigma_high = 1
	sigma_low = 6
	hibridar(img1, img2,  sigma_low,  sigma_high)

if __name__ == '__main__':
	main()
