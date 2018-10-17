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
	kx, ky = cv.getDerivKernels(dx, dy, ksize)

	print(kx)
	print(ky)

def ejercicio1c(src, ksize, sigma, type_border):

	img =  cv.GaussianBlur(src,(ksize,ksize),sigma, borderType = type_border)
	img =  cv.Laplacian(img, 2, delta = 50, borderType = type_border)

	plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
	plt.title('Ejercicio1c'), plt.xticks([]), plt.yticks([])

	plt.show()


#FUNCIONES EJERCICIO 2

def ejercicio2a(src, kx, ky, border_type):
	res = apply_separable_mask(src, kx, ky, border_type)

	plt.subplot(1,1,1),plt.imshow(res,cmap = 'gray')
	plt.title('Ejercicio2a'), plt.xticks([]), plt.yticks([])

	plt.show()

def ejercicio2bc(src, ksize, dx, dy, border_type):
	kx, ky = cv.getDerivKernels(dx, dy, ksize)
	img = apply_separable_mask(src, kx, ky, border_type)

	plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
	plt.title('Ejercicio2bc'), plt.xticks([]), plt.yticks([])
	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()

def ejercicio2d(src, sigma, n_levels):
	img_per_row = 3
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
	gaussianPyramid = gaussian_pyramid(img,sigma, n_levels + 1)
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
	size = (math.floor(6 * sigma + 1))
	
	if(size % 2 == 0):
		size = size - 1

	return (int(size), int(size))

#Aplica el kernel 'kx', sobre la imagen 'img'
def apply_kernel_x(img,kx):
	dim_img = img.shape
	dim_kernel = len(kx)
	kernel_border = (dim_kernel - 1) // 2
	
	res = np.zeros(dim_img)

	for i in range(0, dim_img[0]):
		for j in range(kernel_border, dim_img[1] - kernel_border):
			for v in range(-kernel_border, kernel_border+1):	
				res[i,j] = res[i,j] + kx[v] * img[i, j - v]


	return res

#Aplica el kernel 'ky' sobre la imagen 'img'
def apply_kernel_y(img,ky):
	dim_img = img.shape
	dim_kernel = len(ky)
	kernel_border = (dim_kernel - 1) // 2

	res = np.zeros(dim_img)

	
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


def make_default_borders(img, ksize):
	shape = img.shape
	newCol = np.zeros((shape[0], (ksize[0] - 1) // 2))
	newRow = np.zeros(((ksize[0] - 1) // 2, shape[1] + newCol.shape[1]))


	res = np.column_stack((newCol,img))
	res = np.row_stack((newRow, res))

	shape = res.shape
	newCol = np.zeros((shape[0], (ksize[0] - 1) // 2))
	newRow = np.zeros(((ksize[0] - 1) // 2, shape[1] + newCol.shape[1]))

	res = np.column_stack((res, newCol))
	res = np.row_stack((res, newRow))

	return res

def make_reflex_borders(img, ksize):
	res = copy.deepcopy(img)

	shape = res.shape


	newCol = res[0:shape[0], 0:(ksize[1] - 1) // 2]
	res = np.column_stack((newCol[:,::-1], res))

	shape = res.shape

	newCol = res[0:shape[0], shape[1] - (ksize[1] - 1) // 2:shape[1]]
	res = np.column_stack((res, newCol[:,::-1]))

	shape = res.shape

	newRow = res[0:(ksize[0] - 1) // 2,0:shape[1]]
	res = np.row_stack((newRow[::-1], res))

	shape = res.shape

	newRow = res[shape[0] - (ksize[0] - 1) // 2:shape[0],0:shape[1]]
	res = np.row_stack((res, newRow[::-1]))


	return res
		
#PIRAMIDE GAUSSIANA
def gaussian_pyramid(src, sigma, n_levels):
	ksize = getOptSKernelSigma(sigma)
	res = copy.deepcopy(src)
	gaussianPyramid = [res]

	

	for i in range(0, n_levels):
		kernel = cv.getGaussianKernel(3, sigma)
		res = apply_separable_mask(res, kernel, kernel, cv.BORDER_REFLECT)
		res = cv.pyrDown(res)
		gaussianPyramid.append(res)
	
	return gaussianPyramid

#PIRAMIDE LAPLACIANA
def laplacian_pyramid(gaussianPyramid):
	laplacianPyramid = []

	for i in range(1, len(gaussianPyramid)):
		res = cv.pyrUp(gaussianPyramid[i])

		if res.shape[0] != gaussianPyramid[i-1].shape[0]:
			res = np.delete(res, res.shape[0]-1,0)

		if res.shape[1] != gaussianPyramid[i-1].shape[1]:
			res = np.delete(res, res.shape[1]-1,1)

		res = gaussianPyramid[i-1] - res
		laplacianPyramid.append(res)

	laplacianPyramid.reverse()

	return laplacianPyramid

#EJERCICIO 3

#FUNCIONES AUXILIARES
def hibridar(img1, img2, sigma_low, sigma_high): 
	size =  int(math.floor(6*sigma_high + 1))
	ksize = (size, size)
	img_high_frequency = img1 - cv.GaussianBlur(img1,ksize,sigma_high)

	size =  int(math.floor(6*sigma_low + 1))
	ksize = (size, size)	
	img_low_frequency = cv.GaussianBlur(img2,ksize,sigma_low)

	hybrid = img_high_frequency + img_low_frequency

	plt.subplot(1,1,1),plt.imshow(hybrid,cmap = 'gray')
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
