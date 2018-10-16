import copy
import numpy as np
import math
import sys
import cv2 as cv

#CONSTANTES
DEFAULT_BORDER = 0
REFLEX_BORDER = 1

#Devuelve el tamano de kernel optimo para un valor dado de sigma
def getOptSKernelSigma(sigma):
	size = (math.floor(6 * sigma + 1))
	
	if(size % 2 == 0):
		size = size - 1

	return (int(size), int(size))

#Aplica el kernel, kx, sobre la imagen, img
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

#Aplica el kernel, ky, sobre la imagen, img
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

def apply_separable_mask(img, kx, ky, borderType = DEFAULT_BORDER):
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
def make_borders(img, ksize, type_border = DEFAULT_BORDER):
	if(type_border == DEFAULT_BORDER):
		res = make_default_borders(img, ksize)

	elif(type_border == REFLEX_BORDER):
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
	gaussianPyramid = []

	for i in range(0, n_levels):
		kernel = cv.getGaussianKernel(3, sigma)
		res = apply_separable_mask(res, kernel, kernel, REFLEX_BORDER)
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



