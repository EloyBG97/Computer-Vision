import cv2 as cv
import sys
from matplotlib import pyplot as plt

def main():

	if len(sys.argv) != 5:
		print 'Uso: ' + sys.argv[0] + ' <image_path> <sigma> <border_type>\n'
		sys.exit(0)

	image_path = sys.argv[1]
	ksize = int(sys.argv[2])
	sigma = int(sys.argv[3])
	border_type = int(sys.argv[4]) 

	img = cv.imread(image_path, cv.IMREAD_COLOR)

	img =  cv.GaussianBlur(img,(ksize,ksize),sigma)
	img =  cv.Laplacian(img, 2, delta = 50)

	plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
	plt.title('Original'), plt.xticks([]), plt.yticks([])

	plt.show()


if __name__ == '__main__':
	main()

'''
import cv2 as cv

def main():
	name_im = raw_input("Introduzca el nombre de la imagen: ")
	img = cv.imread("imagenes/"+ name_im + ".bmp", cv.IMREAD_COLOR)
 

        img1 =  cv.GaussianBlur(img,(3,3),1)
	img2 =  cv.Laplacian(img1, 2, delta = 50)

	cv.imwrite("imagenes/" + name_im + "_L_1.bmp",img2)

	img1 =  cv.GaussianBlur(img,(3,3),3)
	img2 =  cv.Laplacian(img1, 2, delta = 50)

	cv.imwrite("imagenes/" + name_im + "_L_3.bmp",img2)


if __name__ == '__main__':
	main()

import cv2 as cv
import sys

def main():
	if len(sys.argv) != 5:
		print 'Uso: ' + sys.argv[0] + ' <image_path> <kernel_size> <sigma> <border_type>\n'
		sys.exit(0)

	image_path = sys.argv[1]
	ksize = int(sys.argv[2])
	sigma = int(sys.argv[3])
	border_type = int(sys.argv[4])

	img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

	img1 =  cv.GaussianBlur(img,(ksize,ksize),sigma, borderType = border_type)
	img2 =  cv.Laplacian(img1, 2, delta = 50, borderType = border_type)

	cv.imshow('laplacian',img2)
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()

'''
