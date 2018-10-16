import cv2 as cv
import sys
from matplotlib import pyplot as plt
import math


def hibridar(img1, img2, sigma_low, sigma_high): 
	size =  int(math.floor(6*sigma_high + 1))
	ksize = (size, size)
	img_high_frequency = img1 - cv.GaussianBlur(img1,ksize,sigma_high)

	size =  int(math.floor(6*sigma_low + 1))
	ksize = (size, size)	
	img_low_frequency = cv.GaussianBlur(img2,ksize,sigma_low)

	hybrid = img_high_frequency + img_low_frequency

	return hybrid

def main():
	if len(sys.argv) != 2:
		print 'Uso: ' + sys.argv[0] + ' <numero_pareja>\n'
		print "1. Submarine + Fish\n"
		print "2. Einstein + Marylin\n"
		print "3. Bird + Plane\n"
		print "4. Cat + Dog\n"
		print "5. Motorcycle + Bicycle\n"
		sys.exit(0)

	sigma_low = 0
	sigma_high = 0

	n_hybridation = int(sys.argv[1])
	
	if n_hybridation == 1:
		img1 = cv.imread("imagenes/submarine.bmp", cv.IMREAD_GRAYSCALE)
		img2 = cv.imread("imagenes/fish.bmp", cv.IMREAD_GRAYSCALE)
		sigma_high = 3
		sigma_low = 8

	elif n_hybridation == 2:
		img1 = cv.imread("imagenes/einstein.bmp", cv.IMREAD_GRAYSCALE)
		img2 = cv.imread("imagenes/marilyn.bmp", cv.IMREAD_GRAYSCALE)
		sigma_high = 2
		sigma_low = 4
		
	elif n_hybridation == 3:
		img1 = cv.imread("imagenes/bird.bmp", cv.IMREAD_GRAYSCALE)
		img2 = cv.imread("imagenes/plane.bmp", cv.IMREAD_GRAYSCALE)
		sigma_high = 2
		sigma_low = 6

	elif n_hybridation == 4:
		img1 = cv.imread("imagenes/cat.bmp", cv.IMREAD_GRAYSCALE)
		img2 = cv.imread("imagenes/dog.bmp", cv.IMREAD_GRAYSCALE)
		sigma_high = 3
		sigma_low = 5

	elif n_hybridation == 5:
		img1 = cv.imread("imagenes/bicycle.bmp", cv.IMREAD_GRAYSCALE)
		img2 = cv.imread("imagenes/motorcycle.bmp", cv.IMREAD_GRAYSCALE)
		sigma_high = 1
		sigma_low = 6

	hybrid = hibridar(img1, img2,  sigma_low,  sigma_high)

	#cv.imshow('hibrida.bmp',hybrid)

	cv.imwrite('imagenes/hybrid' + str(n_hybridation) + '.bmp', hybrid)


	plt.subplot(1,1,1),plt.imshow(hybrid,cmap = 'gray')
	plt.title('Original'), plt.xticks([]), plt.yticks([])
	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()