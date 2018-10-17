import cv2 as cv
import sys
import convolution as conv
import numpy as np
from matplotlib import pyplot as plt


def main():
	if len(sys.argv) != 3:
		print ('Uso: ' + sys.argv[0] + ' <image_path> <write_bit>\n')
		sys.exit(0)

	image_path = sys.argv[1]

	kernelX = np.array([-1,0,1])
	kernelY = np.array([1,-2,1])
	img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
	res = conv.apply_separable_mask(img, kernelX, kernelY, conv.REFLEX_BORDER)


	if(sys.argv[2] == "1"):
		cv.imwrite("Ejercicio2a.bmp", res)
	plt.subplot(1,1,1),plt.imshow(res,cmap = 'gray')
	plt.title('Original1'), plt.xticks([]), plt.yticks([])


	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()
	
	
