import cv2 as cv
import sys
import convolution as conv
from matplotlib import pyplot as plt


def main():
	if len(sys.argv) != 2:
		print 'Uso: ' + sys.argv[0] + ' <image_path>\n'
		sys.exit(0)

	image_path = sys.argv[1]

	img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

	kx, ky = cv.getDerivKernels(1,1, 3)
	res = conv.apply_separable_mask(img, kx, ky, conv.DEFAULT_BORDER)

	plt.subplot(1,1,1),plt.imshow(res,cmap = 'gray')
	plt.title('Original'), plt.xticks([]), plt.yticks([])
	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()
	
	
