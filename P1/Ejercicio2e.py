import cv2 as cv
import sys
import convolution as conv
from matplotlib import pyplot as plt


def main():
	if len(sys.argv) != 4:
		print 'Uso: ' + sys.argv[0] + ' <image_path> <sigma> <n_levels>\n'
		sys.exit(0)

	img_per_row = 3

	image_path = sys.argv[1]
	sigma = float(sys.argv[2])
	n_levels = int(sys.argv[3])

	img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

	gaussianPyramid = conv.gaussian_pyramid(img,sigma, n_levels + 1)
	laplacianPyramid = conv.laplacian_pyramid(gaussianPyramid)

	n_rows = n_levels // img_per_row
	row = n_levels % img_per_row

	for i in range(0,len(laplacianPyramid)):
		plt.subplot(n_rows + n_rows,img_per_row,i + 1),plt.imshow(laplacianPyramid[i],cmap = 'gray')	
		plt.title('Level ' + str(i)), plt.xticks([]), plt.yticks([])

	plt.show()
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()
	
	
