import cv2
import numpy as np
import os

if __name__ == '__main__':
	dirfiles = []
	for root,directories,filenames in os.walk('assets/Emojis'):
		for filename in filenames:
			dirfiles.append(os.path.join(root,filename))
	for blockname in dirfiles:
		print(blockname)
		im = cv2.imread(blockname)
		for i in range(im.shape[0]):
			for j in range(im.shape[1]):
				if im[i,j,0] == 255 and im[i,j,1] == 255 and im[i,j,2] == 255:
					im[i,j,:] = np.array([0,0,0])
				else:
					break
			for j in range(im.shape[1]-1,0,-1):
				if im[i,j,0] == 255 and im[i,j,1] == 255 and im[i,j,2] == 255:
					im[i,j,:] = np.array([0,0,0])
				else:
					break

		for j in range(im.shape[1]):
			for i in range(im.shape[0]):
				if im[i,j,0] == 255 and im[i,j,1] == 255 and im[i,j,2] == 255:
					im[i,j,:] = np.array([0,0,0])
				else:
					break
			for i in range(im.shape[0]-1,0,-1):
				if im[i,j,0] == 255 and im[i,j,1] == 255 and im[i,j,2] == 255:
					im[i,j,:] = np.array([0,0,0])
				else:
					break
		cv2.imwrite(blockname,im)
