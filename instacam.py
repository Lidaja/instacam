from InstagramAPI import InstagramAPI
import numpy as np
import numpy.random as npr
import cv2
import time
import datetime
import os
import copy


user,pwd = 'ljinstacam', 'instacampassword'

faceCascade = cv2.CascadeClassifier(os.getcwd()+'/assets/haarcascade.xml')
i = 0

def get_mouse(event,x,y,flags,param):
	global mouseX,mouseY
	global i
	global uploadSuccess
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y
		if mouseY >= arrowOffsetY and mouseY <= arrowOffsetY+right.shape[0]:
			if mouseX >= arrowOffsetXR and mouseX <= arrowOffsetXR+right.shape[1]:
				i+=1
			if mouseX >= arrowOffsetXL-left.shape[1] and mouseX <= arrowOffsetXL:
				i-=1
		if uploadSuccess == False and mouseY >= uploadOffsetY and mouseY <= uploadOffsetY+upload.shape[0]:
			if mouseX >= uploadOffsetX and mouseX <= uploadOffsetX+upload.shape[1]:
				timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mm%Ss')
				photo_path =os.getcwd()+'/'+timestamp+'.jpg'
				print(cv2.imwrite(photo_path,dst))
				caption = "Sample photo"
				print("START UPLOAD")
				InstagramAPI.uploadPhoto(photo_path, caption = caption)
				print("SUCCESS")
				uploadSuccess = True
		elif uploadSuccess == True:
			uploadSuccess = False

def load_image(filename):
	return cv2.imread(os.getcwd()+'/'+filename,cv2.IMREAD_UNCHANGED)

def blur(img):
	kernel = np.ones((3,3),np.float32)/25
	return cv2.filter2D(img,-1,kernel)

def edge(img):
	return to3D(cv2.Canny(grayscale(img),100,200))

def normal(img):
	return img

def grayscale(img):
	return to3D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def to3D(img):
	return np.repeat(img[:,:,np.newaxis],3,axis=2)

def overlay(img, over, startX, startY):
	overAlpha = over[:,:,3]/255
	imageAlpha = 1 - overAlpha
	overlayMult = np.multiply(over[:,:,0:3],to3D(overAlpha))
	imageSplice = img[startY:startY+over.shape[0],startX:startX+over.shape[1],0:3]
	imageMult = np.multiply(imageSplice,to3D(imageAlpha))
	toOverlay = np.add(overlayMult,imageMult)
	img[startY:startY+over.shape[0],startX:startX+over.shape[1],0:3] = toOverlay
	return img

def detection(img):
	imgcopy = copy.deepcopy(img)
	gray = grayscale(imgcopy)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	for (x, y, w, h) in faces:
		cv2.rectangle(imgcopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
	return imgcopy

def create_map(dirname):
	dirfiles = []
	for root,directories,filenames in os.walk('assets/'+dirname):
		for filename in filenames:
			dirfiles.append(os.path.join(root,filename))

	blockMap = {}
	for blockname in dirfiles:
		block = load_image(blockname)[:,:,:3]
		rAvg = np.mean(block[:,:,0])
		bAvg = np.mean(block[:,:,1])
		gAvg = np.mean(block[:,:,2])
		blockMap[(rAvg,bAvg,gAvg)] = block
	return blockMap

def blockify(im):
	blockMap = emojiMap
	blockSize = 40
	numBlocksWidth = im.shape[0]//blockSize
	numBlocksHeight = im.shape[1]//blockSize
	blockKeys = blockMap.keys()
	canvas = np.zeros(im.shape,dtype=np.uint8)
	mapMat = np.array(list(blockKeys))
	for i in range(numBlocksHeight):
		for j in range(numBlocksWidth):
			block = im[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize,:]
			rAvg = np.mean(block[:,:,0])
			bAvg = np.mean(block[:,:,1])
			gAvg = np.mean(block[:,:,2])
			means = np.array([[rAvg,bAvg,gAvg]])
			meanMat = np.repeat(means,len(blockKeys),0)
			diffMat = np.sum(np.square(np.subtract(meanMat,mapMat)),1)
			minIndex = np.argmin(diffMat)
			key = mapMat[minIndex,:]
			toInsert = cv2.resize(blockMap[tuple(key)],(block.shape[0],block.shape[1]))
			canvas[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize,:] = toInsert
	return canvas

if __name__ == '__main__':
	#InstagramAPI = InstagramAPI(user,pwd)
	#InstagramAPI.login() # login
	filters = [normal,blur,grayscale, detection,edge,blockify]

	cap = cv2.VideoCapture(0)
	ret,img = cap.read()
	if not ret:
		img = load_image('assets/image.jpg')
	right = load_image('assets/right.png')
	left = load_image('assets/left.png')
	upload = load_image('assets/upload.png')
	success = load_image('assets/success.png')

	emojiMap = create_map('Emojis')
	colorMap = create_map('Colors')

	arrowOffsetXR = img.shape[0]//2+100
	arrowOffsetXL = img.shape[0]//2-100
	arrowOffsetY = 50

	uploadOffsetY = img.shape[0]-upload.shape[0]-50
	uploadOffsetX = img.shape[1]//2-upload.shape[1]//2

	successOffsetY = img.shape[0]//2-success.shape[0]//2
	successOffsetX = img.shape[1]//2-success.shape[1]//2

	cv2.namedWindow('image')
	cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.setMouseCallback('image',get_mouse)
	i = 0
	uploadSuccess = False
	while(True):
		if ret:
			ret,img = cap.read()
		dst = filters[i % len(filters)](img)
		toShow = overlay(copy.deepcopy(dst),right,arrowOffsetXR,arrowOffsetY)
		toShow = overlay(toShow,left,arrowOffsetXL-left.shape[0],arrowOffsetY)
		toShow = overlay(toShow,upload,uploadOffsetX,uploadOffsetY)
		if uploadSuccess:
			toShow = overlay(toShow,success,successOffsetX,successOffsetY)
		cv2.imshow('image',toShow)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
