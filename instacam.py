from InstagramAPI import InstagramAPI
import numpy as np
import numpy.random as npr
import cv2
import time
import datetime
import os
import copy
from threading import Thread
import math
import random
from scipy.signal import convolve2d as conv
user,pwd = 'ljinstacam', 'instacampassword'

faceCascade = cv2.CascadeClassifier(os.getcwd()+'/assets/haarcascade.xml')
i = 0

def get_mouse(event,x,y,flags,param):
	global mouseX,mouseY
	global i
	global captured
	global mouseClick
	global startX
	global startY
	global xRatio
	global yRatio
	global reset
	global uploading

	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y
		if not captured:
			if mouseY >= captureOffsetY and mouseY <= captureOffsetY + capture.shape[0] and mouseX >= captureOffsetX and mouseX <= captureOffsetX+capture.shape[1]:
					captured = True
			else:
				mouseClick = True
				startX = x
				startY = y
		elif captured:
			xRatio = 0
			yRatio = 0
			if not uploading and mouseY >= xOffsetY and mouseY <= xOffsetY+exit.shape[0]:
				if mouseX >= xOffsetX and mouseX <= xOffsetX+exit.shape[1]:
					captured = False

		
			if not uploading and mouseY >= uploadOffsetY and mouseY <= uploadOffsetY+upload.shape[0]:
				if mouseX >= uploadOffsetX and mouseX <= uploadOffsetX+upload.shape[1]:
					uploading = True
					uThread = UploadThread(dst)
					uThread.daemon = True
					uThread.start()

	if event == cv2.EVENT_LBUTTONUP:
		mouseClick = False
		reset = True
	if mouseClick:
		xRatio = (x-startX)/frame.shape[1]
		yRatio = (y-startY)/frame.shape[0]
		

		
def load_image(filename):
	return cv2.imread(os.getcwd()+'/'+filename,cv2.IMREAD_UNCHANGED)

class UploadThread(Thread):
	def __init__(self,img):
		Thread.__init__(self)
		self.img = img
	def run(self):
		global uploading
		global captured
		
		timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mm%Ss')
		photo_path =os.getcwd()+'/'+timestamp+'.jpg'
		print(cv2.imwrite(photo_path,self.img))
		caption = "Sample photo"
		print("START UPLOAD")
		InstagramAPI.uploadPhoto(photo_path, caption = caption)
		print("SUCCESS")
		uploadSuccess = True
		os.remove(photo_path)
		uploading = False
		captured = False

class BlockifyThread(Thread):
	def __init__(self, img,blockMap):
		Thread.__init__(self)
		self.img = img
		self.canvas = np.zeros(img.shape,dtype=np.uint8)
		self.blockMap = blockMap
	def run(self):
		blockSize = 30+round(20*yRatio)
		numBlocksWidth = self.img.shape[1]//blockSize
		numBlocksHeight = self.img.shape[0]//blockSize
		blockKeys = self.blockMap.keys()
		mapMat = np.array(list(blockKeys))
		for i in range(numBlocksHeight):
			for j in range(numBlocksWidth):
				block = self.img[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize,:]
				rAvg = np.mean(block[:,:,0])
				bAvg = np.mean(block[:,:,1])
				gAvg = np.mean(block[:,:,2])
				means = np.array([[rAvg,bAvg,gAvg]])
				meanMat = np.repeat(means,len(blockKeys),0)
				diffMat = np.sum(np.square(np.subtract(meanMat,mapMat)),1)
				minIndex = np.argmin(diffMat)
				key = mapMat[minIndex,:]
				newBlock = self.blockMap[tuple(key)]
				toInsert = cv2.resize(newBlock,(block.shape[0],block.shape[1]))
				cols = toInsert.shape[0]
				rows = toInsert.shape[1]
				angle = random.randint(0,359)
				R = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
				self.canvas[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize,:] = cv2.warpAffine(toInsert,R,(cols,rows))

	def join(self):
		Thread.join(self)
		return self.canvas

def colorfy(img, getUI = True):
	filtered = blockify(img,colorMap)
	if getUI:
		ui = overlay(copy.copy(filtered),colorfyText,textOffsetX,img.shape[0]-colorfyText.shape[0])
		return filtered, ui
	return filtered

def emojify(img, getUI = True):
	filtered = blockify(img,emojiMap)
	if getUI:
		ui = overlay(copy.copy(filtered),emojifyText,textOffsetX,img.shape[0]-emojifyText.shape[0])
		return filtered, ui
	return filtered

def pixellate(img, getUI = True):
	numBlocks = 9+round(8*yRatio)
	blockWidth = img.shape[1]//numBlocks
	blockHeight = img.shape[0]//numBlocks
	for m in range(numBlocks):
		for n in range(numBlocks):
			img[m*blockHeight:(m+1)*blockHeight,n*blockWidth:(n+1)*blockWidth,0] = np.mean(img[m*blockHeight:(m+1)*blockHeight,n*blockWidth:(n+1)*blockWidth,0]) * np.ones((blockHeight,blockWidth),dtype=np.uint8)
			img[m*blockHeight:(m+1)*blockHeight,n*blockWidth:(n+1)*blockWidth,1] = np.mean(img[m*blockHeight:(m+1)*blockHeight,n*blockWidth:(n+1)*blockWidth,1]) * np.ones((blockHeight,blockWidth),dtype=np.uint8)
			img[m*blockHeight:(m+1)*blockHeight,n*blockWidth:(n+1)*blockWidth,2] = np.mean(img[m*blockHeight:(m+1)*blockHeight,n*blockWidth:(n+1)*blockWidth,2]) * np.ones((blockHeight,blockWidth),dtype=np.uint8)
	if getUI:
		return img,img
	return img

def blockify(img,blockMap):
	Threads = []
	dst = np.zeros((0,img.shape[1],3),dtype=np.uint8)
	numThreads = 2
	for n in range(numThreads):
		toPass = img[n*(img.shape[0]//numThreads):(n+1)*(img.shape[0]//numThreads),:,:]
		T = BlockifyThread(toPass,blockMap)
		Threads.append(T)
	for T in Threads:
		T.start()
	for T in Threads:
		new = T.join()
		dst = np.concatenate((dst,new),axis=0)
	return dst	

def blur(img, getUI = True):
	kSize = 9 + round(7*yRatio)
	kernel = np.ones((kSize,kSize),np.float32)/(kSize*kSize)
	filtered = cv2.filter2D(img,-1,kernel)
	if getUI:
		ui = overlay(copy.copy(filtered),blurText,textOffsetX,img.shape[0]-blurText.shape[0])
		return filtered,ui
	return filtered

def edge(img, getUI = True):
	filtered = to3D(cv2.Canny(grayscale(img)[0],100+(yRatio*100),200))
	if getUI:
		ui = overlay(copy.copy(filtered),edgeText,textOffsetX,img.shape[0]-edgeText.shape[0])
		return filtered, ui
	return filtered

def shift(img,getUI = True):
	flip = True
	offset = max(1,1+round(200*yRatio))
	for n in range(img.shape[0]):
		if flip:
			img[n,offset:,:] = img[n,0:-offset,:]
			flip = not flip
		else:	
			img[n,0:-offset,:] = img[n,offset:,:]
			flip = not flip
	if getUI:
		return img,img
	return img

def normal(img, getUI = True):
	if getUI:
		return img, copy.copy(img)
	return img

def grayscale(img, getUI = True):
	filtered = to3D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	if getUI:
		ui = overlay(copy.copy(filtered),grayscaleText,textOffsetX,img.shape[0]-grayscaleText.shape[0])
		return filtered,ui
	return filtered

def fade(img, getUI = True):
	kSize = 9
	kernel = np.ones((kSize,kSize),np.float32)/(kSize*kSize)
	blurredImg = cv2.filter2D(img,-1,kernel)
	blurredLast = cv2.filter2D(lastFrame,-1,kernel)
	canvas = np.zeros(img.shape,dtype=np.uint8)
	diffImg = np.sum(blurredImg-blurredLast[:,:img.shape[1],:],axis=2)
	thresh = (200+(yRatio*55))*3
	canvas[np.where(diffImg > thresh)] = img[np.where(diffImg > thresh)]
	if getUI:
		return canvas,canvas
	return canvas

def seizure(img, getUI = True):
	filtered = img-lastFrame[:,:img.shape[1],:]
	if getUI:
		return filtered,filtered
	return filtered

def to3D(img):
	return np.repeat(img[:,:,np.newaxis],3,axis=2)

def overlay(img, over, startX, startY):
	try:
		overAlpha = over[:,:,3]/255
		imageAlpha = 1 - overAlpha
		overlayMult = np.multiply(over[:,:,0:3],to3D(overAlpha))
		imageSplice = img[startY:startY+over.shape[0],startX:startX+over.shape[1],0:3]
		imageMult = np.multiply(imageSplice,to3D(imageAlpha))
		toOverlay = np.add(overlayMult,imageMult)
		img[startY:startY+over.shape[0],startX:startX+over.shape[1],0:3] = toOverlay
		return img
	except:
		return img

def detection(img, getUI = True):
	imgcopy = copy.deepcopy(img)
	gray, ui = grayscale(imgcopy)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	overlayIndex = round(yRatio*10)
	emoji = emojiOverlays[overlayIndex % len(emojiOverlays)]
	for (x, y, w, h) in faces:
		imgcopy = overlay(imgcopy, cv2.resize(emoji,(w,h)), x, y)
	if getUI:
		ui = overlay(copy.copy(imgcopy),detectionText,textOffsetX,img.shape[0]-detectionText.shape[0])
		return imgcopy, ui
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

if __name__ == '__main__':
	#InstagramAPI = InstagramAPI(user,pwd)
	#InstagramAPI.login()
	filters = [normal,blur, grayscale, shift,detection,edge,fade,seizure,pixellate,colorfy,emojify]
	cap = cv2.VideoCapture(0)
	ret,frame = cap.read()
	if not ret:
		frame = load_image('assets/image.jpg')

	#Load UI elements
	right = cv2.resize(load_image('assets/ui/right.png'),(0,0),fx=0.7,fy=0.8)
	left = cv2.resize(load_image('assets/ui/left.png'),(0,0),fx=0.7,fy=0.8)
	#upload = cv2.resize(load_image('assets/ui/upload.png'),(0,0),fx=0.7,fy=0.8)
	upload = cv2.resize(load_image('assets/ui/ig.png'),(0,0),fx=0.3,fy=0.3)
	success = cv2.resize(load_image('assets/ui/success.png'),(0,0),fx=0.7,fy=0.8)
	capture = cv2.resize(load_image('assets/ui/capture.png'),(0,0),fx=0.5,fy=0.5)
	load = load_image('assets/ui/load.png')
	exit = load_image('assets/ui/x.png')
	
	lastFrame = np.zeros(frame.shape,dtype=np.uint8)
	textOffsetX = 10
	
	blurText = load_image('assets/ui/blur.png')
	grayscaleText = load_image('assets/ui/grayscale.png')
	edgeText = load_image('assets/ui/edge.png')
	detectionText = load_image('assets/ui/detection.png')
	colorfyText = load_image('assets/ui/colorfy.png')
	emojifyText = load_image('assets/ui/emojify.png')

	smile = load_image('assets/faces/smile.png')
	wink = load_image('assets/faces/wink.png')
	heart = load_image('assets/faces/heart.png')
	fish = load_image('assets/faces/fish.png')
	
	emojiOverlays = [smile,wink,heart,fish]

	emojiMap = create_map('Emojis')
	colorMap = create_map('Colors')


	arrowOffsetXR = frame.shape[1]-right.shape[1]-100
	arrowOffsetXL = 100
	arrowOffsetY = 50

	xOffsetX = 15
	xOffsetY = 15

	mouseClick = False

	uploadOffsetY = frame.shape[0]-upload.shape[0]-50
	uploadOffsetX = frame.shape[1]//2-upload.shape[1]//2

	loadOffsetX = frame.shape[0]//2-load.shape[0]//2
	loadOffsetY = frame.shape[1]//2-load.shape[1]//2

	captureOffsetY = frame.shape[0]-capture.shape[0]-25
	captureOffsetX = frame.shape[1]//2-capture.shape[1]//2

	successOffsetY = frame.shape[0]//2-success.shape[0]//2
	successOffsetX = frame.shape[1]//2-success.shape[1]//2

	cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.setMouseCallback('image',get_mouse)
	i = 0
	captured = False
	uploading = False
	startX = 0
	startY = 0
	xRatio = 0
	yRatio = 0
	loadRotation = 0
	reset = False
	while(True):
		tick = datetime.datetime.now()	
		if ret:
			ret,frame = cap.read()
			frame = np.flip(frame,1)
		else:
			frame = load_image('assets/image.jpg')
		if not captured:
	
			if mouseClick:
				if xRatio > 0.05:
					xRatio = min(xRatio,0.95)
					split = math.floor(frame.shape[1]*xRatio)
					imA = frame[:,:split,:]
					imB = frame[:,split:,:]
					dstA = filters[(i-1) % len(filters)](copy.deepcopy(imA), False)
					dstB = filters[i % len(filters)](copy.deepcopy(imB), False)
					dst = np.concatenate((dstA,dstB),axis=1)
				elif xRatio < -0.05:
					xRatio = max(xRatio,-0.95)
					split = math.floor(frame.shape[1]*(1+xRatio))
					imA = frame[:,:split,:]
					imB = frame[:,split:,:]
					dstA = filters[i % len(filters)](copy.deepcopy(imA), False)
					dstB = filters[(i+1) % len(filters)](copy.deepcopy(imB), False)
					dst = np.concatenate((dstA,dstB),axis=1)
				else:
					dst = filters[i % len(filters)](copy.deepcopy(frame), False)
				toShow = dst
			elif reset:
				if xRatio > 0.5:
					xRatio = 0
					yRatio = 0
					i -= 1
				elif xRatio < -0.5:
					i += 1
					xRatio = 0
					yRatio = 0
				reset = False
		
			else:
				dst,ui = filters[i % len(filters)](copy.deepcopy(frame))
				toShow = overlay(ui,capture,captureOffsetX,captureOffsetY)
		else:
			toShow = overlay(copy.copy(dst),exit,xOffsetX,xOffsetY)
			toShow = overlay(toShow,upload,uploadOffsetX,uploadOffsetY)
			if uploading:
				loadCols = load.shape[0]
				loadRows = load.shape[1]
				M = cv2.getRotationMatrix2D((loadCols/2,loadRows/2),loadRotation,1)
				toShow = overlay(toShow,cv2.warpAffine(load,M,(loadCols,loadRows)),loadOffsetX,loadOffsetY)
				loadRotation -= 4
		cv2.imshow('image',toShow)
		lastFrame = frame
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		tock = datetime.datetime.now()
		diff = tock-tick
		#print(diff.total_seconds())
	cv2.destroyAllWindows()
