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
user,pwd = 'ljinstacam', 'instacampassword'

faceCascade = cv2.CascadeClassifier(os.getcwd()+'/assets/haarcascade.xml')
i = 0

def get_mouse(event,x,y,flags,param):
	global mouseX,mouseY
	global i
	global uploadSuccess
	global captured
	global rSwipe
	global lSwipe
	global mouseClick
	global startX
	global startY
	global xRatio
	global yRatio
	global reset

	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y
		if not captured:
			"""
			if mouseY >= arrowOffsetY and mouseY <= arrowOffsetY+right.shape[0]:
				if mouseX >= arrowOffsetXR and mouseX <= arrowOffsetXR+right.shape[1]:
					#i+=1
					rSwipe = True
				elif mouseX >= arrowOffsetXL and mouseX <= arrowOffsetXL+left.shape[1]:
					lSwipe = True
					#i-=1
			"""
			if mouseY >= captureOffsetY and mouseY <= captureOffsetY + capture.shape[0] and mouseX >= captureOffsetX and mouseX <= captureOffsetX+capture.shape[1]:
					captured = True
			else:
				mouseClick = True
				startX = x
				startY = y
		elif captured:
			xRatio = 0
			yRatio = 0
			if mouseY >= xOffsetY and mouseY <= xOffsetY+exit.shape[0]:
				if mouseX >= xOffsetX and mouseX <= xOffsetX+exit.shape[1]:
					captured = False

		
			if not uploadSuccess and mouseY >= uploadOffsetY and mouseY <= uploadOffsetY+upload.shape[0]:
				if mouseX >= uploadOffsetX and mouseX <= uploadOffsetX+upload.shape[1]:
					timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mm%Ss')
					photo_path =os.getcwd()+'/'+timestamp+'.jpg'
					print(cv2.imwrite(photo_path,dst))
					caption = "Sample photo"
					print("START UPLOAD")
					InstagramAPI.uploadPhoto(photo_path, caption = caption)
					print("SUCCESS")
					uploadSuccess = True
					os.remove(photo_path)
			elif uploadSuccess:
				uploadSuccess = False
				captured = False

	if event == cv2.EVENT_LBUTTONUP:
		mouseClick = False
		reset = True
	if mouseClick:
		xRatio = (x-startX)/frame.shape[1]
		yRatio = (y-startY)/frame.shape[0]
		

		
def load_image(filename):
	return cv2.imread(os.getcwd()+'/'+filename,cv2.IMREAD_UNCHANGED)

class BlockifyThread(Thread):
	def __init__(self, img,blockMap):
		Thread.__init__(self)
		self.img = img
		self.canvas = np.zeros(img.shape,dtype=np.uint8)
		self.blockMap = blockMap
	def run(self):
		blockSize = 16+round(15*yRatio)
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
	filtered = to3D(cv2.Canny(grayscale(img)[0],100,200))
	if getUI:
		ui = overlay(copy.copy(filtered),edgeText,textOffsetX,img.shape[0]-edgeText.shape[0])
		return filtered, ui
	return filtered

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
	for (x, y, w, h) in faces:
		imgcopy = overlay(imgcopy, cv2.resize(smile,(w,h)), x, y)
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
	InstagramAPI = InstagramAPI(user,pwd)
	InstagramAPI.login()
	filters = [normal,blur, grayscale, detection,edge,colorfy,emojify]
	cap = cv2.VideoCapture(0)
	ret,frame = cap.read()
	if not ret:
		frame = load_image('assets/image.jpg')

	#Load UI elements
	right = cv2.resize(load_image('assets/ui/right.png'),(0,0),fx=0.7,fy=0.8)
	left = cv2.resize(load_image('assets/ui/left.png'),(0,0),fx=0.7,fy=0.8)
	upload = cv2.resize(load_image('assets/ui/upload.png'),(0,0),fx=0.7,fy=0.8)
	success = cv2.resize(load_image('assets/ui/success.png'),(0,0),fx=0.7,fy=0.8)
	capture = cv2.resize(load_image('assets/ui/capture.png'),(0,0),fx=0.5,fy=0.5)
	exit = load_image('assets/ui/x.png')
	
	textOffsetX = 10
	
	blurText = load_image('assets/ui/blur.png')
	grayscaleText = load_image('assets/ui/grayscale.png')
	edgeText = load_image('assets/ui/edge.png')
	detectionText = load_image('assets/ui/detection.png')
	colorfyText = load_image('assets/ui/colorfy.png')
	emojifyText = load_image('assets/ui/emojify.png')

	smile = load_image('assets/smile.png')
	
	emojiMap = create_map('Emojis')
	colorMap = create_map('Colors')


	arrowOffsetXR = frame.shape[1]-right.shape[1]-100#//2+200
	arrowOffsetXL = 100
	arrowOffsetY = 50

	xOffsetX = 15
	xOffsetY = 15

	mouseClick = False

	uploadOffsetY = frame.shape[0]-upload.shape[0]-50
	uploadOffsetX = frame.shape[1]//2-upload.shape[1]//2

	captureOffsetY = frame.shape[0]-capture.shape[0]-25
	captureOffsetX = frame.shape[1]//2-capture.shape[1]//2

	successOffsetY = frame.shape[0]//2-success.shape[0]//2
	successOffsetX = frame.shape[1]//2-success.shape[1]//2

	cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.setMouseCallback('image',get_mouse)
	i = 0
	uploadSuccess = False
	captured = False
	rSwipe = False
	lSwipe = False
	decrement = 0.08
	counter = 1-decrement
	startX = 0
	startY = 0
	xRatio = 0
	yRatio = 0
	reset = False
	while(True):
		tick = datetime.datetime.now()	
		if ret:
			ret,frame = cap.read()
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
				#toShow = overlay(ui,right,arrowOffsetXR,arrowOffsetY)
				#toShow = overlay(toShow,left,arrowOffsetXL,arrowOffsetY)
				toShow = overlay(ui,capture,captureOffsetX,captureOffsetY)
		else:
			toShow = overlay(copy.copy(dst),exit,xOffsetX,xOffsetY)
			toShow = overlay(toShow,upload,uploadOffsetX,uploadOffsetY)
			if uploadSuccess:
				toShow = overlay(toShow,success,successOffsetX,successOffsetY)
		cv2.imshow('image',toShow)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		tock = datetime.datetime.now()
		diff = tock-tick
		#print(diff.total_seconds())
	cv2.destroyAllWindows()
