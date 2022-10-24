import numpy as np
import cv2 
import time
import matplotlib.pyplot as plt

path_model = 'Depth Estimation'

model_name = "model-f6b98070.onnx"

model = cv2.dnn.readNet(r'models\model-f6b98070.onnx')

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Read in the image
img = cv2.imread(r'Media\colorImg2.jpg')
img = cv2.resize(img,(640,480))
imgHeight, imgWidth, channels = img.shape

# start time to calculate FPS
start = time.time()


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Create Blob from Input Image
# MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

# Set input to the model
model.setInput(blob)

# Make forward pass in model
output = model.forward()

output = output[0,:,:]
output = cv2.resize(output, (imgWidth, imgHeight))

# Normalize the output
output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow('image', img)
cv2.imshow('Depth Map', output)

output = output*255
cv2.imwrite('Media\depthImg2.png', output)
cv2.imwrite('Media\colorImg2.jpg', img)

cv2.waitKey(0)
