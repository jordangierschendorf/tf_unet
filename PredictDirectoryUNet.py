from tf_unet import util, image_util, unet
import sys
import glob
import os

#Script which predict a flow of images. Can be used with GPU Tensorflow without overload graphic memory


#CHANGE YO YOUR PATH
listOfFile = glob.glob("C:/Work/Projets/Git - UNets/DataTest/*.png")
modelPath = "C:/Work/Projets/Git - UNets/Model/model/model.ckpt"

#GIVE YOUR CONFIGURATION USE TO GENERATE CURRENT model.ckpt
Layers = 3
Features = 16
Channels = 3
nClass = 2
filterSize = 7

print('\n')
print('Path to model : ' + modelPath)
print('Layers = ' + str(Layers))
print('Features = ' + str(Features))
print('Channels = ' + str(Channels))
print('Class number = ' + str(nClass))
print('Convolution size filter = ' + str(filterSize))

net = unet.Unet(layers=int(Layers),
				features_root=int(Features),
				channels=int(Channels),
				n_class=int(nClass),
				filter_size=int(filterSize))

for i in range(0, len(listOfFile)):
	imagePath = listOfFile[i]
	data_provider = image_util.SingleImageDataProvider(imagePath)
	predicter = net.predict(modelPath, data_provider.img)
	util.PlotSingleImagePrediction(predicter, output_path=imagePath)

print("\nPrediction finished !")