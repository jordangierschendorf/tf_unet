from tf_unet import util, image_util, unet
import sys
import glob
import os

if __name__ == "__main__":

	#CHANGE YO YOUR PATH
	listOfFile = glob.glob("Path/To/ImageToPredict/*.png")
	modelPath = "Path/To/YourModel/model.ckpt"

	#GIVE YOUR CONFIGURATION USE TO GENERATE CURRENT model.ckpt
	Layers = 3
	Features = 16
	Channels = 3
	regCoef = 0.01
	nClass = 2
	filterSize = 7

	print('\n')
	print('Path to model : ' + modelPath)
	print('Layers = ' + Layers)
	print('Features = ' + Features)
	print('Channels = ' + Channels)
	print('Regulation Coef = ' + regCoef)
	print('Class number = ' + nClass)
	print('Convolution size filter = ' + filterSize)

	net = unet.Unet(layers=int(Layers), features_root=int(Features), channels=int(Channels), regularisationConstant=float(regCoef), n_class=int(nClass), filter_size=int(filterSize))

	for i in range(0, len(listOfFile)):
		imagePath = listOfFile[i]
		data_provider = image_util.SingleImageDataProvider(imagePath)
		predicter = net.predict(modelPath, data_provider.img)
		util.PlotSingleImagePrediction(data_provider, predicter, output_path=imagePath)

	print("\nPrediction finished !")