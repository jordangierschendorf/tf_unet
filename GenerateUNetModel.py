import sys, getopt, os
import numpy as np
import shutil
from tf_unet import image_util, unet


"""
Original paper :
https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
    Parameters['layers'] = 5
    Parameters['convolutionFilter'] = 3
"""


#Change the following field to configure your UNet
def devTest():
    Parameters = dict()
    Parameters['layers'] = 3
    Parameters['convolutionFilter'] = 7
    Parameters['depthConvolutionFilter'] = 16
    Parameters['channel'] = 3
    Parameters['outputClassNumber'] = 2
    Parameters['optimizerType'] = "momentum"
    Parameters['optimizerValue'] = 0.90
    Parameters['learningRate'] = 0.01
    Parameters['batchSize'] = 2
    Parameters['decayRate'] = 0.95
    Parameters['dropout'] = 0.8
    Parameters['epoch'] = 1
    Parameters['iterations'] = 0
    Parameters['datasrc'] = "C:/Work/Projets/Git - UNets/Data"
    Parameters['output'] = "C:/Work/Projets/Git - UNets/Model"
    return Parameters

if __name__ == "__main__":

    Parameters = devTest()
   
# !!! Remove current path in Parameters[output] directory to create a new one !!!
if (os.path.isdir(Parameters['output'] + '/model')):
    shutil.rmtree(Parameters['output'])
os.mkdir(Parameters['output'])
os.mkdir(Parameters['output'] + '/model')
os.mkdir(Parameters['output'] + '/trainPrediction')

#Load images to train UNet
data_provider = image_util.ImageDataProvider(Parameters['datasrc'] + '/*', data_suffix=".png", mask_suffix="_mask.png")

# Compute the theoric total number of convolution filters :
# ConvFilter in descending path + ConvFilter in expanding path + output convolution (1x1) + up-convolution (2x2
totConvFilter = (Parameters['layers']*2) + (Parameters['layers']-1)*2 + 1 + (Parameters['layers'] - 1)
print("Total number of convolution filters (attempted) : " + str(totConvFilter))

#Compute the number of iterations necessary to see all train dataset in one batch
if int(Parameters['iterations']) == 0:
    Parameters['iterations'] = round(len(data_provider.data_files) / int(Parameters['batchSize']))

net = unet.Unet(layers=int(Parameters['layers']),
                features_root=int(Parameters['depthConvolutionFilter']),
                channels=int(Parameters['channel']),
                n_class=int(Parameters['outputClassNumber']),
                filter_size=int(Parameters['convolutionFilter']))

print("\tUNet created")
trainer = unet.Trainer(net,
                       optimizer=str(Parameters['optimizerType']),
                       batch_size=int(Parameters['batchSize']),
                       opt_kwargs=dict(momentum=np.float32(Parameters['optimizerValue']),
                                       learning_rate=np.float32(Parameters['learningRate']),
                                       decay_rate=np.float32(Parameters['decayRate'])))
print("\tUNet initialized")
path = trainer.train(data_provider=data_provider,
                     output_path=str(Parameters['output']),
                     dropout=np.float32(Parameters['dropout']),
                     training_iters=int(Parameters['iterations']),
                     epochs=int(Parameters['epoch']))

print("\n\tEND of learning step !")