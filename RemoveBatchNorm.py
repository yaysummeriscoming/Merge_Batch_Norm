import keras.backend as K
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
import numpy as np
from kerassurgeon.operations import delete_layer

# This function deletes all instances of a specific layer type in a model
def DeleteLayerType(model, layerType):
    for curLayer in model.layers:
        if type(curLayer) is layerType:
            model = delete_layer(model=model, layer=curLayer, copy=False)

    return model


def RemoveBatchNorm(model, deleteBNLayers=True, verbose=False):

    if verbose:
        print('Merging batch norm into layers')
    
    lastLayer = model.layers[0]

    # Loop through each layer and merge the BN parameters into the preceding conv layer
    for curLayer in model.layers:
        if type(curLayer) is BatchNormalization and type(lastLayer) is Convolution2D:

            # The last layer must have a bias term to be able to merge the BN transform
            assert len(lastLayer.weights) > 1

            # Get the weights, bias terms and BN parameters
            convWeights = K.get_value(lastLayer.weights[0])
            convBias = K.get_value(lastLayer.weights[1])

            BN_gamma    = K.get_value(curLayer.gamma)
            BN_beta     = K.get_value(curLayer.beta)
            BN_variance = K.get_value(curLayer.moving_variance)
            BN_mean     = K.get_value(curLayer.moving_mean)

            if verbose:
                print('BN gamma average is: ')
                print(BN_gamma)

                print('BN beta average is: ')
                print(BN_beta)

                print('BN variane average is: ')
                print(BN_variance)

                print('BN mean average is: ')
                print(BN_mean)

            # Compute the new conv layer weights & bias
            scalingConstant = BN_gamma / np.sqrt(BN_variance + curLayer.epsilon)
            scalingConstantBroadcasted = np.broadcast_to(scalingConstant, convWeights.shape)

            newConvWeights = scalingConstantBroadcasted * convWeights
            newConvBias = scalingConstant * (convBias - BN_mean) + BN_beta

            # Update the conv layer weights & bias
            K.set_value(lastLayer.weights[0], newConvWeights)
            K.set_value(lastLayer.weights[1], newConvBias)

            # Set the batch norm transformation to the identity
            BN_gamma    = np.ones(shape=BN_gamma.shape, dtype=BN_gamma.dtype)
            BN_beta     = np.zeros(shape=BN_beta.shape, dtype=BN_beta.dtype)
            BN_variance = np.ones(shape=BN_variance.shape, dtype=BN_variance.dtype)
            BN_mean     = np.zeros(shape=BN_mean.shape, dtype=BN_mean.dtype)

            K.set_value(curLayer.gamma, BN_gamma)
            K.set_value(curLayer.beta, BN_beta)
            K.set_value(curLayer.moving_variance, BN_variance)
            K.set_value(curLayer.moving_mean, BN_mean)

        lastLayer = curLayer

    # Delete the batch normalisation layers if specified
    if deleteBNLayers:
        model = DeleteLayerType(model=model, layerType=BatchNormalization)

    return model
