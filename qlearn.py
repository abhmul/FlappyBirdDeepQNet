

from keras.initializations import normal, identity
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def baselineConv():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, init=lambda shape, name: normal(shape, scale=0.01, name=name), subsample=(4, 4),
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, init=lambda shape, name: normal(shape, scale=0.01, name=name), subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(2, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    model.compile(loss='mse', optimizer='rmsprop')
    return model

