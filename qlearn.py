from __future__ import print_function

import sys
from collections import deque
import numpy as np
from random import random, randrange, sample

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

sys.path.append('game/')
import wrapped_flappy_bird as game

from keras.initializations import normal, identity
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense

BATCH = 32 # size of minibatch

def baselineConv(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, init=lambda shape, name: normal(shape, scale=0.01, name=name), subsample=(4, 4),
                            input_shape=input_shape))
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


def preprocess(img, shape):
    img = rgb2gray(img)
    img = resize(img, shape)
    img = np.array(rescale_intensity(img, out_range=(0, 255))).astype(float)
    return img / 255.0

class FlappyAI(object):

    def __init__(self, model_constructor):

        self.game = 'bird' # the name of the game being played for log files
        self.config = 'nothreshold'
        self.actions = 2 # number of valid actions
        self.gamma = 0.99 # decay rate of past observations
        self.observations = 3200. # timesteps to observe before training
        self.explore = 3000000. # frames over which to anneal epsilon
        self.final_eps = 0.0001 # final value of epsilon
        self.init_eps = 0.1 # starting value of epsilon
        self.replay_mem = 50000 # number of previous transitions to remember
        self.frames_per_action = 1

        self.img_rows , self.img_cols = 80, 80
        self.img_shape = (self.img_rows, self.img_cols)
        self.img_channels = 4 #We stack 4 frames

        # store the previous observations in replay memory
        self.D = deque(maxlen=self.replay_mem)

        self.model = model_constructor(self.img_shape)

    def _remorph_targets(self, targets, Q, rewards, actions, ends):

        not_ends = ~ends
        Q_prime = np.multiply(not_ends, np.max(Q, axis=1))
        targets[:, actions] = rewards + self.gamma * Q_prime
        return targets

    def _train_batch(self, eps, loss):

        # Reduce our eps as we get more sure
        eps = max(self.final_eps, eps - (self.init_eps - self.final_eps) / self.explore)

        # Sample from our replay memory
        minibatch = np.array(sample(self.D, BATCH))

        img_batch = minibatch[:, 0]

        actions = minibatch[:, 1]
        rewards = minibatch[:, 2]
        nxt_state = minibatch[:, 3]
        ends = minibatch[:, 4]

        targets = self.model.predict(img_batch)
        Q_sa = self.model.predict(nxt_state)

        targets = self._remorph_targets(targets, Q_sa, rewards, actions, ends)

        loss += self.model.train_on_batch(img_batch, targets)

        Q_max = np.max(Q_sa)

        return eps, loss, Q_max

    @staticmethod
    def logger(timestep, mode, eps, action, reward, max_q, loss):

        msg = 'TIMESTEP: {0}\n\tMODE: {1}\n\tEPSILON: {2}\n\tACTION: {3}\n\tREWARD: {4}\n\t' \
              'Q_MAX: {5}\n\tLOSS: {6}'
        msg = msg.format(timestep, mode, eps, action, reward, max_q, loss)
        print(msg)

    def fit(self, run=False):

        # open up a game state to communicate with emulator
        game_state = game.GameState()

        # Get First State
        img_t, r_0, end = game_state.first_state()

        # Do some preprocessing
        img_t = preprocess(img_t, self.img_shape)

        # Build the first state
        state_t = np.stack((img_t, img_t, img_t, img_t), axis=0)

        # Reshape for Keras
        state_t = state_t.reshape((1,) + state_t.shape)

        if run:
            mode = 'run'
            observe_turns = float('inf') # We want to not train at all
            eps = self.final_eps
            self.model.load_weights('model.h5')
            self.model.compile(loss='mse', optimizer='rmsprop')
        else:
            mode = 'observe'
            observe_turns = self.observations
            eps = self.init_eps

        t = 0
        while True:
            q_max = 'n/a'
            loss = 0
            action_index = 0
            action_t = np.zeros((self.actions,))
            if not t % self.frames_per_action:
                # Pick a random action if below epsilon
                if random() <= eps:
                    action_index = randrange(self.actions)
                    action_t[action_index] = 1

                else:
                    action_index = np.argmax(self.model.predict(state_t))
                    action_t[action_index] = 1

            # Run our action
            img_t1, r_t, end = game_state.frame_step(action_t)

            img_t1 = preprocess(img_t1, self.img_shape)
            state_t1 = np.append(img_t1, state_t[:, :3, :, :], axis=1)

            # Store the state in our replay memory
            self.D.append((state_t, action_index, r_t, state_t1, end))

            # Train once we've observed long enough
            if t > observe_turns:
                mode = 'train'
                eps, loss, q_max = self._train_batch(eps, loss)


            state_t = state_t1
            t += 1

            # save progress every 100 iterations
            if t % 100 == 0:
                self.logger(t, mode, eps, action_index, r_t, q_max, loss)
                self.model.save_weights('weights_%s.h5' % t, overwrite=True)

if __name__ == '__main__':
    bot = FlappyAI(baselineConv)
    bot.fit()

























