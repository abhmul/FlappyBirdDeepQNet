from __future__ import print_function

import sys
from collections import deque
import numpy as np
from random import random, randrange, sample

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

import wrapped_flappy_bird as game

from keras.initializations import normal, identity
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, Dropout, MaxPooling2D

BATCH = 32  # size of minibatch


def baselineConv(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, init=lambda shape, name: normal(shape, scale=0.01, name=name), subsample=(4, 4),
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 4, 4, init=lambda shape, name: normal(shape, scale=0.01, name=name), subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(256, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(Dense(2, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    model.compile(loss='mse', optimizer='rmsprop')
    return model


def preprocess(img, shape):
    img = rgb2gray(img)
    img = resize(img, shape)
    img = rescale_intensity(img, out_range=(0, 255)).astype(float)
    return img / 255.0


class FlappyAI(object):
    def __init__(self, model_constructor, mode='observe'):

        self.game = 'bird'  # the name of the game being played for log files
        self.actions = 2  # number of valid actions
        self.gamma = 0.99  # decay rate of past observations
        self.observations = 10000  # timesteps to observe before training
        self.explore = 3000000.  # frames over which to anneal epsilon
        self.final_eps = 0.0001  # final value of epsilon
        self.init_eps = 0.1  # starting value of epsilon
        self.eps = self.init_eps
        self.replay_mem = 50000  # number of previous transitions to remember
        self.frames_per_action = 1
        self.mode = mode

        self.img_rows, self.img_cols = 80, 80
        self.img_shape = (self.img_rows, self.img_cols)
        self.img_channels = 4  # We stack 4 frames

        # store the previous observations in replay memory
        self.D = deque(maxlen=self.replay_mem)

        self.model = model_constructor((4,) + self.img_shape)

    def _q_prime(self, states, action_inds, rewards, nxt_states, terminals):

        # We want this to be False when terminating so we don't add on the Q value
        not_terminals = ~terminals

        q_t = self.model.predict(states)
        q_prime = self.model.predict(nxt_states)

        q_t[np.arange(states.shape[0]), action_inds] = rewards + self.gamma * np.max(q_prime, axis=1) * not_terminals

        return q_t

    def _batch(self, minibatch):

        states, action_inds, rewards, nxt_states, terminals = zip(*minibatch)

        states = np.stack(states, axis=0).reshape(len(states), -1, self.img_rows, self.img_cols)
        action_inds = np.array(action_inds)
        rewards = np.array(rewards)
        nxt_states = np.stack(nxt_states, axis=0).reshape(len(nxt_states), -1, self.img_rows, self.img_cols)
        terminals = np.array(terminals)

        # Calculate the Q_prime values
        q_t = self._q_prime(states, action_inds, rewards, nxt_states, terminals)

        return states, q_t

    def _batch_gen(self):

        while True:
            shuffled = np.array(sample(self.D, len(self.D)))
            replay_inds = np.arange(len(shuffled))
            np.random.shuffle(replay_inds)
            for i in xrange(0, len(replay_inds), BATCH):
                minibatch_inds = replay_inds[i:min(i+32, len(replay_inds))]
                minibatch = shuffled[minibatch_inds]
                yield self._batch(minibatch)


    @staticmethod
    def logger(timestep, mode, eps, action, reward, max_q, loss):

        msg = 'TIMESTEP: {0}\n\tMODE: {1}\n\tEPSILON: {2}\n\tACTION: {3}\n\tREWARD: {4}\n\t' \
              'Q_MAX: {5}\n\tLOSS: {6}'
        msg = msg.format(timestep, mode, eps, action, reward, max_q, loss)
        print(msg)

    def play(self, game_state, state_t):

        t = 0
        while True:
            action_index = 0
            q_max = 0
            action_t = np.zeros((self.actions,))
            if not t % self.frames_per_action:
                # Pick a random action if below epsilon
                if random() <= self.eps:
                    action_index = randrange(self.actions)
                    q_max = 'RANDOM ACTION'
                else:
                    q = self.model.predict(state_t)
                    action_index = np.argmax(q)
                    q_max = np.max(q)

            # Run our action
            action_t[action_index] = 1
            img_t1, r_t, end = game_state.frame_step(action_t)

            img_t1 = preprocess(img_t1, self.img_shape).reshape((1, 1,) + self.img_shape)
            state_t1 = np.append(img_t1, state_t[:, :3, :, :], axis=1)  # (new image, old, older, oldest)

            # Store the state in our replay memory
            self.D.append((state_t, action_index, r_t, state_t1, end))

            if self.mode == 'train':
                # Reduce our eps as we get more sure
                self.eps = max(self.final_eps, self.eps - (self.init_eps - self.final_eps) / self.explore)
                minibatch = sample(self.D, BATCH)
                states, q_t = self._batch(minibatch)
                loss = self.model.train_on_batch(states, q_t)
                self.logger(t, self.mode, self.eps, action_index, r_t, q_max, loss)
                if t % 100 == 0:
                    # save progress every 100 iterations
                    self.model.save_weights('weights_%s.h5' % t, overwrite=True)

            if t == self.observations + 1:
                print('-----------FINISHED OBSERVING----------')
                self.mode = 'train'
                # Train over all of our observations
                self.model.fit_generator(self._batch_gen(), samples_per_epoch=len(self.D), nb_epoch=1, verbose=1)
                print('-----------FINISHED TRAINING ON OBSERVATIONS-------------')

            if self.mode == 'observe' and not t % 100:
                print('--------OBSERVATION # %s---------------' % t)

            state_t = state_t1
            t += 1

    def fit(self):

        # open up a game state to communicate with emulator
        game_state = game.GameState()
        # Get First State
        img_t, r_0, end = game_state.first_state()
        # Do some preprocessing
        img_t = preprocess(img_t, self.img_shape)
        # Build the first state
        state_t = np.stack((img_t, img_t, img_t, img_t), axis=0)  # (4, 80, 80)
        # Reshape for Keras
        state_t = state_t.reshape((1,) + state_t.shape)  # (1, 4, 80, 80)

        # Set it up for running if we need to
        if self.mode == 'run':
            self.eps = self.final_eps
            self.model.load_weights('model.h5')
            self.model.compile(loss='mse', optimizer='rmsprop')

        self.play(game_state, state_t)


if __name__ == '__main__':
    bot = FlappyAI(baselineConv)
    bot.fit()
