from env import Environment_
from brain import Brain
from replay import ReplayBuffer
import numpy as np
from collections import deque
import cv2
import random
import tensorflow as tf
grid_shape = (16,16)
input_shape = (16,16,4)

replays = ReplayBuffer(100000)
brain = Brain(input_shape=input_shape, n_output=4)
brain.model = tf.keras.models.load_model('brain')
e = Environment_(grid_size=grid_shape)
epochs = 0


def generator():
    cnt = 1
    while 1:
        yield "{:3d}".format(cnt)
        cnt+=1

gen = generator()
while 1:
    epochs += 1
    obs = e.reset()
    done = False
    state_stack = deque(maxlen=4)
    cum_reward = 0
    state_stack.append(obs)  
    while not done:  
        
        if len(state_stack) < 4:
            action = random.randint(0,3)
        else:
            action, o = brain.get_action(np.expand_dims(np.dstack([*state_stack]),axis=0), use_epsilon = False)
            print(o)
        obs, reward, done, epi_step = e.step(action)
        if epi_step == 0:
            done = True
        if len(state_stack) < 4:
            state_stack.append(obs/255)
            continue
        else:
            s = np.dstack([*state_stack])
            state_stack.append(obs/255)
            s_prime = np.dstack([*state_stack])
            replays.push([s, s_prime, reward, action, done])
      
        cum_reward+= reward
        cv2.imshow("tmp", cv2.resize((obs).astype(np.uint8), (300,300), interpolation=cv2.INTER_AREA))
        
        key = cv2.waitKey(30)&0xff
        if key == ord('r'):
            done = True
    print('reward : ', cum_reward)

