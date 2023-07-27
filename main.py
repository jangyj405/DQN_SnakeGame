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
brain = Brain(input_shape=input_shape, n_output=4, load_model=True, epsilon = 0.9)
e = Environment_(grid_size=grid_shape)
epochs = 0
while 1:
    t_buffer = []
    obs = e.reset()
    done = False
    state_stack = deque(maxlen=4)
    cum_reward = 0
    state_stack.append(obs/255) 
    while not done:  
        epochs += 1
        if len(state_stack) < 4:
            action = random.randint(0,3)
        else:
            action, o = brain.get_action(np.expand_dims(np.dstack([*state_stack]),axis=0), use_epsilon = True)
        obs, reward, done, epi_step = e.step(action)
        if  epi_step == 0:
            done = True
        if len(state_stack) < 4:
            state_stack.append(obs/255)
            continue
        else:
            s = np.dstack([*state_stack])
            state_stack.append(obs/255)
            s_prime = np.dstack([*state_stack])
            t_buffer.append([s, s_prime, reward, action, done])
      
        if replays.size < 10000:
            continue
        else:
            loss = brain.train_step(replays.sample(256))
            print(tf.reduce_mean(loss))
        
        cum_reward+= reward
        cv2.imshow("tmp", cv2.resize((s_prime*255).astype(np.uint8), (300,300), interpolation=cv2.INTER_AREA))
        key = cv2.waitKey(30) & 0xff
        if key == ord('s'):
            brain.model.save('brain')    
        if epochs % 10 == 0:
            brain.epsilon = max(0.01, brain.epsilon - 0.001)
            brain.target_model.set_weights(brain.model.get_weights())
        if epochs % 100 == 0:
            print(brain.epsilon)
            brain.model.save('brain')    
        if epochs % 1000 == 0:
            epochs = 0
    if replays.size > replays.capacity * 0.9:
        if cum_reward > -20:
            for step in t_buffer:
                replays.push(step)
    else:
        for step in t_buffer:
                replays.push(step)
    if replays.size > 10000:
        print('reward : ', cum_reward)
