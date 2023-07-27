from collections import deque
import random
import tensorflow as tf


class ReplayBuffer():
    def __init__(self, maxlen) -> None:
        self.buffer = deque(maxlen=maxlen)
    
    def sample(self,batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, s_prime, r, a, done = zip(*samples)
        return tf.Variable(s), tf.Variable(s_prime), tf.Variable(r, dtype=tf.float32), tf.Variable(a), tf.Variable(done, dtype=tf.float32)
    
    def push(self, sample):
        self.buffer.append(sample)
    
    @property
    def size(self,):
        return len(self.buffer)
    @property
    def capacity(self):
        return self.buffer.maxlen