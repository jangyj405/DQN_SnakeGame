import numpy as np
import random
import tensorflow as tf
MAX_STEP = 200
class Environment_():
    def __init__(self, grid_size, ):
        self.grid_size = grid_size
        h,w = self.grid_size
        self.grid = np.zeros(grid_size, dtype=np.uint8)
        self.snake = [(h//2, (w//2)),(h//2, (w//2-1))]
        self.actionsets = [(0,-1),(0,1),(-1,0),(1,0)]
        self.apple = self.get_apple()
        self.epi_step = MAX_STEP
        pass

    def get_apple(self,):
        h,w = self.grid_size
        y,x = 0,0
        while 1:
            y,x = random.randint(0, h-1),random.randint(0, w-1)
            if (y,x) in self.snake:
                continue
            else:
                break
        return (y,x)
    
    def step(self, action):
        h,w = self.grid_size
        (dy,dx) = self.actionsets[action]
        (hy,hx) = self.snake[0]
        reward = 0
        done = 0
        if (hy+dy, hx+dx) == self.snake[1]:
            dx, dy = -dx, -dy
        if (hy+dy, hx+dx) == self.apple:
            self.snake.insert(0,self.apple)
            self.apple = self.get_apple()
            reward = 10
            self.epi_step = MAX_STEP
        else:
            if (hy+dy, hx+dx) in self.snake:
                done = 1
                reward = -20
            elif hy+dy < 0 or hy+dy >= h or hx+dx < 0 or hx+dx >= w:
                done = 1
                reward = -20
            else:
                ay,ax = self.apple
                if (pow(hy+dy-ay,2)+ pow(hx+dx-ax,2)) < (pow(hy-ay,2)+ pow(hx-ax,2)):
                    reward = 1
                else:
                    reward = -1
                if (pow(hy+dy-ay,2)+ pow(hx+dx-ax,2)) == 1:
                    reward = 3
                pass
            self.epi_step -= 1
            self.snake.pop()
            self.snake.insert(0,(hy+dy, hx+dx))

        self.grid.fill(0)
        for body in self.snake:
            by,bx = body
            if 0 <= bx < w and 0<=by<h:
                self.grid[by,bx] = int(100 * (self.epi_step/MAX_STEP))
        ay,ax = self.apple
        self.grid[ay,ax] = 255

        return self.get_state(), reward, done, self.epi_step
    def dist_apple(self, head):
        hy,hx = head
        ay,ax = self.apple
        return (hy-ay)**2 + (hx-ax)**2
    def get_state(self):
        return self.grid.copy()
    
    def reset(self):
        self.grid.fill(0)
        self.epi_step = MAX_STEP
        h,w = self.grid_size
        self.snake = [(h//2, (w//2)),(h//2, (w//2-1))]
        self.apple = self.get_apple()

        for body in self.snake:
            by,bx = body
            self.grid[by,bx] = int(100 * (self.epi_step/MAX_STEP))
        ay,ax = self.apple
        self.grid[ay,ax] = 255

        return self.get_state()


    def get_state_(self):
        h,w = self.grid_size
        dirs = [(dy,dx) for dy in range(-1,2) for dx in range(-1,2) if dx or dy]
        results = [0 for _ in range(len(dirs) * 3)]
        for dir, i in zip(dirs, range(len(dirs))):
            hy,hx = self.snake[0]
            dy,dx = dir
            dist = 0
            
            hy, hx, dist = hy+dy, hx+dx, dist+1
            if hy < 0 or hy >= h or hx < 0 or hx >= w:
                results[3*i + 0] = -1
                break
            elif (hy, hx) in self.snake:
                results[3*i + 1] = -1
                break
            elif (hy, hx) == self.apple:
                results[3*i + 2] = 1
                break
        results.append(self.dist_apple(self.snake[0]))
        return results
if __name__=='__main__':
    import cv2

    env = Environment_((32,32))
    while 1:
        env.reset()
        done = False
        while not done:
            s,r,done,_ = env.step(random.randint(0,3))
            cv2.imshow("tmp", cv2.resize(s, (300,300), interpolation=cv2.INTER_AREA))
            cv2.waitKey(30)