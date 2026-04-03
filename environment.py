import torch
import random
import torch.nn.functional as F

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class snake():
    def __init__(self):

        self.length=1
        self.head=None
        self.body=[]
        self.score=0
        self.direction=(0,1)
        

class snake_environment():
    def __init__(self, width, height):
        self.death_reason=""
        if width<5 or height <5 :
            raise ValueError("minimal board is 5x5")
        self.d_model=4
        self.width=width
        self.height=height
        self.spawn_snake()
        self.gameover=False
        self.fruit_pos = None 
        self.steps_left=self.width*self.height*2

        self.fruit_spawn()
    def spawn_snake(self):
        self.snake=snake()
        x=random.randrange(1,self.width-2)
        y=random.randrange(1,self.height-2)
        for i in range (self.snake.length):
            if len(self.snake.body)==0 :
                self.snake.body.append((x,y))
            r=random.randint(0, 3)
            if r==0:
                self.snake.body.append((x+1,y))
                self.snake.direction=(1,0)
            elif r==1:
                self.snake.body.append((x-1,y))
                self.snake.direction=(-1,0)
            elif r==2 :
                self.snake.body.append((x,y+1))
                self.snake.direction=(0,-1)
            elif r==3:
                self.snake.body.append((x,y-1))
                self.snake.direction=(0,1)


            x=self.snake.body[-1][0]
            y=self.snake.body[-1][1]
        # TODO: make this more robust to any length   
        self.snake.head=self.snake.body[-1]



    def reset(self):
        self.spawn_snake()
        
        self.steps_left=self.width*self.height*2

        self.gameover=False
        self.fruit_pos=None

        self.fruit_spawn()
    def render(self):
        grid = [['_' for _ in range(self.width)] for _ in range(self.height)]
        for x in range (self.width):
            for y in range (self.height):
                if (x, y)== self.snake.head :
                    grid[y][x] ="X"
                elif (x, y) in self.snake.body :
                    grid[y][x] ="O"
                elif (x, y)== self.fruit_pos:
                    grid[y][x] ="F"

        for y in grid:
            print(' '.join(y))
    def fruit_spawn(self):
        # Note: For a very large snake filling the board
        while True :
            x=random.randint(0,self.width-1)
            y=random.randint(0,self.height-1)
            if (x,y) not in self.snake.body :
                self.fruit_pos=(x,y)
                break

    
    def get_state(self):
        state=torch.zeros((self.width,self.height), dtype=torch.int64, device=device)
        if self.gameover== True :
            encoded_state= F.one_hot(state, num_classes=self.d_model)
        else:
            for x in range (self.width) :
                for y in range (self.height) :
                    if (x, y)==self.fruit_pos :
                        state[x][y]= 2
                    elif (x, y) == self.snake.head: 
                        state[x][y] = 1
                    elif (x, y) in self.snake.body:
                        state[x][y] = 3
            encoded_state= F.one_hot(state, num_classes=self.d_model)
        encoded_state=encoded_state.to(torch.float32)
        return encoded_state
        



        
    def step(self,action):
        reward=0
        if self.gameover:
            reward+=-1.0
            print("game ended")
            return (self.get_state(), reward, self.gameover)
        self.steps_left -= 1
        current_manhaten_distance=abs(self.snake.head[0]-self.fruit_pos[0])+abs(self.snake.head[1]-self.fruit_pos[1])

        current_direction=self.snake.direction
        #  0_straight // 1_right // 2_left
        if action==0 :
            new_direction=current_direction
        elif action==1 :
            new_direction=(current_direction[1],-current_direction[0])
        elif action==2 :
            new_direction=(-current_direction[1],current_direction[0])
        else:
            raise ValueError("action not 0 , 1 or 2")

        self.snake.direction=new_direction
        #snake moving part
        new_x = self.snake.head[0] + self.snake.direction[0]
        new_y = self.snake.head[1] - self.snake.direction[1]
        new_head = (new_x, new_y)

        new_manhaten_distance=abs(new_head[0]-self.fruit_pos[0])+abs(new_head[1]-self.fruit_pos[1])
        if current_manhaten_distance<new_manhaten_distance :
            reward-=0.01
        else:
            reward+=0.01
        

        if not (-1<new_head[0]<self.width) or not(-1<new_head[1]<self.height) or new_head in self.snake.body:
            self.gameover=True
            if (new_head in self.snake.body):
                self.death_reason = 'body'
            if (not (-1<new_head[0]<self.width) or not(-1<new_head[1]<self.height)):
                self.death_reason = 'wall'
            reward+=-1.0
            return (self.get_state(), reward, self.gameover) 
        elif  self.steps_left <= 0:
            self.gameover=True
            reward+=-1.0
            return (self.get_state(), reward, self.gameover)
        elif self.width*self.height<len(self.snake.body)+1:
            self.gameover=True
            reward+=10.0
            self.snake.score +=100
            return (self.get_state(), reward, self.gameover)
        elif new_head == self.fruit_pos :
            self.steps_left = self.width*self.height*2
            self.snake.body.append(new_head)
            self.snake.head=new_head
            self.fruit_spawn()
            reward+=2.0
            self.snake.score +=1
            return (self.get_state(), reward, self.gameover)
        
        self.snake.body.append(new_head)
        self.snake.head=new_head
        self.snake.body.pop(0)

        return (self.get_state(), reward, self.gameover)
