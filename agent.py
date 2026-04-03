import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from q_network_model import QNetworkMLP, QNetworkCNN

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


optimizer=optim.Adam
loss_fuction=nn.MSELoss()

memory_length=10000

class AgentMLP():
    def __init__(self, input_size, hidden_size, output_size=3):


        self.gamma=0.99
        self.epsilon=1
        self.epsilon_min=0.0001
        self.epsilon_decay=0.9995
        self.learning_rate = 0.001



        self.model=QNetworkMLP(input_size, hidden_size, output_size)
        self.model.to(device)

        self.target_model=QNetworkMLP(input_size, hidden_size, output_size)
        self.target_model.to(device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_function= loss_fuction

        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        
        self.memory = deque(maxlen=memory_length)



    def get_action(self, state):

        if random.random()<=self.epsilon:
            return random.randint(0,2)
        else:
            with torch.no_grad():
                q_values=self.model(state)
                action=torch.argmax(q_values).item()
        return action
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self,batch_size):
        if len(self.memory)<batch_size:
            return
        else:
            mini_batch=random.sample(self.memory, batch_size)
            states = torch.stack([m[0] for m in mini_batch]).to(device)
            states=states.squeeze(1)
            #states shape=[3,25]
            actions = torch.tensor([m[1] for m in mini_batch], dtype=torch.int64, device=device)
            actions=actions.unsqueeze(1)
            #actions shape: [3,1]
            rewards = torch.tensor([m[2] for m in mini_batch], dtype=torch.float32, device=device)
            rewards=rewards.unsqueeze(1)
            #rewards shape: [3,1]
            next_states = torch.stack([m[3] for m in mini_batch]).to(device)
            next_states=next_states.squeeze(1)
            #next_states shape=[3,25]
            
            dones = torch.tensor([m[4] for m in mini_batch], dtype=torch.float32, device=device)
            dones=dones.unsqueeze(1)
            #dones shape: [3,1] 
            with torch.no_grad():
                q_target_values=self.target_model.forward(next_states)
                max_q=torch.max(q_target_values,dim=1)[0].unsqueeze(1)
                target=rewards + self.gamma * max_q*(1-dones)

            q_values=self.model.forward(states)
            prediction=torch.gather(q_values,1,actions)
            prediction=prediction
            loss=self.loss_function(prediction,target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
            
class AgentCNN():
    def __init__(self, in_channels, width, height, hidden_size, output_size=3):


        self.gamma=0.99
        self.epsilon=1
        self.epsilon_min=0.0001
        self.epsilon_decay=0.9995
        self.learning_rate = 0.001



        self.model=QNetworkCNN(in_channels, width, height, hidden_size, output_size)
        self.model.to(device)

        self.target_model=QNetworkCNN(in_channels, width, height, hidden_size, output_size)
        self.target_model.to(device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_function= loss_fuction

        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        
        self.memory = deque(maxlen=memory_length)



    def get_action(self, state):

        if random.random()<=self.epsilon:
            return random.randint(0,2)
        else:
            with torch.no_grad():
                q_values=self.model(state)
                action=torch.argmax(q_values).item()
        return action
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self,batch_size):
        if len(self.memory)<batch_size:
            return
        else:
            mini_batch=random.sample(self.memory, batch_size)
            states = torch.stack([m[0] for m in mini_batch]).to(device)
            #states shape: [batch_size, 5, 5, 4]

            actions = torch.tensor([m[1] for m in mini_batch], dtype=torch.int64, device=device)
            actions=actions.unsqueeze(1)

            #actions shape: [batch_size,1]
            rewards = torch.tensor([m[2] for m in mini_batch], dtype=torch.float32, device=device)
            rewards=rewards.unsqueeze(1)

            #rewards shape: [batch_size,1]
            next_states = torch.stack([m[3] for m in mini_batch]).to(device)
            #next_states shape=[batch_size, 5, 5, 4]
            
            dones = torch.tensor([m[4] for m in mini_batch], dtype=torch.float32, device=device)
            dones=dones.unsqueeze(1)
            #dones shape: [batch_size, 1] 

            with torch.no_grad():
                q_target_values=self.target_model.forward(next_states)
                max_q=torch.max(q_target_values,dim=1)[0].unsqueeze(1)
                target=rewards + self.gamma * max_q*(1-dones)

            q_values=self.model.forward(states)
            prediction=torch.gather(q_values,1,actions)
            prediction=prediction
            loss=self.loss_function(prediction,target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
            return loss.item()
        