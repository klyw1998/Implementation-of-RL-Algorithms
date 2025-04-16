import numpy as np
import gymnasium as gym
# import gym
# from gym.version import VERSION
import torch
import torch.nn.functional as F
import random
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import rl_utils


# Check torch version and gym version
print(torch.__version__)
# print(VERSION)

# Check whether GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


class ReplayBuffer:
    "Experience Replay Buffer"
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size) # queue, FIFO

    def add(self, state, action, reward, next_state, done): # add data into buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): # sample data from the buffer, the amount is batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    "Q-Network wirh only one hidden layer"
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Use the ReLU activation function in the hidden layer
        return self.fc2(x)

class DQN:
    "DQN algorithm"
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # Q network
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # target network
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr) # use adam optimizer
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # epsilon-greedy policy
        self.target_update = target_update # target network update frequence
        self.count = 0
        self.device = device

    def take_action(self, state): # take action following epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state_current = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state_current).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states']).float().to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states']).to(self.device)
        dones = torch.tensor(transition_dict['dones']).float().view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions) # Q_value
        # biggest q_value in next state
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # TD loss target
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) # mean squared error loss function
        self.optimizer.zero_grad() # the default gradient will be accumulated in PyTorch so we need to set it to zero
        dqn_loss.backward() # update parameters backwards
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # update target net
        self.count +=1


lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64

env_name = 'CartPole-v1'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state, info = env.reset()
            episode_over = False
            while not episode_over:
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                episode_over = done or truncated
                episode_return += reward
                state = next_state
                # when buffer size is larger than minimal size, do the Q net training
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()