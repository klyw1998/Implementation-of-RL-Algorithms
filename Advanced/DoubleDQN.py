import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import rl_utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    "DQN algorithm, including double DQN"
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # Q network
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # target network
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr) # use adam optimizer
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # epsilon-greedy policy
        self.target_update = target_update # target network update frequence
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state): # take action following epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return  self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).float().view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states'])).to(self.device)
        dones = torch.tensor(transition_dict['dones']).float().view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions) # Q_value
        if self.dqn_type == "DoubleDQN": # DoubleDQN
            max_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_actions)
        if self.dqn_type == "VanillaDQN": # DQN
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # TD loss target
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) # mean squared error loss function
        self.optimizer.zero_grad() # the default gradient will be accumulated in PyTorch so we need to set it to zero
        dqn_loss.backward() # update parameters backwards
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # update target net
        self.count +=1


lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64

env_name = 'Pendulum-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 21 # separate continuous action to discrete actions

def dis_to_con(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, info = env.reset()
                episode_over = False
                while not episode_over:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995 # smoothing
                    max_q_value_list.append(max_q_value)  # save max Q for each state
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, done, truncated, info = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    episode_over = done or truncated
                    episode_return += reward
                    state = next_state
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list, max_q_value_list

random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
# agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, "VanillaDQN")
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, "DoubleDQN")
return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('DQN on {}'.format(env_name))
plt.title('DoubleDQN on {}'.format(env_name))
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
# plt.title('DQN on {}'.format(env_name))
plt.title('DoubleDQN on {}'.format(env_name))
plt.show()