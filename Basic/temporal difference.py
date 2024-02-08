import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0 # Agent's Current Position's x-coordinate
        self.y = self.nrow - 1 # Agent's Current Position's y-coordinate

    def step(self, action): # call this external function to change current position
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self): # return to the origin point
        self.x =0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

class Sarsa:
    ''' Sarsa Algorithm '''
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # Initialize Q(s, a) table
        self.n_action = n_action
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # parameter for epsilon greedy

    def take_action(self, state): # choose next operation via epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state): # to print the policy
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


class nstep_Sarsa:
    ''' N-step Sarsa Algorithm '''
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # Initialize Q(s, a) table
        self.n_action = n_action
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon
        self.n = n # n for n-step sarsa
        self.state_list = [] # keep previous states
        self.action_list = [] # keep previous actions
        self.reward_list = [] # keep previous rewards

    def take_action(self, state): # choose next operation via epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state): # to print the policy
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n: # if current state list can be updated for n steps
            G = self.Q_table[s1, a1] # get Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i] # calculate rewards backwards, considering discount factor gamma
                # if reach the GOAL!, the last several steps (if exist) also will be updated
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []


class QLearning:
    ''' Q-Learning Algorithm '''
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # Initialize Q(s, a) table
        self.n_action = n_action
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # parameter for epsilon greedy

    def take_action(self, state): # choose next operation via epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state): # to print the policy
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)

np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
n_step = 5
#agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
#agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500

return_list = [] # record the reward of every sequence

# for i in range(10): # show 10 process bars
#     # process bar function of 'tqdm'
#     with tqdm(total = int(num_episodes / 10), desc = 'Iteration %d' % i) as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#             state = env.reset()
#             action = agent.take_action(state)
#             done = False
#             while not done:
#                 next_state, reward, done = env.step(action)
#                 next_action = agent.take_action(next_state)
#                 episode_return += reward # without considering discount factor decaying
#                 #agent.update(state, action, reward, next_state, next_action)
#                 agent.update(state, action, reward, next_state, next_action, done)
#                 state = next_state
#                 action = next_action
#             return_list.append(episode_return)
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
#             pbar.update(1)

for i in range(10): # show 10 process bars
    # process bar function of 'tqdm'
    with tqdm(total = int(num_episodes / 10), desc = 'Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward # without considering discount factor decaying
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('Sarsa on {}'.format('Cliff Walking'))
# plt.title('ntep_Sarsa on {}'.format('Cliff Walking'))
plt.title('Q-Learning on {}'.format('Cliff Walking'))
plt.show()

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('XXXX', end = ' ')
            elif (i * env.ncol + j) in end:
                print('GOAL!', end = ' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'O'
                print(pi_str, end=' ')
        print()

action_meaning = ['A', 'V', '<', '>']
print('The final converged policy via Sarsa Algorithm is :')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])