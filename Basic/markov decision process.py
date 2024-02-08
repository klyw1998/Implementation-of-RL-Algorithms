import numpy as np
np.random.seed(0)
# define probability matrix for the state transformation
P = [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
     [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
     [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0] # define reward function
gamma = 0.5 # define discount factor

# given a sequence, compute the reward from the start state, following the sequence chain
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

# # for example, a state sequence is s1-s2-s3-s6
# chain = [1,2,3,6]
# start_index = 0
# G = compute_return(start_index, chain, gamma)
# print('The reward is: %s' %G)

def compute(P, rewards, gamma, states_num):
    ''' Compute the analytical solution using the matrix form of the Bellman equation, state_num is the number of states in Markov Reward Process'''
    rewards = np.array(rewards).reshape((-1, 1)) # reshape the rewards into the form of column vector
    value =  np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

# V = compute(P, rewards, gamma, 6)
# print('every state value of Markov Reward Process is separately : \n', V)

S = ['s1','s2','s3','s4','s5'] # state set
A = ['stay at s1','go to s1','go to s2','go to s3','go to s4','go to s5','possibly go to'] # action set

# state transition probability
P = {
    's1-stay at s1-s1': 1.0,"s1-go to s2-s2": 1.0,
    's2-go to s1-s1': 1.0,'s2-go to s3-s3': 1.0,
    's3-go to s4-s4': 1.0,'s3-go to s5-s5': 1.0,
    's4-go to s5-s5': 1.0,'s4-possibly go to-s2': 0.2,
    's4-possibly go to-s3': 0.4,'s4-possibly go to-s4': 0.4,
}

# action's reward function
R = {
    's1-stay at s1':-1, 's1-go to s2':0,
    's2-go to s1':-1, 's2-go to s3':-2,
    's3-go to s4':-2, 's3-go to s5':0,
    's4-go to s5':10, 's4-possibly go to':1,
}
gamma = 0.5 # discount factor
MDP = (S, A, P, R, gamma)

# policy 1, random
Pi_1 = {
    's1-stay at s1':0.5, 's1-go to s2':0.5,
    's2-go to s1':0.5, 's2-go to s3':0.5,
    's3-go to s4':0.5, 's3-go to s5':0.5,
    's4-go to s5':0.5, 's4-possibly go to':0.5,
}

# policy 2
Pi_2 = {
    's1-stay at s1':0.6, 's1-go to s2':0.4,
    's2-go to s1':0.3, 's2-go to s3':0.7,
    's3-go to s4':0.5, 's3-go to s5':0.5,
    's4-go to s5':0.1, 's4-possibly go to':0.9,
}

# connect 2 string with '-' so that we can use P and R
def join(str1, str2):
    return str1 + '-' + str2

# # probability matrix for transfermation from mdp to mrp
# P_from_mdp_to_mrp = [
#     [0.5, 0.5, 0.0, 0.0, 0.0],
#     [0.5, 0.0, 0.5, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.5, 0.5],
#     [0.0, 0.1, 0.2, 0.2, 0.5],
#     [0.0, 0.0, 0.0, 0.0, 1.0],
# ]
# probability_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
# print(P_from_mdp_to_mrp)
# rewards_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0.0]
#
# V_test = compute(probability_from_mdp_to_mrp, rewards_from_mdp_to_mrp, gamma, 5)
# print('the state values in MDP are', V_test)

def sample(MDP, Pi, timestep_max, number):
    ''' Sample funtction. Policy Pi. Time step is limited. Number is the number of the sample sequence.'''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] # randomly choose a start state except 's5', which is S[0], S[1], S[2] or S[3]
        # one sampling will be finished if current state is the end state or the time steps are too long
        while s != 's5' and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # at state 's', choose action according to policy Pi
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0) # dict.get(key[, value]), str.join(sequence)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # receive the next state 's_next' according to state transformation probability matrix
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next)) # append tuple (s,a,r,s_next) to the episode sequence
            s = s_next # s_next is the current state
        episodes.append(episode)
    return episodes


# # sample five times and timestep_max is no larger than 20
# episodes = sample(MDP, Pi_1, 20, 5)
# print('the first sequence \n', episodes[0])
# print('the second sequence \n', episodes[1])
# print('the fifth sequence \n', episodes[4])


# calculate the values of all states for all sampling sequences
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1): # calculate backwards in a sequence
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] += 1
            V[s] += (G - V[s]) / N[s]


# timestep_max = 20
# # sample 1000 times
# episodes = sample(MDP, Pi_1, timestep_max, 1000)
# gamma = 0.5
# V = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
# N = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
# MC(episodes, V, N, gamma)
# print('the states values in MDP by using Monte Carlo method are\n', V)


def occupancy(episodes, s, a, timestep_max, gamma):
    '''calculate visitation frequence of pair (s, a) to estimate the occupancy measure of certain policy'''
    rho = 0
    total_times = np.zeros(timestep_max) # record how many times every timestep t has been visited
    occur_times = np.zeros(timestep_max) # record the occur times of (s_t, a_t) = (s, a)
    for episode in episodes:
        for i in range(len(episode)): # an example of episode[i] is [('s1', 'go to s2', 0, 's2'), ('s2', 'go to s3', -2, 's3'), ('s3', 'go to s5', 0, 's5')]
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in range(timestep_max):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, 's4', 'possibly go to', timestep_max, gamma)
rho_2 = occupancy(episodes_2, 's4', 'possibly go to', timestep_max, gamma)
print(rho_1, rho_2)