import copy
import gym


class CliffWalkingEnv:
    """Cliff-walking Environment"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol # define grid world's column
        self.nrow = nrow # define grid world's row
        # transformation matrix P[state][action] = [(p, next_state, reward, done)] includes next state and reward
        self.P = self.createP()

    def createP(self):
        # initialization
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # four kinds of movement, change[0]:up, change[1]:down, change[2]:left, change[3]:right. Original point is (0,0) which is at the topleft corner
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol): # point(j,i)
                for a in range(4):
                    # if the position is at the cliff or the target point, the reward is zero as there is no more interaction
                    if i == self.nrow - 1 and j > 0: # when i ==  self.nrow - 1, the agent is at the cliff (the last row)
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)] # [(probability, next_state, reward, done)]
                        continue
                    # other positions
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # if the next position is at the cliff or the target point
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    """ Policy Iteration ALgorithm """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow # initialized the value of states as 0's
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)] # set the original policy as go to the four direction with same probability
        self.theta = theta # the threshold to stop the policy iteration
        self.gamma = gamma # discount factor

    def policy_evaluation(self): # as the name said
        cnt = 1 # counter
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] # calculate the Q(s,a) value in state s
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
            cnt += 1
        print('policy evaluation completes after %d rounds' %cnt)

    def policy_improvement(self): # as the name said
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # count how many times the max q value appears
            # set the probability evenly for actions with the max Q value(s) and set 0 for other actions
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print('Policy Improvement Completed')
        return self.pi

    def policy_iteration(self): # as the name said
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi) # deepcopy the list for further comparison
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break


class ValueIteration:
    """Policy Iteration Algorithm"""
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
            cnt += 1
        print('value iteration completes after %d rounds' %cnt)
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # count how many times the max q value appears
            # set the probability evenly for actions with the max Q value(s) and set 0 for other actions
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print('The State Values are:')
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print('The Policy is:')
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print('XXXX', end = ' ')
            elif (i * agent.env.ncol + j) in end:
                print('GOAL!', end = ' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'O'
                print(pi_str, end=' ')
        print()


# env = CliffWalkingEnv()
# action_meaning = ['A', 'V', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent1 = PolicyIteration(env, theta, gamma) # definition
# agent1.policy_iteration() # calculation
# print_agent(agent1, action_meaning, list(range(37, 47)), [47])
#
# agent2 = ValueIteration(env, theta, gamma) # definition
# agent2.value_iteration() # calculation
# print_agent(agent2, action_meaning, list(range(37, 47)), [47])


env = gym.make("FrozenLake-v1", render_mode='ansi') # create the environment
env = env.unwrapped # unwrapped so that states transformation matrix P can be visited ## this line could be deleted, not a big deal
env.reset()
print(env.render()) # environment rendering

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0: # if the reward is 1, the location is (one of ) the end(s)
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print('Index of holes: ', holes) # attention: the index of one set is disordered
print('Index of ends: ', ends)

for a in env.P[10]: # print the state transformation info of the one-step top-left point
    print(env.P[10][a])
for a in env.P[11]: # print the state transformation info of the one-step top point
    print(env.P[11][a])
for a in env.P[14]: # print the state transformation info of the one-step left point
    print(env.P[14][a])

action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent3 = PolicyIteration(env, theta, gamma)
agent3.policy_iteration()
print_agent(agent3, action_meaning, [5, 7, 11, 12], [15])

agent4 = ValueIteration(env, theta, gamma)
agent4.value_iteration()
print_agent(agent4, action_meaning, [5, 7, 11, 12], [15])