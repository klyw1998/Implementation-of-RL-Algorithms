import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # To solve "FigureCanvasAgg is non-interactive" problem under default pycharm
import matplotlib.pyplot as plt


class BernoulliBandit:
    """ 伯努利多臂老虎机，输入k表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) # 随机生成的K个0～1的数作为每根拉杆的获奖概率
        self.best_id_x = np.argmax(self.probs) # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_id_x] # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后，根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # an array of zeros with size K, 储存每根拉杆尝试次数, 初始为0
        self.regret = 0. # 当前步的累积懊悔，初始化为0
        self.actions = [] # 列表，用于记录每一步的动作
        self.regrets = [] # 列表，用于记录每一步的累积懊悔

    def update_regret(self,k):
        # 计算累积懊悔并保存，k为本次动作拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        # code for the choose policy here
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数， num_steps 为运行总次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """ epsilon-Greedy 算法， 继承solver类 """
    def __init__(self, bandit, epsilon = 0.01, init_prob = 1.0):
        super(EpsilonGreedy, self).__init__(bandit) # 这里采用python2 写法以表述清楚类的继承关系
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K) # 初始化K个拉杆的期望奖励估值

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 以 self.epsilon 的概率随机锁定选择一根拉杆
        else:
            k = np.argmax(self.estimates) # 以 (1 - slef.epsilon) 的概率选择期望奖励估值最大的拉杆
        r = self.bandit.step(k) # 得到本次动作的奖励，一定概率为1， 否之为0
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) # 更新拉动拉杆k的动作奖励估值
        return k


class DecayingEpsilonGreedy(Solver):
    """ epsilon随时间衰减的epsilon算法，继承solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.rand() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)  # 以 1 / self.total_count 的概率随机锁定选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 以 (1 - slef.epsilon) 的概率选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励，一定概率为1， 否之为0
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 更新拉动拉杆k的动作奖励估值
        return k


class UCB(Solver):
    """ UCB algorithm，继承Solver类"""
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1))) # calculate UCB
        k = np.argmax(ucb) # choose the bandit with biggest UCB
        r = self.bandit.step(k)  # 得到本次动作的奖励，一定概率为1， 否之为0
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 更新拉动拉杆k的动作奖励估值
        return k


class ThompsonSampling(Solver):
    """ Thompson Sampling algorithm, 继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # 列表，储存各拉杆reward = 1的次数
        self._b = np.ones(self.bandit.K) # 列表，储存各拉杆reward = 0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a,self._b) # 按照 Beta 分布随机取样
        k = np.argmax(samples) # choose the bandit with the biggest sampling
        r = self.bandit.step(k)
        self._a[k] += r # update the first parameter of Beta distribution if r==1
        self._b[k] += (1-r) # update the second parameter of Beta distribution if r==0
        return k


def plot_results(solvers, solver_names):
    """ 生成累积懊悔随着时间变化的图像。输入solvers是一个列表，列表中的每个元素是一种特定的策略。而solver_names也是一个列表，储存每个策略的名称 """
    for id_x, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label = solver_names[id_x])
    plt.xlabel("time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()


K = 10
bandit_10_arm = BernoulliBandit(K)

# np.random.seed(2) # 设置随机种子，使实验具有可重复性
# print("随机生成了一个%d臂伯努利老虎机" % K)
# print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" % (bandit_10_arm.best_id_x, bandit_10_arm.best_prob))


# np.random.seed(2) # 设置随机种子，使实验具有可重复性
# epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon = 0.01)
# epsilon_greedy_solver.run(5000)
# print("epsilon-greedy algorithm's cumulative regrets is: ", epsilon_greedy_solver.regret)
# plot_results([epsilon_greedy_solver],["Epsilon Greedy"])


# np.random.seed(2) # 设置随机种子，使实验具有可重复性
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon = e) for e in epsilons]
# epsilon_greedy_solve_names = ["epsilon = {}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)
# plot_results(epsilon_greedy_solver_list ,epsilon_greedy_solve_names)


# np.random.seed(2) # 设置随机种子，使实验具有可重复性
# decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
# decaying_epsilon_greedy_solver.run(5000)
# print("decaying-epsilon-greedy algorithm's cumulative regrets is: ", decaying_epsilon_greedy_solver.regret)
# plot_results([decaying_epsilon_greedy_solver],["Epsilon Greedy"])


# np.random.seed(2) # 设置随机种子，使实验具有可重复性
# coef = 1 # 控制不确定性比重系数
# UCB_solver = UCB(bandit_10_arm, coef)
# UCB_solver.run(5000)
# print("UCB algorithm's cumulative regrets is: ", UCB_solver.regret)
# plot_results([UCB_solver],["UCB"])


np.random.seed(2) # 设置随机种子，使实验具有可重复性
thompson_sampling_solver = ThompsonSampling(bandit_10_arm )
thompson_sampling_solver.run(5000)
print("Thompson Sampling algorithm's cumulative regrets is: ", thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver],["ThompsonSampling"])