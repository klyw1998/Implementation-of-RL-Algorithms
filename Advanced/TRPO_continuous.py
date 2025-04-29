import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import copy

# Check whether GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def compute_advantage(gamma, lmbda, td_delta):
    # generalized advantage estimation
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mean = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std_dev = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = 2.0 * torch.tanh(self.fc_mean(x))
        std_dev = F.softplus(self.fc_std_dev(x))
        return mean, std_dev # mean and standard deviation of gaussian distribution

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class TRPOContinuous:
    "TRPO algo. for continuous action space"
    def __init__(self, hidden_dim, state_space, action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        # parameters in policy net do not need a optimizer to update
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda # GAE parameter
        self.kl_constraint = kl_constraint # max limit for KL distance
        self.alpha = alpha # linear research parameter
        self.device = device

    def take_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float).to(self.device), 0)
        mean, std_dev = self.actor(state)
        action_dist = torch.distributions.Normal(mean, std_dev)
        action = action_dist.sample()
        return [action.item()]

    def hessian_matrix_vector_product(self, states, action_dists_old, vector, damping=0.1):
        # calculate the product of the hessian matrix and a vector
        mean, std_dev = self.actor(states)
        action_dists_new = torch.distributions.Normal(mean, std_dev)
        kl = torch.mean(torch.distributions.kl.kl_divergence(action_dists_old, action_dists_new)) # calculate the mean kl distance
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(),  create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # calculate the product of the gradient vector of kl distance and the vector
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        # return grad2_vector + damping * vector
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        # conjugate gradient method
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            rdr_new = torch.dot(r, r)
            if rdr_new < 1e-10:
                break
            beta = rdr_new / rdr
            p = r + beta * p
            rdr = rdr_new
        return x

    def compute_surrogate_obj(self, states, actions, advantage, log_probs_old, actor):
        # calculate the policy target
        mean, std_dev = actor(states)
        action_dists = torch.distributions.Normal(mean, std_dev)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - log_probs_old)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, log_probs_old, action_dists_old, max_vec):
        # line search
        para_old= torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        obj_old = self.compute_surrogate_obj(states, actions, advantage, log_probs_old, self.actor)
        for i in range(15):
            coef = self.alpha ** i
            para_new = para_old + coef * max_vec
            actor_new = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(para_new, actor_new.parameters())
            mean, std_dev = self.actor(states)
            action_dists_new = torch.distributions.Normal(mean, std_dev)
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(action_dists_old,action_dists_new))
            obj_new = self.compute_surrogate_obj(states, actions, advantage, log_probs_old, actor_new)
            if obj_new > obj_old and kl_div < self.kl_constraint:
                return para_new
        return para_old
    def policy_learn(self, states, actions, action_dists_old, log_probs_old, advantage):
        # update policy parameters
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, log_probs_old, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # calculate x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states, action_dists_old)
        Hd = self.hessian_matrix_vector_product(states, action_dists_old, descent_direction)
        coef_max = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        para_new = self.line_search(states, actions, advantage, log_probs_old, action_dists_old, descent_direction * coef_max)  # linear search
        torch.nn.utils.convert_parameters.vector_to_parameters(para_new, self.actor.parameters())

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states'])).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).float().view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states'])).to(self.device)
        dones = torch.tensor(transition_dict['dones']).float().view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0 # change the rewards for 
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device) # maintain the critic in cpu but not cuda
        mean, std_dev = self.actor(states)
        action_dists_old = torch.distributions.Normal(mean.detach(), std_dev.detach())
        log_probs_old = action_dists_old.log_prob(actions)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, action_dists_old, log_probs_old, advantage)

num_episodes = 1000
hidden_dim = 128
gamma = 0.98
lmbda =0.95
critic_lr = 1e-2
kl_constraint = 5e-4
alpha = 0.5

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)

# print("State space: ", env.observation_space)
# print("State shape: ", env.observation_space.shape)
# print("- low: ", env.observation_space.low)
# print("- high: ", env.observation_space.high)
# print("State space samples: ", np.array([env.observation_space.sample() for i in range(5)]))
# print("\n")
# print("Action space:", env.action_space)
# print("Action shape:", env.action_space.shape)
# print("Action space samples:", np.array([env.action_space.sample() for i in range(5)]))

agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()