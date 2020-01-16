import gym
import time
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

env = gym.make('CartPole-v1')

def random_policy(obs, command):
    return np.random.randint(env.action_space.n)

#Visualise agent function
def visualise_agent(policy, command, n=5):
    try:
        for trial_i in range(n):
            current_command = deepcopy(command)
            observation = env.reset()
            done=False
            t=0
            episode_return=0
            while not done:
                env.render()
                action = policy(torch.tensor([observation]).double(), torch.tensor([command]).double())
                observation, reward, done, info = env.step(action)
                episode_return+=reward
                current_command[0]-= reward
                current_command[1] = max(1, current_command[1]-1)
                t+=1
            env.render()
            time.sleep(1.5)
            print("Episode {} finished after {} timesteps. Return = {}".format(trial_i, t, episode_return))
        env.close()
    except KeyboardInterrupt:
        env.close()
        
#Behaviour function - Neural Network
class FCNN_AGENT(torch.nn.Module):
    def __init__(self, command_scale):
        super().__init__()
        hidden_size=64
        self.command_scale=command_scale
        self.observation_embedding = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), hidden_size),
            torch.nn.Tanh()
        )
        self.command_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_size),
            torch.nn.Sigmoid()
        )
        self.to_output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, env.action_space.n)
        )
    
    def forward(self, observation, command):
        obs_emebdding = self.observation_embedding(observation)
        cmd_embedding = self.command_embedding(command*self.command_scale)
        embedding = torch.mul(obs_emebdding, cmd_embedding)
        action_prob_logits = self.to_output(embedding)
        return action_prob_logits
    
    def create_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

#Fill the replay buffer with more experience
def collect_experience(policy, replay_buffer, replay_size, last_few, n_episodes=100, log_to_tensorboard=True):
    global i_episode
    init_replay_buffer = deepcopy(replay_buffer)
    try:
        for _ in range(n_episodes):
            command = sample_command(init_replay_buffer, last_few)
            if log_to_tensorboard: writer.add_scalar('Command desired reward/Episode', command[0], i_episode)    # write loss to a graph
            if log_to_tensorboard: writer.add_scalar('Command horizon/Episode', command[1], i_episode)    # write loss to a graph
            observation = env.reset()
            episode_mem = {'observation':[],
                           'action':[],
                           'reward':[],}
            done=False
            while not done:
                action = policy(torch.tensor([observation]).double(), torch.tensor([command]).double())
                new_observation, reward, done, info = env.step(action)
                
                episode_mem['observation'].append(observation)
                episode_mem['action'].append(action)
                episode_mem['reward'].append(reward)
                
                observation=new_observation
                command[0]-= reward
                command[1] = max(1, command[1]-1)
            episode_mem['return']=sum(episode_mem['reward'])
            episode_mem['episode_len']=len(episode_mem['observation'])
            replay_buffer.append(episode_mem)
            i_episode+=1
            if log_to_tensorboard: writer.add_scalar('Return/Episode', sum(episode_mem['reward']), i_episode)    # write loss to a graph
            print("Episode {} finished after {} timesteps. Return = {}".format(i_episode, len(episode_mem['observation']), sum(episode_mem['reward'])))
        env.close()
    except KeyboardInterrupt:
        env.close()
    replay_buffer = sorted(replay_buffer, key=lambda x:x['return'])[-replay_size:]
    return replay_buffer

#Sample exploratory command
def sample_command(replay_buffer, last_few):
    if len(replay_buffer)==0:
        return [1, 1]
    else:
        command_samples = replay_buffer[-last_few:]
        lengths = [mem['episode_len'] for mem in command_samples]
        returns = [mem['return'] for mem in command_samples]
        mean_return, std_return = np.mean(returns), np.std(returns)
        command_horizon = np.mean(lengths)
        desired_reward = np.random.uniform(mean_return, mean_return+std_return)
        return [desired_reward, command_horizon]

#Improve behviour function by training on replay buffer
def train_net(policy_net, replay_buffer, n_updates=100, batch_size=64, log_to_tensorboard=True):
    global i_updates
    all_costs = []
    for i in range(n_updates):
        batch_observations = np.zeros((batch_size, np.prod(env.observation_space.shape)))
        batch_commands = np.zeros((batch_size, 2))
        batch_label = np.zeros((batch_size))
        for b in range(batch_size):
            sample_episode = np.random.randint(0, len(replay_buffer))
            sample_t1 = np.random.randint(0, len(replay_buffer[sample_episode]['observation']))
            sample_t2 = len(replay_buffer[sample_episode]['observation'])
            ##sample_t2 = np.random.randint(sample_t1+1, len(replay_buffer[sample_episode]['observation'])+1)
            sample_horizon = sample_t2-sample_t1
            sample_mem = replay_buffer[sample_episode]['observation'][sample_t1]
            sample_desired_reward = sum(replay_buffer[sample_episode]['reward'][sample_t1:sample_t2])
            network_input = np.append(sample_mem, [sample_desired_reward, sample_horizon])
            label = replay_buffer[sample_episode]['action'][sample_t1]
            batch_observations[b] = sample_mem
            batch_commands[b] = [sample_desired_reward, sample_horizon]
            batch_label[b] = label
        batch_observations = torch.tensor(batch_observations).double()
        batch_commands = torch.tensor(batch_commands).double()
        batch_label = torch.tensor(batch_label).long()
        pred = policy_net(batch_observations, batch_commands)
        cost = F.cross_entropy(pred, batch_label)
        if log_to_tensorboard: writer.add_scalar('Cost/NN update', cost.item() , i_updates)    # write loss to a graph
        all_costs.append(cost.item())
        cost.backward()
        policy_net.optimizer.step()
        policy_net.optimizer.zero_grad()
        i_updates+=1
    return np.mean(all_costs)

#Return a greedy policy from a given network
def create_greedy_policy(policy_network):
    def policy(obs, command):
        action_logits = policy_network(obs, command)
        action_probs = F.softmax(action_logits, dim=-1)
        action = np.argmax(action_probs.detach().numpy())
        return action
    return policy

#Return a stochastic policy from a given network
def create_stochastic_policy(policy_network):
    def policy(obs, command):
        action_logits = policy_network(obs, command)
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.distributions.Categorical(action_probs).sample().item()
        return action
    return policy

#Initialize vars
i_episode=0 #number of episodes trained so far
i_updates=0 #number of parameter updates to the neural network so far
replay_buffer = []
log_to_tensorboard = False

## HYPERPARAMS
replay_size = 700
last_few = 50
batch_size = 256
n_warm_up_episodes = 50
n_episodes_per_iter = 25
n_updates_per_iter = 15
command_scale = 0.02
lr = 0.001

# Initialize behaviour function
agent = FCNN_AGENT(command_scale).double()
agent.create_optimizer(lr)

stochastic_policy = create_stochastic_policy(agent)
greedy_policy = create_greedy_policy(agent)

# SET UP TRAINING VISUALISATION
if log_to_tensorboard: from torch.utils.tensorboard import SummaryWriter
if log_to_tensorboard: writer = SummaryWriter() # we will use this to show our models performance on a graph using tensorboard

#Collect warm up episodes
replay_buffer = collect_experience(random_policy, replay_buffer, replay_size, last_few, n_warm_up_episodes, log_to_tensorboard)
train_net(agent, replay_buffer, n_updates_per_iter, batch_size, log_to_tensorboard)

#Collect experience and train behaviour function for given number of iterations
n_iters = 1000
for i in range(n_iters):
    replay_buffer = collect_experience(stochastic_policy, replay_buffer, replay_size, last_few, n_episodes_per_iter, log_to_tensorboard)
    train_net(agent, replay_buffer, n_updates_per_iter, batch_size, log_to_tensorboard)

#Visualise final trained agent with greedy policy
visualise_agent(greedy_policy, command=[150, 400], n=5)

#Visualise final trained agent with stochastic policy
visualise_agent(stochastic_policy, command=[150, 400], n=5)
