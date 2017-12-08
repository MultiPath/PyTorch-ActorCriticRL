import gym
import logging
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.multiprocessing as mp
import copy

import time
import random
import numpy as np
import gc
import os
import subprocess

from collections import deque
from torch.autograd import Variable
from gym import wrappers
from collections import namedtuple
from mpi4py import MPI

# prepare the environment
gym.undo_logger_setup()
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -- hyper parameters

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# setup random seeds

MAX_EPISODES = 5000
MAX_STEPS = 3000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
WORKERS = 65
TRIAL = 4
ELITES = 4
BATCH_SIZE = 128
INNER_STEP = 5
SEED = 206
SIGMA = 0.10
gpu = -1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu > -1:
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31-1)
    def next_seed(self):
        result = np.random.randint(self.limit)
        return result
    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result

class StateBuffer:
    
	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		count = min(count, self.len)
		batch = np.float32(random.sample(self.buffer, count))
		return batch

	def len(self):
		return self.len

	def add(self, states):
		self.len += len(states)
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.extend(states)


Game = namedtuple('Game', 
    ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise'])
games = {}

bipedhard = Game(env_name = 'BipedalWalkerHardcore-v2', # 'BipedalWalker-v2'
                input_size=24,
                output_size=4,
                time_factor=0,
                layers=[40, 40],
                activation='tanh',
                noise_bias=0.0,
                output_noise=[False, False, False])
games['bipedhard'] = bipedhard

def make_env(env_name, seed=-1):
    env = gym.make(env_name)
    if seed >= 0:
        env.seed(seed)
    return env


# ---------------- SGN ----------------------------------------------------- #
class SGNET(nn.Module):
    def __init__(self, game):
        super(SGNET, self).__init__()
        self.fc11 = nn.Linear(game.input_size, 40)
        self.fc12 = nn.Linear(game.output_size, 40)
        self.fc2  = nn.Linear(80, 40)
        self.fc3  = nn.Linear(40, game.output_size)

    def forward(self, a, s):
        s1 = F.relu(self.fc11(s))
        a1 = F.relu(self.fc12(a))
        x = torch.cat([s1, a1], dim=1)
        return self.fc3(F.relu(self.fc2(x)))

class SGN(nn.Module):
    
    def __init__(self, model):
        super(SGN, self).__init__()
        self.saved_variables = []
        self.net = model

    def forward(self, a=None, s=None, grad_y=None, mode='forward'):
        if mode == 'forward':
            y = a.mean(-1)
            self.saved_variables = [Variable(a), Variable(s)]
            return y

        else:
            a, s = self.saved_variables
            syn_grad_a = self.net(a, s) / a.size(0)
            return syn_grad_a


class SurrogateFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a, s, sgn=None):
        x = sgn(a=a, s=s, mode='forward')
        ctx.sgn = sgn  
        return x

    @staticmethod
    def backward(ctx, grad_y):
        grad_x = ctx.sgn(grad_y=grad_y, mode='backward')
        return grad_x, None, None

surrogate = SurrogateFunction.apply


# ---------------- AGENT ---------------------------------------------------- #
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class Agent(nn.Module):
    ''' simple feedforward model '''
    def __init__(self, game):
        super(Agent, self).__init__()
        self.env_name = game.env_name
        self.id = -1

        # main network
        self.net = nn.ModuleList()
        input_size = game.input_size
        for size in game.layers:
            self.net.append(nn.Linear(input_size, size))
            input_size = size
        self.output = nn.Linear(input_size, game.output_size)
        self.env = None

    def reset(self):
        for l in self.net:
            l.reset_parameters()
        self.output.reset_parameters()

    def make_env(self, seed=-1, video_callable=False):
        if self.env is None:
            self.env = make_env(self.env_name, seed=seed)
            if video_callable:
                self.env = wrappers.Monitor(self.env, video_callable=video_callable,
                                            directory='./{}/'.format(self.env_name), force=True)
    
    def forward(self, ob, sgn=None):
        x = ob
        for l in self.net:
            x = F.relu(l(x))
        ac = F.tanh(self.output(x))

        if sgn is None:
            return ac, None
        else:
            return ac, surrogate(ac, ob, sgn).mean()           


    def act(self, observation, gpu=-1, sgn=None):
        observation = torch.Tensor(observation)
        if gpu > -1:
            observation = observation.cuda(gpu)
        ob = Variable(observation)
        ac, fake_loss = self(ob, sgn=sgn)

        with torch.cuda.device(gpu): 
            ac = ac.cpu().data.tolist()
        return ac, fake_loss

# ---------------- Simulator ---------------------------------------------------- #

class Simulator(object):
    
    def __init__(self, agent, sgn, gpu=-1):
        self.gpu = gpu
        self.agent = agent
        self.sgn = sgn
        self.optimizer = torch.optim.Adam(self.agent.parameters(), 0.001) 
        self.optim_sgn = torch.optim.Adam(self.sgn.parameters(), 0.01) 

        if self.gpu > -1:
            self.agent = agent.cuda(gpu)
            self.sgn = agent.cuda(gpu)
        
        dummy_loss = 0
        for p in sgn.parameters():
            dummy_loss += torch.mean(p)
        dummy_loss.backward()

    def agent_update(self, observations):
        observations = torch.from_numpy(observations)
        if self.gpu > -1:
            observations = observations.cuda(self.gpu)
        observations =  Variable(observations)
        predict_actions, fake_loss = self.agent(observations, self.sgn)
        self.optimizer.zero_grad()
        fake_loss.backward()
        self.optimizer.step()

    def simulate(self, episodes=1, seed=None, max_step=-1, train=True):
        steps, rewards, states = [], [], []
        agent = self.agent
        for ep in range(episodes):
            if seed is not None:            
                set_seed(seed[ep])

            ob = agent.env.reset()
            total_reward = 0
            stumbled = False

            if max_step == -1:
                max_step = MAX_STEPS

            for step in range(max_step):
                action, _ = agent.act(ob, gpu=self.gpu)
                new_ob, reward, done, info = agent.env.step(action)

                if train and  (reward == -100):
                    stumbled = True
                    reward = 0
                total_reward += reward

                if done:
                    if train and (not stumbled) and (total_reward > MAX_TOTAL_REWARD):
                        total_reward += 100    
                    break   

                states.append(ob)
                ob = new_ob   # important need to update the observation

            rewards.append(total_reward)
            steps.append(step)

        rewards = np.mean(rewards)
        steps = np.mean(steps)
        return rewards, steps, states


# ------------------- EvoGrad --------------------------------------- #
class EvoGrad(object):
    
    def __init__(self, simulator, 
                sigma_init=0.1,
                sigma_alpha=0.1):   # use PEPG for EvoGradient
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_max_change = 0.2
        self.sigma_limit = 0.01
        self.sigma_decay = 0.999

        self.simulator = simulator
        self.sigma = {name: p.data * 0 + sigma_init 
                        for name, p in self.simulator.sgn.named_parameters()}
        self.population = WORKERS - 1
        self.batch_size = self.population // 2   # use antithetic sampling
        
    def sigma_shift(self):
        sigma, size = 0, 0
        for name in self.sigma:
            sigma += self.sigma[name].sum()
            size  += torch.numel(self.sigma[name])
        rms = sigma / size - self.sigma_init
        return rms

    def mutate(self):
        noise_states = []
        state_dict = self.simulator.sgn.state_dict()
        for _ in range(self.batch_size):
            noise_state = dict()
            for w in state_dict:
                noise_state[w] = state_dict[w] + torch.normal(mean=0, std=self.sigma[w])
            noise_states.append(noise_state)
        
        # antithetic noise
        for i in range(self.batch_size):
            noise_state = dict()
            for w in state_dict:
                noise_state[w] = -noise_states[i][w] + 2 * state_dict[w]
            noise_states.append(noise_state) 

        return noise_states

    def evolve(self, agent_states, sgn_states, rewards):
        # agent_states: WORKERS x ELITES
        # sgn_states:   WORKERS
        # rewards:      WORKERS x [ELITES]
        def compute_ranks(x):
            assert x.ndim == 1
            ranks = np.empty(x.size, dtype=int)
            ranks[x.argsort()] = np.arange(x.size)
            return ranks

        def compute_centered_ranks(x):
            """
            https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
            """
            y = compute_ranks(x).astype(np.float32)
            y /= (x.size - 1)
            y -= .5
            return y
        
        # Agent Selection:
        rewards = np.array(rewards, dtype=np.float32)   # ELITES x WORKERS
        bestidx = np.argsort(rewards)[::-1][:ELITES]    # best index
        new_agent_states = [agent_states[idx] for idx in bestidx]
        infostr = '{} agents: Average Rewards: {:.3f}, Min: {:.3f}, Max: {}'.format(rewards.size, np.mean(rewards), 
            np.min(rewards), ' '.join(['{:.3f}({})'.format(a, bestidx[it]) for it, a in enumerate(rewards[bestidx])]))
        infostr += ' rms: {:.6f}'.format(self.sigma_shift())
        logger.info(infostr)

        # Reward reshaping
        rewards = rewards.reshape(-1, ELITES)[:-1].mean(-1)  # (65-1) * 4
        rewards = compute_centered_ranks(rewards)
        b = np.mean(rewards)
        # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalized reward
        # weights = normalized_reward / (WORKERS-1) / (self.sigma * self.sigma)

        # SGN updates (PEPG)
        rT = (rewards[:self.batch_size] - rewards[self.batch_size:]) / 2.0 
        rS = (rewards[:self.batch_size] + rewards[self.batch_size:]) / 2.0 
        self.simulator.optim_sgn.zero_grad()
        for name, p in self.simulator.sgn.named_parameters():
            delta_sigma = torch.zeros(*self.sigma[name].size())
            for i in range(self.batch_size):
                epsilon = sgn_states[i][name] - p.data
                p.grad.data -= rT[i] * epsilon / self.batch_size  # update mu
                delta_sigma += (rS[i] - b) * ((epsilon * epsilon - self.sigma[name] * self.sigma[name]) / self.sigma[name]) / self.batch_size

            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = torch.max(-self.sigma_max_change * self.sigma[name], change_sigma)
            change_sigma = torch.min( self.sigma_max_change * self.sigma[name], change_sigma)
    
            self.sigma[name] += change_sigma
        self.simulator.optim_sgn.step()

        return new_agent_states

# --------------------------------------------------------------------------------- #
simulator = Simulator(
            Agent(games['bipedhard']), 
            SGN(SGNET(games['bipedhard'])), gpu=-1)
evo = EvoGrad(simulator, sigma_init=0.1)

def slave():
    # make enviromnent
    simulator.agent.make_env(SEED)
    while True:
        agent_states, sgn_state, batches, seeds = comm.recv(source=0)
        rewards, steps, states = [], [], []

        if (len(batches) > 0) and (sgn_state is not None):   # load the sgn state
            simulator.sgn.load_state_dict(sgn_state)

        for it, agent_state in enumerate(agent_states):
            if agent_state is not None:  # load agent state
                simulator.agent.load_state_dict(agent_state[0])
                simulator.optimizer.load_state_dict(agent_state[1])

            # local updates of agents with synthetic gradients.
            if len(batches) > 0:
                for batch in batches:
                    simulator.agent_update(batch)
                agent_states[it] = (simulator.agent.state_dict(), simulator.optimizer.state_dict())

            # collect the simulation results
            reward, step, state = simulator.simulate(TRIAL, seed=seeds)
            rewards.append(reward)
            steps.append(step)
            states += state
        
        data = (rewards, steps, states, agent_states)
        comm.send(data, dest=0)


def master():
    print('I am the master.')
    seeder = Seeder(SEED)
    memory = StateBuffer(MAX_BUFFER)

    msg_agents = []
    for _ in range(ELITES):   # randomly initialize some agents.
        msg_agents.append((simulator.agent.state_dict(), simulator.optimizer.state_dict()))
        simulator.agent.reset()

    msg_sgns   = [None for _ in range(WORKERS)]  # synthetic gradients
    msg_batch  = []
    msg_seeds  = seeder.next_batch(TRIAL)
    
    for epoch in range(200):
        t_start = time.time()
        for i in range(WORKERS):
            msg_batch = [] if i == (WORKERS - 1) else msg_batch   # leave the last batch not training.
            msg_message = (msg_agents, msg_sgns[i], msg_batch, msg_seeds)
            comm.send(msg_message, dest=i+1)
        
        msg_agents, msg_reward, msg_steps = [], [], []
        for i in range(WORKERS):
            rewards, steps, observations, agent_states = comm.recv(source=i+1)
            if (epoch == 0) and (i > 0):
                continue    
            
            msg_agents += agent_states
            msg_reward += rewards   
            msg_steps  += steps  
            memory.add(observations)

        if epoch > 0:
            msg_agents = evo.evolve(msg_agents, msg_sgns, msg_reward)
        else:
            logger.info('initial reward: {} '.format(msg_reward))

        msg_batch = [memory.sample(BATCH_SIZE) for _ in range(INNER_STEP)]
        msg_seeds = seeder.next_batch(TRIAL)
        msg_sgns  = evo.mutate() + [None]
        logger.info('Epoch {}, time cost: {:.3f} s／ average / {:.3f} steps'.format(epoch, time.time() - t_start, np.mean(msg_steps)))


def main():
    print("process", rank, "out of total ", comm.Get_size(), "started")
    if (rank == 0):
        master()
    else:
        slave()

def mpi_fork(n):
    if n<=1:
        return "child"
    
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        print( ["mpirun --allow-run-as-root", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = comm.Get_size()
        rank = comm.Get_rank()
        print('assigning the rank and nworkers', nworkers, rank)
        return "child"

if __name__ == "__main__":
    if "parent" == mpi_fork(WORKERS+1): sys.exit()  
    main()
