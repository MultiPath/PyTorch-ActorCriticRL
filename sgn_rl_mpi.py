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
import utils
import time
import random
import numpy as np
import buffer
import gc
import os
import subprocess

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
POPULATION = 51
WORKERS = 64
SAMPLES = 16
BATCH_SIZE = 128
INNER_STEP = 5
SEED = 19920206
SIGMA = 0.1
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
    
        # action noise
        self.noise = utils.OrnsteinUhlenbeckActionNoise(game.output_size)
        self.env = None

        # weight initialization --- try Gaussian initialization ?? ---
        # for l in self.net:
        #     init.xavier_normal(l.weight, gain=init.calculate_gain('relu'))
        #     init.constant(l.bias, 0.0)
        # init.xavier_normal(self.output.weight, gain=init.calculate_gain('tanh'))
        # init.constant(self.output.bias, 0.0) 


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
        self.optim_sgn = torch.optim.Adam(self.sgn.parameters(), 0.001) 

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

    def simulate(self, episodes=1, seed=None, max_step=-1, train=True, noisy=False):
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

        return rewards, steps, states

simulator = Simulator(
            Agent(games['bipedhard']), 
            SGN(SGNET(games['bipedhard'])),
            gpu=-1)



# ------------------- Evo --------------------------------------- #
def mutate(state_dict, population, sigma=0.1):
    noise_states = []
    for _ in range(population):
        noise_state = dict()
        for w in state_dict:
            noise_state[w] = torch.normal(state_dict[w], sigma)
        noise_states.append(noise_state)
    return noise_states


def evolve(agent_states, sgn_states, reward, sigma=0.1, k=1):
    
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
    
    assert len(agent_states) % k == 0, 'k must be divided by the agent number'

    # reward shaping
    logger.info('Rewards: {}, Max: {}, Min: {}'.format(np.mean(reward), np.max(reward), np.min(reward)))
    reward  = np.array(reward, dtype=np.float32)
    _reward = compute_centered_ranks(reward)
    idx = np.argsort(_reward)[::-1]
    best_idx = idx[:k]   # top K agents will be saved.
    logger.info('Top K: {}'.format(reward[best_idx]))

    # Agent selection
    new_agents = []
    for i in range(len(agent_states) // k):
        for idx in best_idx:
            new_agents.append(agent_states[idx])

    # ES updates for SGN
    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
    weights = normalized_reward / len(sgn_states) / (sigma * sigma)

    simulator.optim_sgn.zero_grad()
    for name, p in simulator.sgn.named_parameters():
        for i in range(len(sgn_states)):
            p.grad.data += weights[i] * (p.data - sgn_states[i][name])
    simulator.optim_sgn.step()

    return new_agents


def slave():
    simulator.agent.make_env(19920206)
    while True:
        _agent, _sgn, _batch, _seeds = comm.recv(source=0)

        if _agent is not None:
            simulator.agent.load_state_dict(_agent[0])
            simulator.optimizer.load_state_dict(_agent[1])
        
        if _sgn is not None:
            simulator.sgn.load_state_dict(_sgn)

        if len(_batch) > 0:
            for ob in _batch:
                simulator.agent_update(ob)

        rewards, steps, states = simulator.simulate(SAMPLES, seed=_seeds)
        agent_states = [simulator.agent.state_dict(), simulator.optimizer.state_dict()]
        data = (rewards, steps, states, agent_states)
        
        comm.send(data, dest=0)


def master():
    print('I am the master.')

    seeder = Seeder(SEED)
    memory = buffer.StateBuffer(MAX_BUFFER)

    msg_agents = [None for _ in range(WORKERS)]
    msg_sgns   = [None for _ in range(WORKERS)]
    msg_batch  = []
    msg_seeds  = seeder.next_batch(SAMPLES)
    
    for epoch in range(10):
        t_start = time.time()
        for i in range(WORKERS):
            msg_message = (msg_agents[i], msg_sgns[i], msg_batch, msg_seeds)
            comm.send(msg_message, dest=i+1)

        msg_agents, msg_reward = [], []
        for i in range(WORKERS):
            rewards, steps, states, agent_states = comm.recv(source=i+1)
            msg_agents.append(agent_states)
            
            if (epoch == 0) and (i > 0):
                continue      

            memory.add(states)
            msg_reward.append(np.mean(rewards))

    
        if epoch > 0:
            t = time.time()
            msg_agents = evolve(msg_agents, msg_sgns, msg_reward, sigma=0.1, k=4)
            
        msg_batch = [memory.sample(BATCH_SIZE) for _ in range(INNER_STEP)]
        msg_seeds = seeder.next_batch(SAMPLES)
        msg_sgns  = mutate(simulator.sgn.state_dict(), WORKERS, sigma=0.1)

        logger.info('time cost: {} s'.format(time.time() - t_start))


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
        print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
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
