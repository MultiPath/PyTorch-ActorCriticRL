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

from torch.autograd import Variable
from gym import wrappers
from collections import namedtuple

# prepare the environment
gym.undo_logger_setup()
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -- hyper parameters
# setup random seeds

MAX_EPISODES = 5000
MAX_STEPS = 3000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
POPULATION = 51
SAMPLES = 16
BATCH_SIZE = 128
SEED = 19920206
SIGMA = 0.1
gpu = -1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if gpu > -1:
    torch.cuda.manual_seed_all(SEED)
SEEDS = (torch.rand(10000000) * 10000000).long()

Game = namedtuple('Game', 
    ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise'])
games = {}

bipedhard = Game(env_name='BipedalWalker-v2',  # BipedalWalkerHardcore-v2
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

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, a=None, s=None, grad_y=None, mode='forward'):
        if mode == 'forward':
            y = a.mean(-1)
            self.saved_variables = [Variable(a), Variable(s)]
            return y

        else:
            a, s = self.saved_variables
            syn_grad_a = self.net(a, s)
            syn_grad_a = SGN.squash(syn_grad_a)
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
# random agent
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


class Simulator(object):
    
    def __init__(self, agent, sgn, gpu=-1):
        self.gpu = gpu
        self.moving_agent = agent
        
        # population based training
        self.agents = [copy.deepcopy(agent) for _ in range(POPULATION)]
        self.sgns = [copy.deepcopy(sgn) for _ in range(POPULATION)]
        self.optimizers = [torch.optim.Adam(self.agents[i].parameters(), 0.001) for i in range(POPULATION)]
        self.buffer = buffer.StateBuffer(MAX_BUFFER)

        if gpu > -1:
            self.moving_agent = self.moving_agent.cuda(gpu)
            for a in self.agents:
                a = a.cuda(gpu)
            for s in self.sgns:
                s = s.cuda(gpu)

        self.current_sgn = None

    def add_noise(self):
        # 0    -- original model
        # 1~11  - positive noise
        # 12~21 - negative noise  
        s = (POPULATION - 1) // 2
        for i in range(s):
            sgn1 = self.sgns[i * 2 + 1]
            sgn2 = self.sgns[i * 2 + 2]
            for p1, p2 in zip(sgn1.parameters(), sgn2.parameters()):
                p1.data = torch.normal(p1.data, SIGMA)
                p2.data = p2.data * 2 - p1.data


    def agent_train(self, observations, i, batch_size=1):
        observations = torch.from_numpy(observations)
        if self.gpu > -1:
            observations = observations.cuda(self.gpu)
        observations =  Variable(observations)
        observations = observations.view(observations.size(0) // batch_size, batch_size, observations.size(1))
        for step in range(observations.size(0)):
            predict_actions, fake_loss = self.agents[i](observations[step], self.sgns[i])
            self.optimizers[i].zero_grad()
            fake_loss.backward()
            self.optimizers[i].step()
        return None

    def simulate(self, i=-1, queue=None, episodes=1, seed=None, max_step=-1, train=True, noisy=False):
        steps = []
        rewards = []
        states = []

        run_time, py_time, q_time, e_time = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()

        if i > -1:
            agent = self.agents[i]
        else:
            agent = self.moving_agent

        agent.make_env(seed)
        for ep in range(episodes):
            ob = agent.env.reset()
            total_reward = 0
            stumbled = False

            if max_step == -1:
                max_step = MAX_STEPS

            for step in range(max_step):
                _t = time.time()
                action, _ = agent.act(ob, gpu=self.gpu)
                py_time += (time.time() - _t)

                _t = time.time() 
                new_ob, reward, done, info = agent.env.step(action)
                e_time += (time.time() - _t)

                if train and  (reward == -100):
                    stumbled = True
                    reward = 0
                total_reward += reward

                _t = time.time()
                if done:
                    if train and (not stumbled) and (total_reward > MAX_TOTAL_REWARD):
                        total_reward += 100    
                    break   
                # else:
                #     if queue is not None:
                #         queue.put((ob, action, reward, new_ob)) 

                states.append(ob)
                ob = new_ob   # important
                q_time += (time.time() - _t)

            rewards.append(total_reward)
            steps.append(step)
        
        # if queue is not None:
        #     queue.put(None)

        print('simulate time: {}'.format(time.time() - start_time))
        print('simulate_py time: {}'.format(py_time))
        print('simulate_qq time: {}'.format(q_time))
        print('simulate_ee time: {}'.format(e_time))
        return i, np.mean(rewards), np.mean(steps), states

def collect_rewards(res, population=1):
    mean_rewards = torch.Tensor(population).zero_()
    mean_steps = torch.Tensor(population).zero_()
    states = []
    for i, r, s, obs in res:
        mean_rewards[i] += r
        mean_steps[i] += s
        states += obs
    return mean_rewards, mean_steps, states


if __name__ == '__main__': 
    simulator = Simulator(
        Agent(games['bipedhard']), 
        SGN(SGNET(games['bipedhard'])),
        gpu=gpu)


    # ---- multi-processing ------ #
    mp.set_start_method('spawn')
    # simulate from the moving agent for N epsiodes
    
    queue = mp.Manager().Queue()

    for ep in range(2):
        start_t = time.time()
        if ep == 0:
            population = 1
        else:
            population = POPULATION
    
        jobs = []
        pool = mp.Pool(processes=40)
        for i in range(population):
            jobs.append(pool.apply_async(simulator.simulate, (i, queue, SAMPLES, SEEDS[0])))
        pool.close()

        # counter = population
        # while counter > 0:
        #     data = queue.get()
        #     if data is None:
        #         counter -= 1
        #     else:
        #         simulator.buffer.add(*data)

        pool.join()

        rewards, steps, states = collect_rewards([job.get() for job in jobs], population)
        print(len(states))
        print(rewards)

        simulator.buffer.add(states)
        # for i in range(population * SAMPLES):
            # jobs.append(pool.apply_async(simulator.simulate, (i // SAMPLES, queue, 1, SEEDS[i % SAMPLES])))
        #    simulator.simulate(i // SAMPLES, queue, 1, SEEDS[i % SAMPLES])
        # pool.close()
        # pool.join()
        # rewards, steps= collect_rewards([job.get() for job in jobs], population)
        # print(rewards)

        print(time.time() - start_t)
        # add the simulation to the experience replay buffer
        
        
        print(simulator.buffer.len)
        
        # training the agents with SGNs
        obs= simulator.buffer.sample(BATCH_SIZE * 10)
        
        print(obs.shape)
        simulator.add_noise()
        print(time.time() - start_t)
        print('start train.')
        for i in range(POPULATION):
            simulator.agent_train(obs, i, BATCH_SIZE)
        # pool = mp.Pool(processes=POPULATION) 
        # res = []
        # for i in range(POPULATION):
        #     res.append(pool.apply_async(simulator.agent_train, (obs, i, BATCH_SIZE, )))
        
        # pool.close()
        # pool.join()
        # for r in res:
        #    r.get()

        print(time.time() - start_t)
    print('can run.')

    # print(rewards)
    # print(steps)


#print(simulate(agent, ram, 100, train=True))