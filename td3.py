import copy
import gym
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
from env.env import ShakeTableEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(device),
			torch.FloatTensor(self.action[ind]).to(device),
			torch.FloatTensor(self.next_state[ind]).to(device),
			torch.FloatTensor(self.reward[ind]).to(device),
			torch.FloatTensor(self.not_done[ind]).to(device)
		)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)        
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, action_dim)

        self.max_action = max_action
        self.to(device)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 512)
        self.l6 = nn.Linear(512, 256)
        self.l7 = nn.Linear(256, 128)
        self.l8 = nn.Linear(128, 1)

        self.to(device)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
        learning_rate=1e-3,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_dir", type=Path, default="./env")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--do_render", action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':    
    args = parse()
    env = ShakeTableEnv(args.env_dir)
    state_dim = 64
    action_dim = 1
    max_action = 80
    batch_size = 128
    expl_noise = 0.2
    lr = 1e-3
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    agent = TD3(state_dim,
                action_dim,
                max_action,
                discount=0.99,
                tau=0.005,
                learning_rate=lr,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=8)
    np.random.seed(0)
    
    if args.test:
        agent.load('./ckpt/td3')
    
    best_score = -np.inf
    score_history = []
    eps = 1000
    for i in range(eps):
        done = False
        score = 0        
        state = env.reset()  
        steps = 0
        while not done:
            steps += 1
            if args.do_render:
                env.render(mode='all')
                
            if args.test:
                action = agent.select_action(np.array(state))
                state, reward, done, info = env.step(action)
            else:
                action = (
                    agent.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action).astype(np.float32)
                
                next_state, reward, done, info = env.step(action)
                done_bool = float(done)
                replay_buffer.add(state, action, next_state, reward, done_bool)
                state = next_state
                agent.train(replay_buffer, batch_size)
            print(steps, reward)
                
            score += reward
        env.plot_history(name = str(i+1).zfill(4))
        score_history.append(score)
        print('episode {}/{}\tscore {:.2f}\taverage score {:.2f}, rms_T {:.2f}'.format(i+1, eps, score, np.mean(score_history[-30:]), info['rms_T']), flush=True)
                
        if best_score < np.mean(score_history[-50:]):
            agent.save('./ckpt/td3')
    
    
    
    
    
    
    
    
    
    
    
    
    