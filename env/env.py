import gym
import json
import torch
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
from env.model import LSTMStacked


class ShakeTableEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_dir):
        super(ShakeTableEnv, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(env_dir / "env.config.json") as f:
            self.config = json.load(f)

        with open(env_dir / "desired.pkl", "rb") as f:
            self.references = pickle.load(f)
        
        with open(env_dir / "scaler_X.pkl", "rb") as f:
            self.scaler_X = pickle.load(f)

        with open(env_dir / "scaler_y.pkl", "rb") as f:
            self.scaler_y = pickle.load(f)

        self.obs = []
        
        self.model = LSTMStacked(self.config)

        self.model.load_state_dict(torch.load(env_dir / "env.pt", map_location=device))

        # Maximum available displacement command in drive file
        self.action_space = spaces.Box(
            low=-80.0, high=80.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        self.make_figure = False
        
        self.state_dims = 64
        self.future_dims = 32
        self.preview_dims = 32
        self.threshold = 0.5

    def _take_action(self, action):
        action = torch.tensor(action).view(1, 1, 1)
        self.model.eval()

        with torch.no_grad():
            obs, self.model_hidden_state = self.model(action, self.model_hidden_state)

        obs = self.scaler_y.inverse_transform(obs.view(-1, 1).numpy()).item()

        return obs

    def _update_figure(self):
        new_x_data = self.achieved_line.get_xdata() + self.current_step
        new_y_data = self.achieved_line.get_ydata() + self.obs

        self.achieved_line.set_data(new_x_data, new_y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()

    def step(self, action):
        # Execute one time step within the environment
        self.actions.append(action)
        action = self.scaler_X.transform(action.reshape(-1, 1))
        
        
        obs = self._take_action(action)
        self.history.append([self.current_ref[self.current_step], obs])

        reward, rms_T = self.shaping()
        
        self.current_step += 1

        next_state = np.append(self.future_obs(), self.preview_obs())

        done = self.current_step >= len(self.current_ref) - 1
        
        if done:
            reward, rms_T = self.eval()
            
        info = {'rms_T':rms_T, 'history':self.history, 'total_step':len(self.current_ref)}

        return next_state, reward, done, info
    
    def future_obs(self):
        ref = self.current_ref[self.current_step:self.current_step+self.future_dims]
        ref = ref[::-1]
        if self.future_dims > len(ref):
            ref = np.append(np.zeros((1, self.future_dims-len(ref))), ref)
        return ref
        
    
    def preview_obs(self):
        obs_hist = np.array(self.history)[:, 1]
        obs = obs_hist[::-1][:self.preview_dims]
        if self.preview_dims > len(obs):
            obs = np.append(obs, np.zeros((1, self.preview_dims-len(obs))))
        return obs
    
    def shaping(self):
        hist = np.array(self.history)
        numerator = np.sum(np.square(np.subtract(hist[-self.preview_dims:,0], hist[-self.preview_dims:,1])))
        denominator = np.sum(np.square(hist[-self.preview_dims:,0]))
        rms_T = numerator / (denominator + 1)
        reward = 1 / (1 + rms_T)
        return reward, rms_T
    
    def eval(self):
        hist = np.array(self.history)
        numerator = np.sum(np.square(np.subtract(hist[:,0], hist[:,1])))
        denominator = np.sum(np.square(hist[:,0]))
        rms_T = numerator / denominator        
        if rms_T > self.threshold:
            return -3000 * (rms_T + 1), rms_T
        else:
            return 1000 / (rms_T + 1), rms_T
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_ref = random.choice(self.references)
        self.current_step = 0

        self.model_hidden_state = [torch.zeros(self.config["n_lstm_layer"], 1, self.config["hidden_dim"]) for _ in range(2)]

        initial_state = np.append(self.current_ref[0:self.preview_dims], np.zeros((1, self.preview_dims)))
        
        self.history = []
        self.actions = []

        return initial_state

    # TODO: render
    def render(self, mode='all'):
        # Render the environment to the screen
        if not self.make_figure:
            plt.ion()
            self.fig = plt.figure()
            self.make_figure = True
            
        hist = np.array(self.history)
        
        x1 = np.linspace(0, len(self.current_ref)-1, len(self.current_ref))
        x2 = np.linspace(0, len(hist)-1, len(hist))
        
        if mode == 'all':
            plt.plot(x1, self.current_ref, 'r', label='actual acc')
        if mode == 'part':
            plt.plot(x2, hist[:, 0], 'r', label='actual acc')
        plt.plot(x2, hist[:, 1], 'b', label='predict acc')
        plt.xlabel('time step')
        plt.ylabel('Acceleration')
        plt.legend()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.clf()
    
    def plot_history(self, name='acc'):
        hist = np.array(self.history)
        x = np.linspace(0, len(hist)-1, len(hist))
        plt.plot(x, hist[:, 0], 'r', label='actual acc')
        plt.plot(x, hist[:, 1], 'b', label='predicted acc')
        plt.xlabel('time step')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.savefig(name + '_history.JPG', dpi=200)
        plt.clf()
            
            
            
            
            
            
            