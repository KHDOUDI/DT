# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import Dependencies for GYM env
import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from gym.spaces import Dict as dict_
from gym import spaces

# Import libs for Regression Model
import warnings
import pickle as pickle
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import matplotlib as plt
from numpy import mean, std   
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os

# Import libs for DRL
from stable_baselines3 import PPO, A3C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Types of Spaces
'''
Discrete(3).sample()
Box(0,1,shape=(3,)).sample()
Tuple((Discrete(3),Box(0,1,shape=(3,)))).sample()

Dict({
      'GLASS_COLOR':Discrete(2),
      'GLASS_THICKNESS':Discrete(3),
      'GLASS_EXIT_TEMP':Box(500,700,shape=(1,),dtype=int),
      'TOP_AIR_PRESSURE':Box(15,50,shape=(1,),dtype=int),
      'BOTTOM_AIR_PRESSURE':Box(15,50,shape=(1,),dtype=int),
      'TOP_AIR_TEMP':Box(20,90,shape=(1,),dtype=int),      
      'BOTTOM_AIR_TEMP':Box(20,90,shape=(1,),dtype=int),         
      'QUENCH_TIME':Box(150,250,shape=(1,),dtype=int)
      }).sample()

'''


'''
def train(): 

    dataset=pd.read_csv('U00.csv',delimiter=';', decimal=',')    
    X=dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].values
    Y=dataset.iloc[:,[12]].values
    
    X_train, X_val, y_train, Y_val = train_test_split(X,Y, test_size=0.1, random_state=0)
    
    cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
    regressor=RandomForestRegressor(n_estimators=120,random_state=0)
    scores=cross_val_score(regressor,X_train,y_train,scoring='r2',cv=cv,n_jobs=-1)
    
    print('MAE: %.3f (%.3f)' %(mean(scores),std(scores)))
    
    regressor.fit(X_train,y_train.ravel())
    Y_pred_val=regressor.predict(X_val)
    Error = mean_absolute_error(Y_val,Y_pred_val)
    
    print("Model Error:", Error)
    
    pickle.dump(regressor, open("predictor.pkl", 'wb'))

train()
'''
def predict(input):
    loaded_model = pickle.load(open("predictor.pkl", 'rb'))
    result=loaded_model.predict(input)
    return round(float(result),2)

predict(np.array([126.04,11.04,4.96,115.96,1412.05,1440.73,177.94,162.13,225.84,1431.74,13.69,29.03]).reshape(1, -1))
#input = [[126.04,11.04,4.96,115.96,1412.05,1440.73,177.94,162.13,225.84,1431.74,13.69,29.03]]
'''
MOLD_TEMP = [110,140]
FILL_TIME = [5,25]
PLAST_TIME = [0,20]
CYCLE_TIME = [100,140]
CLOSING_FORCE = [500,2500]
CLAMP_FORCE_PEAK = [500,2500]
TORQUE_PEAK_VAL = [100,300]
TORQUE_MEAN_VAL = [50,250]
BACK_PRESS_PEAK_VAL = [100,400]
INJECT_PRESS_PEAK_VAL = [500,2500]
SCREW_HOLD_POS = [5,25]
SHOT_VOLUME = [10,40]
U0_AGENT = [0,85]
U0_VALUE_TARGET = [0,0.5]
'''

# Building an Environment
class InjectionAgent(Env):
    def __init__(self):
        
        #self.target_frag=95
        #global glass_temp, press_bot, press_top, speed, frag_agent, frag_target, ep_length
        self.MOLD_TEMP = [110,140]
        self.FILL_TIME = [5,25]
        self.PLAST_TIME = [0,20]
        self.CYCLE_TIME = [100,140]
        self.CLOSING_FORCE = [500,2500]
        self.CLAMP_FORCE_PEAK = [500,2500]
        self.TORQUE_PEAK_VAL = [100,300]
        self.TORQUE_MEAN_VAL = [50,250]
        self.BACK_PRESS_PEAK_VAL = [100,400]
        self.INJECT_PRESS_PEAK_VAL = [500,2500]
        self.SCREW_HOLD_POS = [5,25]
        self.SHOT_VOLUME = [10,40]
        self.U0_AGENT = [0,85]
        self.U0_VALUE_TARGET = [0.25,0.5]
        self.ep_length=1000

        self.action_space=MultiDiscrete(12*[3])
        
        self.observation_space= spaces.Box(low = np.array([self.MOLD_TEMP[0], self.FILL_TIME[0], self.PLAST_TIME[0], self.CYCLE_TIME[0], self.CLOSING_FORCE[0], 
                                                           self.CLAMP_FORCE_PEAK[0],self.TORQUE_PEAK_VAL[0],self.TORQUE_MEAN_VAL[0],self.BACK_PRESS_PEAK_VAL[0],
                                                           self.INJECT_PRESS_PEAK_VAL[0],self.SCREW_HOLD_POS[0],self.SHOT_VOLUME[0],self.U0_AGENT[0],self.U0_VALUE_TARGET[0]]), 
                                           high = np.array([self.MOLD_TEMP[1], self.FILL_TIME[1], self.PLAST_TIME[1], self.CYCLE_TIME[1], self.CLOSING_FORCE[1], 
                                                            self.CLAMP_FORCE_PEAK[1],self.TORQUE_PEAK_VAL[1],self.TORQUE_MEAN_VAL[1],self.BACK_PRESS_PEAK_VAL[1],
                                                            self.INJECT_PRESS_PEAK_VAL[1],self.SCREW_HOLD_POS[1],self.SHOT_VOLUME[1],self.U0_AGENT[1],self.U0_VALUE_TARGET[1]]), 
                                           dtype = int)
        
        self.state = np.array([self.MOLD_TEMP[0], 
                               self.FILL_TIME[0], 
                               self.PLAST_TIME[0], 
                               self.CYCLE_TIME[0], 
                               self.CLOSING_FORCE[0], 
                               self.CLAMP_FORCE_PEAK[0],
                               self.TORQUE_PEAK_VAL[0],
                               self.TORQUE_MEAN_VAL[0],
                               self.BACK_PRESS_PEAK_VAL[0],                               
                               self.INJECT_PRESS_PEAK_VAL[0],
                               self.SCREW_HOLD_POS[0],
                               self.SHOT_VOLUME[0],
                               self.U0_AGENT[0],
                               round(random.uniform(self.U0_VALUE_TARGET[0],self.U0_VALUE_TARGET[1]),2)])
        self.episode_length=self.ep_length
        
        
    def step(self,action):
        
        '''
        0 => 0 => NADA
        1 => 1 => UP
        2 => -1 => DOWN
        '''
        action = np.where(action==2, -2, action)
        action = np.where(action==1, 2, action)
        
        if self.state[0] <= self.MOLD_TEMP[0]:
            self.state[0]=self.MOLD_TEMP[0]
        if self.state[0] >= self.MOLD_TEMP[1]:
            self.state[0]=self.MOLD_TEMP[1]
            
        if self.state[1] <= self.FILL_TIME[0]:
            self.state[1]=self.FILL_TIME[0]
        if self.state[1] >= self.FILL_TIME[1]:
            self.state[1]=self.FILL_TIME[1]
            
        if self.state[2] <= self.PLAST_TIME[0]:
            self.state[2]=self.PLAST_TIME[0]
        if self.state[2] >= self.PLAST_TIME[1]:
            self.state[2]=self.PLAST_TIME[1]
            
        if self.state[3] <= self.CYCLE_TIME[0]:
            self.state[3]=self.CYCLE_TIME[0]
        if self.state[3] >= self.CYCLE_TIME[1]:
            self.state[3]=self.CYCLE_TIME[1]

        if self.state[4] <= self.CLOSING_FORCE[0]:
            self.state[4]= self.CLOSING_FORCE[0]
        if self.state[4] >= self.CLOSING_FORCE[1]:
            self.state[4]= self.CLOSING_FORCE[1]
            
        if self.state[5] <= self.CLAMP_FORCE_PEAK[0]:
            self.state[5]= self.CLAMP_FORCE_PEAK[0]
        if self.state[5] >= self.CLAMP_FORCE_PEAK[1]:
            self.state[5]= self.CLAMP_FORCE_PEAK[1]
            
        if self.state[6] <= self.TORQUE_PEAK_VAL[0]:
            self.state[6]= self.TORQUE_PEAK_VAL[0]
        if self.state[6] >= self.TORQUE_PEAK_VAL[1]:
            self.state[6]= self.TORQUE_PEAK_VAL[1]

        if self.state[7] <= self.TORQUE_MEAN_VAL[0]:
            self.state[7]= self.TORQUE_MEAN_VAL[0]
        if self.state[7] >= self.TORQUE_MEAN_VAL[1]:
            self.state[7]= self.TORQUE_MEAN_VAL[1]

        if self.state[8] <= self.BACK_PRESS_PEAK_VAL[0]:
            self.state[8]= self.BACK_PRESS_PEAK_VAL[0]
        if self.state[8] >= self.BACK_PRESS_PEAK_VAL[1]:
            self.state[8]= self.BACK_PRESS_PEAK_VAL[1]

        if self.state[9] <= self.INJECT_PRESS_PEAK_VAL[0]:
            self.state[9]= self.INJECT_PRESS_PEAK_VAL[0]
        if self.state[9] >= self.INJECT_PRESS_PEAK_VAL[1]:
            self.state[9]= self.INJECT_PRESS_PEAK_VAL[1]

        if self.state[10] <= self.SCREW_HOLD_POS[0]:
            self.state[10]= self.SCREW_HOLD_POS[0]
        if self.state[10] >= self.SCREW_HOLD_POS[1]:
            self.state[3]= self.SCREW_HOLD_POS[1]

        if self.state[11] <= self.SHOT_VOLUME[0]:
            self.state[11]= self.SHOT_VOLUME[0]
        if self.state[11] >= self.SHOT_VOLUME[1]:
            self.state[11]= self.SHOT_VOLUME[1] 
            
        mold_temp = self.state[0]+action[0]
        fill_time = self.state[1]+action[1]
        plast_time = self.state[2]+action[2]
        cycle_time = self.state[3]+action[3]
        closing_force = self.state[4]+action[4]
        clamping_force_peak = self.state[5]+action[5]
        torque_peak_val = self.state[6]+action[6]
        torque_mean_val = self.state[7]+action[7]
        back_press_peak_val = self.state[8]+action[8]
        inject_press_peak_val = self.state[9]+action[9]
        screw_hold_pos = self.state[10]+action[10]
        shot_volume = self.state[11]+action[11]        
        u0_target = self.state[13]

        
        self.state = np.array([mold_temp,
                               fill_time,
                               plast_time,
                               cycle_time,
                               closing_force,
                               clamping_force_peak,
                               torque_peak_val,
                               torque_mean_val,
                               back_press_peak_val,
                               inject_press_peak_val,
                               screw_hold_pos,
                               shot_volume,
                               predict(np.array([[mold_temp,
                                                    fill_time,
                                                    plast_time,
                                                    cycle_time,
                                                    closing_force,
                                                    clamping_force_peak,
                                                    torque_peak_val,
                                                    torque_mean_val,
                                                    back_press_peak_val,
                                                    inject_press_peak_val,
                                                    screw_hold_pos,
                                                    shot_volume]])),
                               u0_target])
        
        print(self.state[13],'   ',self.state[12])
        '''
        if abs(self.target_frag-self.state[4])<=3:
            reward=1/(1+(abs(self.target_frag-self.state[4])))
        else:
            reward=-1
        '''    
        if abs(self.state[13]-self.state[12]) <=0.03:
            reward=1/(0.5+abs(self.state[13]-self.state[12]))
        else:
            reward=-1/(0.5+abs(self.state[13]-self.state[12]))
        
        '''
        # Calculate reword
        if self.state[4] == self.target_frag:
            reward = 1
            #done = True
            #print(self.state)
        else:
            reward = -1
            #done = False
        '''    
        
        self.episode_length -=1
        
        if abs(self.state[13]-self.state[12]) <=0.03 or self.episode_length<=0:
            done= True
        else:
            done= False
        
        info = {} 
        
        return self.state, reward, done, info
        
    def render(self):
        pass
    def reset(self):

        reset_MOLD_TEMP = random.randint(self.MOLD_TEMP[0], self.MOLD_TEMP[1])
        reset_FILL_TIME = random.randint(self.FILL_TIME[0], self.FILL_TIME[1])
        reset_PLAST_TIME = random.randint(self.PLAST_TIME[0], self.PLAST_TIME[1])
        reset_CYCLE_TIME = random.randint(self.CYCLE_TIME[0], self.CYCLE_TIME[1])
        reset_CLOSING_FORCE = random.randint(self.CLOSING_FORCE[0], self.CLOSING_FORCE[1])
        reset_CLAMP_FORCE_PEAK = random.randint(self.CLAMP_FORCE_PEAK[0], self.CLAMP_FORCE_PEAK[1])
        reset_TORQUE_PEAK_VAL = random.randint(self.TORQUE_PEAK_VAL[0], self.TORQUE_PEAK_VAL[1])
        reset_TORQUE_MEAN_VAL = random.randint(self.TORQUE_MEAN_VAL[0], self.TORQUE_MEAN_VAL[1])
        reset_BACK_PRESS_PEAK_VAL = random.randint(self.BACK_PRESS_PEAK_VAL[0], self.BACK_PRESS_PEAK_VAL[1])
        reset_INJECT_PRESS_PEAK_VAL = random.randint(self.INJECT_PRESS_PEAK_VAL[0], self.INJECT_PRESS_PEAK_VAL[1])
        reset_SCREW_HOLD_POS = random.randint(self.SCREW_HOLD_POS[0], self.SCREW_HOLD_POS[1])
        reset_SHOT_VOLUME = random.randint(self.SHOT_VOLUME[0], self.SHOT_VOLUME[1])
        reset_U0_VALUE_TARGET = round(random.uniform(self.U0_VALUE_TARGET[0], self.U0_VALUE_TARGET[1]),2)

        self.state = np.array([self.MOLD_TEMP[0], 
                               self.FILL_TIME[0], 
                               self.PLAST_TIME[0], 
                               self.CYCLE_TIME[0], 
                               self.CLOSING_FORCE[0], 
                               self.CLAMP_FORCE_PEAK[0],
                               self.TORQUE_PEAK_VAL[0],
                               self.TORQUE_MEAN_VAL[0],
                               self.BACK_PRESS_PEAK_VAL[0],                               
                               self.INJECT_PRESS_PEAK_VAL[0],
                               self.SCREW_HOLD_POS[0],
                               self.SHOT_VOLUME[0],
                               predict(np.array([[self.MOLD_TEMP[0], 
                                                      self.FILL_TIME[0], 
                                                      self.PLAST_TIME[0], 
                                                      self.CYCLE_TIME[0], 
                                                      self.CLOSING_FORCE[0], 
                                                      self.CLAMP_FORCE_PEAK[0],
                                                      self.TORQUE_PEAK_VAL[0],
                                                      self.TORQUE_MEAN_VAL[0],
                                                      self.BACK_PRESS_PEAK_VAL[0],                               
                                                      self.INJECT_PRESS_PEAK_VAL[0],
                                                      self.SCREW_HOLD_POS[0],
                                                      self.SHOT_VOLUME[0]]])),
                               reset_U0_VALUE_TARGET])
        
        '''       
        self.state = np.array([reset_MOLD_TEMP, 
                               reset_FILL_TIME, 
                               reset_PLAST_TIME, 
                               reset_CYCLE_TIME, 
                               reset_CLOSING_FORCE, 
                               reset_CLAMP_FORCE_PEAK,
                               reset_TORQUE_PEAK_VAL,
                               reset_TORQUE_MEAN_VAL,
                               reset_BACK_PRESS_PEAK_VAL,                               
                               reset_INJECT_PRESS_PEAK_VAL,
                               reset_SCREW_HOLD_POS,
                               reset_SHOT_VOLUME,
                               predict(np.array([[reset_MOLD_TEMP, 
                                                      reset_FILL_TIME, 
                                                      reset_PLAST_TIME, 
                                                      reset_CYCLE_TIME, 
                                                      reset_CLOSING_FORCE, 
                                                      reset_CLAMP_FORCE_PEAK,
                                                      reset_TORQUE_PEAK_VAL,
                                                      reset_TORQUE_MEAN_VAL,
                                                      reset_BACK_PRESS_PEAK_VAL,                               
                                                      reset_INJECT_PRESS_PEAK_VAL,
                                                      reset_SCREW_HOLD_POS,
                                                      reset_SHOT_VOLUME]])),
                               reset_U0_VALUE_TARGET])
        '''        
        self.episode_length=self.ep_length
        return self.state

env=InjectionAgent()

print(env.observation_space.sample())
print(env.action_space.sample())

# Test Environment
'''
episode = 5
for episode in range(1, episode+1):
    warnings.filterwarnings("ignore")
    obs = env.reset()
    done =False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{} Obs:{}'.format(episode,score,obs))
env.close()
'''
# Train Model

log_path = os.path.join('Training','Logs')
model = A3C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=190500)

# Save Model

injection_path = os.path.join('Training','Saved Models','Injection_Model_A3C_')
model.save(injection_path)

#del model

# Load Model
model = A3C.load(injection_path, env)

# Evaluate the Model
evaluate_policy(model, env, n_eval_episodes=5, render=False)

# Test the Model

episode = 1
for episode in range(1, episode+1):
    warnings.filterwarnings("ignore")
    obs = env.reset()
    obs[13] = 0.25
    
    done =False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{} Obs:{}'.format(episode,score,obs))
