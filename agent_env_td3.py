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
from stable_baselines3 import TD3, DDPG
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
def predict(input_):
    loaded_model = pickle.load(open("predictor.pkl", 'rb'))
    result=loaded_model.predict(input_)
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

MOLD_TEMP = list(range(100,201)),
FILL_TIME = list(range(0,51)),
PLAST_TIME = list(range(0,51)),
CYCLE_TIME = list(range(50,301)),
CLOSING_FORCE = list(range(200,3001)),
CLAMP_FORCE_PEAK = list(range(200,3001)),
TORQUE_PEAK_VAL = list(range(50,501)),
TORQUE_MEAN_VAL = list(range(30,451)),
BACK_PRESS_PEAK_VAL = list(range(50,801)),
INJECT_PRESS_PEAK_VAL = list(range(200,3001)),
SCREW_HOLD_POS = list(range(0,51)),
SHOT_VOLUME = list(range(0,101)),


obs_list = [MOLD_TEMP,
        FILL_TIME,
        PLAST_TIME,
        CYCLE_TIME,
        CLOSING_FORCE,
        CLAMP_FORCE_PEAK,
        TORQUE_PEAK_VAL,
        TORQUE_MEAN_VAL,
        BACK_PRESS_PEAK_VAL,
        INJECT_PRESS_PEAK_VAL,
        SCREW_HOLD_POS,
        SHOT_VOLUME,]

mean_list = list((np.mean(x) for x in obs_list))
std_list = list((np.std(x) for x in obs_list))

def normalized_obs_space():

    norm_obs_list = []

    for v in obs_list:    
        mean_= np.mean(v)
        std_= np.std(v)
        norm = []
        for x in v:
            c = ((x - mean_)/std_)
            norm.append(c)
        norm_obs_list.append(norm)

    return (norm_obs_list[0][0].tolist(),
            norm_obs_list[1][0].tolist(),
            norm_obs_list[2][0].tolist(),
            norm_obs_list[3][0].tolist(),
            norm_obs_list[4][0].tolist(),
            norm_obs_list[5][0].tolist(),
            norm_obs_list[6][0].tolist(),
            norm_obs_list[7][0].tolist(),
            norm_obs_list[8][0].tolist(),
            norm_obs_list[9][0].tolist(),
            norm_obs_list[10][0].tolist(),
            norm_obs_list[11][0])

 
normalized_obs = normalized_obs_space()

def denormalized_obs_space(obs):

    denorm_obs_list = []

    for v in obs: 
        c = (v*std_list[obs.index(v)]) + mean_list[obs.index(v)]
        denorm_obs_list.append(round(c,2))

    return denorm_obs_list

def normalize_state(obs):

    norm_obs_list = []

    for v in obs: 
        c = (v*std_list[obs.index(v)]) + mean_list[obs.index(v)]
        c = ((v - mean_list[obs.index(v)])/std_list[obs.index(v)])
        norm_obs_list.append(round(c,6))

    return norm_obs_list

denormalized_obs_space([-1.71482,
                        -1.69775,
                        -1.69775,
                        -1.72514,
                        -1.73143,
                        -1.73143,
                        -1.72821,
                        -1.72793,
                        -1.72974,
                        -1.73143,
                        -1.69775,
                        -1.71482,])

# Building an Environment
class InjectionAgent(Env):
    def __init__(self):
        
        #self.target_frag=95
        #global glass_temp, press_bot, press_top, speed, frag_agent, frag_target, ep_length
        self.MOLD_TEMP = [100,200]
        self.FILL_TIME = [0,50]
        self.PLAST_TIME = [0,50]
        self.CYCLE_TIME = [50,300]
        self.CLOSING_FORCE = [200,3000]
        self.CLAMP_FORCE_PEAK = [200,3000]
        self.TORQUE_PEAK_VAL = [50,500]
        self.TORQUE_MEAN_VAL = [30,450]
        self.BACK_PRESS_PEAK_VAL = [50,800]
        self.INJECT_PRESS_PEAK_VAL = [200,3000]
        self.SCREW_HOLD_POS = [0,50]
        self.SHOT_VOLUME = [0,100]
        self.U0_AGENT = [0,0.85]
        self.U0_VALUE_TARGET = [0.25,0.5]
        self.ep_length=1000

        self.action_space=Box(-1.0,1.0,(12,), dtype="float32")
        
        self.observation_space= spaces.Box(low = np.array([normalized_obs[0][0], normalized_obs[1][0], normalized_obs[2][0], normalized_obs[3][0], normalized_obs[4][0], 
                                                           normalized_obs[5][0],normalized_obs[6][0],normalized_obs[7][0],normalized_obs[8][0],
                                                           normalized_obs[9][0],normalized_obs[10][0],normalized_obs[11][0],self.U0_AGENT[0],self.U0_VALUE_TARGET[0]]), 
                                           high = np.array([normalized_obs[0][-1], normalized_obs[1][-1], normalized_obs[2][-1], normalized_obs[3][-1], normalized_obs[4][-1], 
                                                            normalized_obs[5][-1],normalized_obs[6][-1],normalized_obs[7][-1],normalized_obs[8][-1],
                                                            normalized_obs[9][-1],normalized_obs[10][-1],normalized_obs[11][-1],self.U0_AGENT[1],self.U0_VALUE_TARGET[1]]), 
                                           dtype = float)

        
        self.state = np.array([normalized_obs[0][0], 
                               normalized_obs[1][0], 
                               normalized_obs[2][0], 
                               normalized_obs[3][0], 
                               normalized_obs[4][0], 
                               normalized_obs[5][0],
                               normalized_obs[6][0],
                               normalized_obs[7][0],
                               normalized_obs[8][0],                               
                               normalized_obs[9][0],
                               normalized_obs[10][0],
                               normalized_obs[11][0],
                               self.U0_AGENT[0],
                               self.U0_VALUE_TARGET[0]])
        
        self.episode_length=self.ep_length
        
        
    def step(self,action):
        
        '''
        0 => 0 => NADA
        1 => 1 => UP
        2 => -1 => DOWN
        '''
        
        print("action brut:\n",action)
        denormalized_state = denormalized_obs_space([
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
            self.state[4],
            self.state[5],
            self.state[6],
            self.state[7],
            self.state[8],
            self.state[9],
            self.state[10],
            self.state[11]
            ])
        #denormalized_state = denormalized_state.extend([self.state[12],self.state[13]])

        #print(denormalized_state)        
        
        mold_temp = denormalized_state[0]+action[0]
        fill_time = denormalized_state[1]+action[1]
        plast_time = denormalized_state[2]+action[2]
        cycle_time = denormalized_state[3]+action[3]
        closing_force = denormalized_state[4]+action[4]
        clamping_force_peak = denormalized_state[5]+action[5]
        torque_peak_val = denormalized_state[6]+action[6]
        torque_mean_val = denormalized_state[7]+action[7]
        back_press_peak_val = denormalized_state[8]+action[8]
        inject_press_peak_val = denormalized_state[9]+action[9]
        screw_hold_pos = denormalized_state[10]+action[10]
        shot_volume = denormalized_state[11]+action[11]   
        u0_agent = self.state[12]
        u0_target = self.state[13]

        
        if mold_temp <= self.MOLD_TEMP[0]:
            mold_temp=self.MOLD_TEMP[0]
        if mold_temp >= self.MOLD_TEMP[1]:
            mold_temp=self.MOLD_TEMP[1]
            
        if fill_time <= self.FILL_TIME[0]:
            fill_time=self.FILL_TIME[0]
        if fill_time >= self.FILL_TIME[1]:
            fill_time=self.FILL_TIME[1]
            
        if plast_time <= self.PLAST_TIME[0]:
            plast_time=self.PLAST_TIME[0]
        if plast_time >= self.PLAST_TIME[1]:
            plast_time=self.PLAST_TIME[1]
            
        if cycle_time <= self.CYCLE_TIME[0]:
            cycle_time=self.CYCLE_TIME[0]
        if cycle_time >= self.CYCLE_TIME[1]:
            cycle_time=self.CYCLE_TIME[1]

        if closing_force <= self.CLOSING_FORCE[0]:
            closing_force= self.CLOSING_FORCE[0]
        if closing_force >= self.CLOSING_FORCE[1]:
            closing_force= self.CLOSING_FORCE[1]
            
        if clamping_force_peak <= self.CLAMP_FORCE_PEAK[0]:
            clamping_force_peak= self.CLAMP_FORCE_PEAK[0]
        if clamping_force_peak >= self.CLAMP_FORCE_PEAK[1]:
            clamping_force_peak= self.CLAMP_FORCE_PEAK[1]
            
        if torque_peak_val <= self.TORQUE_PEAK_VAL[0]:
            torque_peak_val= self.TORQUE_PEAK_VAL[0]
        if torque_peak_val >= self.TORQUE_PEAK_VAL[1]:
            torque_peak_val= self.TORQUE_PEAK_VAL[1]

        if torque_mean_val <= self.TORQUE_MEAN_VAL[0]:
            torque_mean_val= self.TORQUE_MEAN_VAL[0]
        if torque_mean_val >= self.TORQUE_MEAN_VAL[1]:
            torque_mean_val= self.TORQUE_MEAN_VAL[1]

        if back_press_peak_val <= self.BACK_PRESS_PEAK_VAL[0]:
            back_press_peak_val= self.BACK_PRESS_PEAK_VAL[0]
        if back_press_peak_val >= self.BACK_PRESS_PEAK_VAL[1]:
            back_press_peak_val= self.BACK_PRESS_PEAK_VAL[1]

        if inject_press_peak_val <= self.INJECT_PRESS_PEAK_VAL[0]:
            inject_press_peak_val= self.INJECT_PRESS_PEAK_VAL[0]
        if inject_press_peak_val >= self.INJECT_PRESS_PEAK_VAL[1]:
            inject_press_peak_val= self.INJECT_PRESS_PEAK_VAL[1]

        if screw_hold_pos <= self.SCREW_HOLD_POS[0]:
            screw_hold_pos= self.SCREW_HOLD_POS[0]
        if screw_hold_pos >= self.SCREW_HOLD_POS[1]:
            screw_hold_pos= self.SCREW_HOLD_POS[1]

        if shot_volume <= self.SHOT_VOLUME[0]:
            shot_volume= self.SHOT_VOLUME[0]
        if shot_volume >= self.SHOT_VOLUME[1]:
            shot_volume= self.SHOT_VOLUME[1] 

        normalized_state = normalize_state([
            mold_temp,
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
            ])
        
        self.state = np.array([normalized_state[0],
                               normalized_state[1],
                               normalized_state[2],
                               normalized_state[3],
                               normalized_state[4],
                               normalized_state[5],
                               normalized_state[6],
                               normalized_state[7],
                               normalized_state[8],
                               normalized_state[9],
                               normalized_state[10],
                               normalized_state[11],
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
        
        print("Target: ",self.state[13],'   ',"Agent result: ",self.state[12])
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
        print("Step Reward: ", reward)
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

        reset_MOLD_TEMP = random.uniform(normalized_obs[0][0], normalized_obs[0][-1])
        reset_FILL_TIME = random.uniform(normalized_obs[1][0], normalized_obs[1][-1])
        reset_PLAST_TIME = random.uniform(normalized_obs[2][0], normalized_obs[2][-1])
        reset_CYCLE_TIME = random.uniform(normalized_obs[3][0], normalized_obs[3][-1])
        reset_CLOSING_FORCE = random.uniform(normalized_obs[4][0], normalized_obs[4][-1])
        reset_CLAMP_FORCE_PEAK = random.uniform(normalized_obs[5][0], normalized_obs[5][-1])
        reset_TORQUE_PEAK_VAL = random.uniform(normalized_obs[6][0], normalized_obs[6][-1])
        reset_TORQUE_MEAN_VAL = random.uniform(normalized_obs[7][0], normalized_obs[7][-1])
        reset_BACK_PRESS_PEAK_VAL = random.uniform(normalized_obs[8][0], normalized_obs[8][-1])
        reset_INJECT_PRESS_PEAK_VAL = random.uniform(normalized_obs[9][0], normalized_obs[9][-1])
        reset_SCREW_HOLD_POS = random.uniform(normalized_obs[10][0], normalized_obs[10][-1])
        reset_SHOT_VOLUME = random.uniform(normalized_obs[11][0], normalized_obs[11][-1])
        reset_U0_VALUE_TARGET = round(random.uniform(self.U0_VALUE_TARGET[0], self.U0_VALUE_TARGET[-1]),2)

        self.state = np.array([normalized_obs[0][0], 
                               normalized_obs[1][0], 
                               normalized_obs[2][0], 
                               normalized_obs[3][0], 
                               normalized_obs[4][0], 
                               normalized_obs[5][0],
                               normalized_obs[6][0],
                               normalized_obs[7][0],
                               normalized_obs[8][0],                               
                               normalized_obs[9][0],
                               normalized_obs[10][0],
                               normalized_obs[11][0],
                               predict(np.array([[normalized_obs[0][0], 
                                                      normalized_obs[1][0], 
                                                      normalized_obs[2][0], 
                                                      normalized_obs[3][0], 
                                                      normalized_obs[4][0], 
                                                      normalized_obs[5][0],
                                                      normalized_obs[6][0],
                                                      normalized_obs[7][0],
                                                      normalized_obs[8][0],                               
                                                      normalized_obs[9][0],
                                                      normalized_obs[10][0],
                                                      normalized_obs[11][0]]])),
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
model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=10000000)
# Save Model
injection_path = os.path.join('Training','Saved Models','Injection_Model_TD3_')
model.save(injection_path)

#del model

# Load Model
model = TD3.load(injection_path, env)

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
    