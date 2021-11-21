import logging
import random
from gym import spaces
import gym
import operator
import numpy as np
from scipy.optimize import curve_fit
logger = logging.getLogger(__name__)

class BOTDAEnv(gym.Env):
#构造函数

  def __init__(self):      
    self.startscanfreq = 10500
    self.endscanfreq = 11000
    self.min_scanstep = 1
    self.max_scanstep = 10
    self.avetimes = 8
    self.snr=7.9
    self.goalRMSE=0.1
    self.scanstep0=self.np_random(self.min_scanstep,self.max_scanstep)
    
    self.low_R = np.array(self.max_scanstep.getRMSE,dtype=np.float32)
    self.high_R = np.array(self.min_scanstep.getRMSE,dtype=np.float32)
    
    #状态空间
    self.observation_space = spaces.Box(self.low_R, self.high_R, dtype=np.float32)
    #动作空间
    self.action_space = spaces.Discrete(2)

#初始化随机数生成器
def seed(self,seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed] 
#选择动作并返回奖励
def step(self,action):
    #选择0表示减小，选择1表示增大
    assert self.action_space.contains(action),"%r(%s)invalid"%(action,type(action))
    self.state=self.scanstep0.getRMSE
    if action == 0: #减小
        self.scanstep = self.scanstep0 -1
        #判断动作好坏
        if snr >snr0:
            reward = +1
        else:
            reward = -1
    if action ==1: #增大
        self.scanstep = self.scanstep0 +1
        if snr >snr0:
            reward = +1
        else:
            reward = -1
    self.next_state=self.scanstep.getRMSE
    return reward

#不需要render函数
#初始化环境
def reset(self):
    self.startscanfreq = 10500
    self.endscanfreq = 11000
    self.min_scanstep = 1
    self.max_scanstep = 10
    self.avetimes = 8
    self.snr=7.9
    self.goalRMSE=0.1
    self.scanstep0=self.np_random(self.min_scanstep,self.max_scanstep)
    self.state=self.scanstep0.getRMSE
    return self.state

#洛伦兹方程
def lorentzFunction(x, gain, bShift, bBandWidth):
    
        return gain / (1 + (x - bShift) *(x - bShift) / bBandWidth / bBandWidth * 4)  
#洛伦兹拟合
def lorentzFit(freqArray, posBGS, g1=0.9, g2=1.1, freq1=10700, freq2=10900 , bw1=10, bw2=200):
        pOpt, pCov = curve_fit(lorentzFunction,
                            freqArray,
                            posBGS,
                            bounds=([g1, freq1,bw1], [g2, freq2,bw2]))
        return pOpt, pCov
#得到状态均方根误差
def getRMSE(realbfs,Fre):
    freqNum = int((endFreq - startFreq)/stepFreq) + 1
    freqArray = np.linspace(startFreq, endFreq, freqNum)
    posNum = 4096
    posArray = np.arange(posNum) * 0.4
    pThreshold = 1.0
    posBathStart, posBathEnd = 1500, 1680 
    posBath=range(posBathStart,posBathEnd)
    Fre=np.zeros(len(posBath))
    fitIndexStart, fitIndexEnd = 300, 480    
    for id in range(len(posBath)):
        print(id)
        posBGS = el.lorentzFunction(freqArray, 1, 10860, 50)+0.16*np.random.randn(len(freqArray))  #通过改变前面的值来改变信噪比
        posBGSForFit = posBGS[fitIndexStart:(fitIndexEnd+1)]
        freqArrForFit = freqArray[fitIndexStart:(fitIndexEnd+1)]        
        pOpt, pCov = el.lorentzFit(freqArrForFit, posBGSForFit, freq1=freqArray[fitIndexStart], freq2=freqArray[fitIndexEnd])
        print(pOpt)
        ValForLrzFit = el.lorentzFunction(freqArrForFit, pOpt[0], pOpt[1], pOpt[2])
        Fre[id]=pOpt[1]

    error = []
    for i in range(len(Fre)):
        error.append(realbfs - Fre[i])
        print("Errors: ", error)
        print(error)
        squaredError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方     
        print("Square Error: ", squaredError)
        print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))
    return RMSE
def close(self):
    if self.viewer:
        self.viewer.close()
        self.viewer = None
