from numpy import *
import numpy as np
import os, time, random
import itertools
import math


#trans_t=[ 3.71190585, 2.9248834,  2.52259759, 4.9119252,  4.04431769, 2.19391224, 4.80655309, 4.17091216, 2.11234727, 2.83141443] #数据从云端转移至目标结点所需时间
trans_t=[[ 0.        ,2.17647377,3.29679232,2.18950348,4.38583234,3.85756898,3.15996467,3.5829669 ,2.45659308,2.31813054],
         [ 2.17647377,0.        ,3.66456959,4.77604433,3.4979844 ,3.2224891 ,3.80051528,3.8814923 ,3.52127043,4.7755423 ],
         [ 3.29679232,3.66456959,0.        ,3.16256958,2.20463728,3.14822921,2.41417024,4.43390898,3.18227195,2.79879864],
         [ 2.18950348,4.77604433,3.16256958,0.        ,4.47448702,3.0085288 ,2.09185824,3.23725722,2.43997862,2.30279477],
         [ 4.38583234,3.4979844 ,2.20463728,4.47448702,0.        ,3.21630472,2.94914786,3.86679094,4.38781627,4.24697552],
         [ 3.85756898,3.2224891 ,3.14822921,3.0085288 ,3.21630472,0.        ,2.70973157,3.17467046,3.66788722,4.37718992],
         [ 3.15996467,3.80051528,2.41417024,2.09185824,2.94914786,2.70973157,0.        ,3.00440565,3.98004107,4.29971972],
         [ 3.5829669 , 3.8814923,4.43390898,3.23725722,3.86679094,3.17467046,3.00440565,0.        ,2.91234216,4.29798604],
         [ 2.45659308,3.52127043,3.18227195,2.43997862,4.38781627,3.66788722,3.98004107,2.91234216,0.        ,4.93660226],
         [ 2.31813054,4.775542  ,2.79879864,2.30279477,4.24697552,4.37718992,4.29971972,4.29798604,4.93660226,0.        ]]


able_node=[[0, 4, 9],                #5个task对应可去的结点
           [1, 0, 3, 5, 9],
           [1, 7, 8],
           [2, 6, 9, 4],
           [5, 8, 6]]
#print(able_node[1])


#deal_speed=[36.49532163116662, 50.03623322644187, 56.38212800158699, 40.84646555385936, 34.00486842274878, 30.679722175898874, 67.50200948342751, 54.18040367983207, 55.23534283113722, 40.2444736755832]   #各结点处理数据速率
deal_speed=[26.49532163116662, 60.03623322644187, 76.38212800158699, 30.84646555385936, 24.00486842274878, 20.679722175898874, 77.50200948342751, 44.18040367983207, 65.23534283113722, 30.2444736755832]   #各结点处理数据速率
data_basic=[141.2314791229417, 137.06836066804385, 160.15981901380255, 186.47506389026267, 215.51872897313837]
p_time_max=np.zeros(10)

class task():


    def TaskState(self):
        self.capacity=[4,3,1,1,3,4,1,3,3,2]   #初始结点容量
        self.capacity_max=[4,3,1,1,3,4,1,3,3,2]  #节点最大容量
        self.p_time=[[0,0,0,0],                    #初始各结点各进程等待执行的时间
           [0,0,0],
           [0],
           [0],
           [0,0,0],
           [0,0,0,0],
           [0],
           [0,0,0],
           [0,0,0],
           [0,0]]  

        self.data=np.zeros(5)
        self.src_node=np.zeros(5,dtype=int)
        self.deal_speed_now=np.zeros(10)      

        #global data,src_node,deal_speed_now
        
        self.data,self.src_node=self.get_data()
        self.deal_speed_now=self.get_speed()
        
        x=list(itertools.chain.from_iterable(self.p_time))
        y=np.array(x)
        
        src_n=np.zeros((5,10))
        for i in range (5):
            src_n[i][self.src_node[i]]=self.data[i]
        src_n=src_n.reshape(50)
        #print('before:',self.src_node)
        #print('after:',src_n)


        state_next=np.concatenate((y,self.deal_speed_now,src_n))
        
        state_next=state_next.reshape(85)
        
        return state_next
    

    def get_speed(self):
        
        deal_speed_now_=np.zeros(10)
        for i in range(10):
            if self.capacity[i]==0:
                gama=0.3
            else:
                gama=1+math.log(self.capacity[i]/self.capacity_max[i],10)
            deal_speed_now_[i]=deal_speed[i]*gama
 
        return deal_speed_now_
    
    def get_data(self):

        data_=np.zeros(5)
        src_node_=np.zeros(5,dtype=int)
        for j in range(5):
            data_[j]=np.random.normal(data_basic[j],10)
            src_node_[j]=np.random.randint(0,10)
            if random.random() <=0.2:
                data_[j]=data_[j]/2            
        return data_,src_node_  


    def task_step(self,action):  #执行每一组task分配  给了action返回observation(下一组task到来时的状态), reward
        #global data,src_node,deal_speed_now

        action=action.reshape((3,5,3,4,3))

        strategy_tmp=[0 for i in range(5)]    #action是一维数组，1表示采取该动作
        #print('action:',action)
        a=np.max(action)
        tmp=np.where(action==a)
        #print(len(tmp[0]))
        choose=np.random.randint(len(tmp[0]))
        for i in range(5):
            strategy_tmp[i]=tmp[i][choose]
        #print('strategy:',strategy_tmp)
                
        ###strategy_tmp=deal_task(capacity)   #获取每个task对应去的node方案
        

        strategy=[0]*5
        tmp_dealtime=[0]*5
        tmp_transtime=[0]*5
        tmp_waittime=[0]*5
        strategy_process=[0]*5
        tmp_task_time=[0]*5
        
        #deal_speed_now=self.get_speed()
        
        for k in range(5) : #k-任务标号  able_node[k][strategy_tmp[k]]-第k个任务分配的node
           
            strategy[k]=able_node[k][strategy_tmp[k]]  #每个task对应去的node方案
            
            tmp_dealtime[k]=self.data[k]/self.deal_speed_now[strategy[k]]  #结点处处理时间
            
            tmp_transtime[k]=trans_t[self.src_node[k]][strategy[k]] #搬移时间
            
            tmp_waittime[k]=np.min(self.p_time[strategy[k]])  #要等待时间最短的进程 等待时间
            strategy_process[k]=np.argmin(self.p_time[strategy[k]])  #对应所去node的进程
            tmp_task_time[k]=tmp_waittime[k]+tmp_transtime[k]+tmp_dealtime[k]
            self.p_time[strategy[k]][strategy_process[k]]=tmp_task_time[k]
        #print('tmp_task_time',tmp_task_time)

        task_time=np.max(tmp_task_time)
        
        for i in range(10):
            p_time_max[i]=np.max(self.p_time[i])
        tmp_load=np.std(p_time_max, ddof=1)
        print(tmp_load)

        t=9  #每组task间隔时间
        for m in range(10):
            for n in range(len(self.p_time[m])):
                self.p_time[m][n]=max(0,self.p_time[m][n]-t)  #更新每个进程的时间
        
        for c in range(10):
            self.capacity[c]=self.p_time[c].count(0)    #更新容量
        


        self.data,self.src_node=self.get_data()
        self.deal_speed_now=self.get_speed()
        
        x=list(itertools.chain.from_iterable(self.p_time))
        y=np.array(x)
        
        src_n=np.zeros((5,10))
        
        for i in range (5):
            src_n[i][self.src_node[i]]=self.data[i]
        src_n=src_n.reshape(50)
        #print('before:',self.src_node)
        #print('after:',src_n)
  
        state_next=np.concatenate((y,self.deal_speed_now,src_n))
        
        state_next=state_next.reshape(85)

        reward=10-task_time
        #print('self.p_time',self.p_time)
        return state_next,reward,tmp_load    


 

 
        



