import random
from numeric import array,norm
import sys
import nef
import nps
import hrr
import datetime
import timeview.components.core as core
import timeview.view
import os
from ca.nengo.math.impl import ConstantFunction
from ca.nengo.model.impl import FunctionInput
from ca.nengo.model import Units

# For experiment running
#directory = 'trevor/data'
directory = None
e_name = 'Normal'
num_expts = 26

# State lengths
delay_dur = 0.3
approach_dur = 0.1
reward_dur = 0.1

# Experiment parameters
state_d = 4
action_d = 2
NperD = 40

# Gate parameters
gate_N = 100
gate_intercept = (-0.15,0.2)
gate_rate = (50,150)

class BanditTask(nef.SimpleNode):
    def __init__(self,name,dims,trials_per_block=40,block_rewards=[[0.21,0.63],[0.63,0.21],[0.12,0.72],[0.72,0.12]]):
        # parameters
        self.dims = dims
        self.trials_per_block = trials_per_block
        if len(block_rewards[0]) != dims:
            raise Exception('block_reward dimensionality must match dims')
        self.block_rewards = block_rewards
        
        # vars and constants
        self.trial_num = 0
        self.delay_t = 0.0
        self.approach_t = 0.0
        self.reward_t = 0.0
        self.reward = [0.0] * dims
        self.thalamus_sum = [0.0] * dims
        self.thalamus_choice = 0
        self.rewarded = 0
        self.reward_val = 1.0
        self.gate_val = [0.9]
        self.vstr_gate_val = [1.0]
        self.data_log = []
        
        # generate random state_d-d unit vector
        self.ctx_val = array([random.gauss(0,1) for i in range(state_d)])
        self.ctx_val /= norm(self.ctx_val)   
        
        self.state = 'delay'

        nef.SimpleNode.__init__(self,name)

    def get_experiment_length(self):
        leeway = 0.003
        return (delay_dur + approach_dur + reward_dur +
                leeway) * self.trials_per_block * len(self.block_rewards)

    def origin_cortex(self):
        return self.ctx_val

    def origin_cortex_gate(self):
        return self.gate_val
    
    def origin_vstr_gate(self):
        return self.vstr_gate_val
    
    def origin_reward(self):
        return self.reward

    def termination_thalamus(self,x):
        t = self.t_start
        
        if self.state == 'delay':
            self.gate_val = [self.gate_val[0]+0.001]
            self.vstr_gate_val = [self.vstr_gate_val[0]+0.001]
            self.reward = [0.0] * self.dims
            if t >= self.delay_t + delay_dur:
                self.state = 'go'
        elif self.state == 'go':
            self.gate_val = [0.0]
            self.thalamus_sum = [0.0] * self.dims
            self.trial_num += 1
            self.approach_t = t
            self.state = 'approach'
        elif self.state == 'approach':
            for i in range(self.dims):
                self.thalamus_sum[i] += x[i]
            
            if t >= self.approach_t + approach_dur:
                thalamus_min = min(self.thalamus_sum)
                for i in range(len(self.thalamus_sum)):
                    if self.thalamus_sum[i] == thalamus_min:
                        self.thalamus_choice = i
                
                block = (self.trial_num-1) / self.trials_per_block
                if (block >= len(self.block_rewards)):
                    self.state = 'reward'
                    return
                
                ##########
                ## NB!! ##
                ##########
                rand = random.random()
                if rand <= self.block_rewards[block][self.thalamus_choice]:
                    self.rewarded = 1
                    self.reward = [-1.0*self.reward_val] * self.dims
                    self.reward[self.thalamus_choice] = self.reward_val
                else:
                    self.rewarded = 0
                    self.reward = [self.reward_val] * self.dims
                    self.reward[self.thalamus_choice] = -1.0 * self.reward_val
                
                # out_file structure:
                #  trial number, choice, rewarded, thalamus_sums
                out_l = str(self.trial_num)+', '+str(self.thalamus_choice)+', '+str(self.rewarded)
                for i in range(len(self.thalamus_sum)):
                    out_l += ', '+str(self.thalamus_sum[i])
                self.data_log.append(out_l)
                
                self.reward_t = t
                self.state = 'reward'
        elif self.state == 'reward':
            self.vstr_gate_val = [0.0]
            if t >= self.reward_t + reward_dur:
                self.gate_val = [1.0]
                self.delay_t = t
                self.state = 'delay'
    
    def write_data_log(self, filename):
        """Attempts to write the contents of self.data_log to
        the file pointed to by the consumed string, filename.
        If there is an error writing to that file,
        the contents of self.data_log are printed to console instead.
        """
        try:
            f = open(filename, 'a+')
        except:
            print "Error opening %s" % filename
            return self.print_data_log()
        
        for line in self.data_log:
            f.write("%s\n" % line)
        f.close()
    
    def print_data_log(self):
        """Prints the contents of self.data_log to the console."""
        for line in self.data_log:
            print line

class BanditWatch:
    def __init__(self,objs):
        self.objs=objs
    def check(self,obj):
        return obj in self.objs
    def measure(self,obj):
        r=[]
        r.append(obj.trial_num)
        r.append(obj.state)
        r.append(obj.thalamus_choice)
        r.append(obj.rewarded)
        for sum in obj.thalamus_sum:
            r.append(sum)
        
        return r
    def views(self,obj):
        return [('bandit task',BanditView,dict(func=self.measure,label="Bandit Task"))]

from javax.swing.event import *
from java.awt import *
from java.awt.event import *
class BanditView(core.DataViewComponent):
    def __init__(self,view,name,func,args=(),label=None):
        core.DataViewComponent.__init__(self,label)
        self.view=view
        self.name=name
        self.func=func
        self.data=self.view.watcher.watch(name,func,args=args)

        self.setSize(200,100)

    def paintComponent(self,g):
        core.DataViewComponent.paintComponent(self,g)
        
        f_size = g.getFont().size
        x_offset = 5

        try:    
            data=self.data.get(start=self.view.current_tick,count=1)[0]
        except:
            return
        
        cur_y = f_size*3
        g.drawString("Trail "+str(data[0]),x_offset,cur_y)
        cur_y += f_size
        g.drawString("State: "+data[1],x_offset,cur_y)
        cur_y += f_size
        g.drawString("Thalamus sum",x_offset,cur_y)
        cur_y += f_size
        cur_x = x_offset
        for sum in data[4:]:
            g.drawString(str(round(sum*100)/100),cur_x,cur_y)
            cur_x += 40
        cur_y += f_size
        g.drawString("Choice: "+str(data[2]),x_offset,cur_y)
        cur_y += f_size
        if data[3]: r_s = "Yes"
        else:       r_s = "No"
        g.drawString("Rewarded: "+r_s,x_offset,cur_y)

def gate_weights(w):
    for i in range(len(w)):
        for j in range(len(w[0])):
            #w[i][j] = -0.02
            w[i][j] = -0.0002
    return w
def rand_weights(w):
    for i in range(len(w)):
        for j in range(len(w[0])):
            w[i][j] = random.uniform(-1e-4,1e-4)
    return w

alpha = 1.0
def pred_error(x):
    # for each action, prediction error is
    #         a   [        R            +   g   * V(S) -   V(S(t-1))]
    return [alpha * (x[2] - x[0]), alpha * (x[3] - x[1])]

def build_network():
    net = nef.Network('BanditTask_o')
    
    experiment = BanditTask('ExperimentRunner',action_d)
    experiment.getTermination('thalamus').setDimensions(action_d)
    net.add(experiment)
    timeview.view.watches.append(BanditWatch([experiment]))
    
    cortex = net.make('Cortex',NperD*state_d,state_d)
    net.connect(experiment.getOrigin('cortex'),cortex)
    
    cortex_gate = net.make('CortexGate',gate_N,1,encoders=[[1.0]],intercept=gate_intercept,max_rate=gate_rate)
    net.connect(experiment.getOrigin('cortex_gate'),cortex_gate)
    net.connect(cortex_gate,cortex,weight_func=gate_weights)
    
    thalamus = net.make('Thalamus',NperD*action_d,action_d)
    net.connect(thalamus,experiment.getTermination('thalamus'))
   
    nps.basalganglia.make_basal_ganglia(net,cortex,thalamus,action_d,NperD,learn=True)
    StrD1 = net.network.getNode('StrD1')
    StrD2 = net.network.getNode('StrD2')

    vStr = net.make('Ventral Striatum',NperD*action_d*2,action_d*2,max_rate=(100,200))
    net.connect(cortex,vStr,index_post=[0,1],weight_func=rand_weights)
    net.connect(experiment.getOrigin('reward'),vStr,index_post=[2,3])
    
    net.connect(vStr,vStr,func=pred_error,index_post=range(action_d),modulatory=True)
    net.connect(vStr,StrD1,func=pred_error,modulatory=True)
    net.connect(vStr,StrD2,func=pred_error,modulatory=True)
    
    l_args = {'stpd':False, 'oja':False, 'rate':1e-7}
    net.learn(vStr,'Cortex','Ventral Striatum',**l_args)
    net.learn_array(StrD1,'Cortex','Ventral Striatum',**l_args)
    net.learn_array(StrD2,'Cortex','Ventral Striatum',**l_args)
    
    vStr_gate = net.make('vStrGate',gate_N,1,encoders=[[1.0]],intercept=gate_intercept,max_rate=gate_rate)
    net.connect(experiment.getOrigin('vstr_gate'),vStr_gate)
    net.connect(vStr_gate,vStr,weight_func=gate_weights)
    
    return net

def run_experiment():
    net = build_network()
    experiment = net.network.getNode('ExperimentRunner')
    
    if directory != None:
        net.network.run(0,experiment.get_experiment_length())
        
        now = datetime.datetime.now()
        f_name = os.path.join(directory, e_name+'-'+now.strftime("%Y-%m-%d_%H-%M-%S")+'.csv')
        f = open(f_name, 'w')
        f.write('delay_dur=%f\napproach_dur=%f\nreward_dur=%f\n' % (delay_dur,approach_dur,reward_dur))
        f.close()
        experiment.write_data_log(f_name)
    return net

if directory is not None:
    for _ in range(num_expts):
        net = run_experiment()
    sys.exit()
else:
    net = build_network()
    net.add_to_nengo()


