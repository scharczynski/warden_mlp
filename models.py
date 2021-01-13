import numpy as np
from maxlikespy.model import Model
import autograd.numpy as np
import autograd.scipy.special as sse
import matplotlib.pyplot as plt
import math


class SigmaMuTauStimWarden(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_2","a_3", "a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class == 2:
                stim_matrix[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class == 3:
                stim_matrix[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class == 4:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 1]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        s, mu, tau, a_1,a_2,a_3, a_4, a_0 = x
        l = 1/tau
        fun1 = 0.5*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
            + (a_2*(self.stim_matrix[:, 1] *fun1.T))
            + (a_3*(self.stim_matrix[:, 2] *fun1.T))
            + (a_4*(self.stim_matrix[:, 3] *fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])


class SigmaMuTau(Model):

    def __init__(self, data):
        super().__init__(data)
        # self.spikes = data['spikes']
        self.param_names = ["sigma", "mu", "tau", "a_1", "a_0"]
        # self.x0 = [100, 5000, 0.001, 1e-5, 1e-5]


    def model(self, x):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''

        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        '''old method'''
        # fun = a_1*np.exp(-0.5*(np.power((self.t-m)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-m)/s))))
        # ) + a_0

        fun = a_1*(0.5*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        return self.model(x)

class Const(Model):

    """Model which contains only a single offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.

    """

    def __init__(self, data):
        super().__init__(data)
        # self.spikes = data['spikes']
        self.param_names = ["a_0"]
        # self.x0 = [0.1]

    def model(self, x, plot=False):
        o = x
        return o

    def objective(self, x):
        fun = self.model(x)
        obj = (np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun)))))
        return obj

    def pso_con(self, x):
        return 1 - x

class SigmaMuTauStim1(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        # self.good_trials = np.where(stim_matrix[:, 0])
        return self.stim_matrix

    def model(self, x, plot=False):
    
        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        fun1 = 0.5*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))
        # y.peak*((1/2)*exp((y.tau/2).*(2*y.mu + y.tau*y.sig^2-2*t)).*erfc((y.mu + y.tau*y.sig^2 - t)/(sqrt(2)*y.sig)));

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0
        # fun = (
        #     (a_1*(fun1[self.good_trials]))
        # ) + a_0
        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        # obj = np.sum(self.spikes[self.good_trials] * (-np.log(fun)) +
        #               (1 - self.spikes[self.good_trials]) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])


class SigmaMuTauStim2(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 2:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        fun1 = 0.5*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])

class SigmaMuTauStim3(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 3:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        fun1 = 0.5*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])

class SigmaMuTauStim4(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu","tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 4:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        fun1 = 0.5*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])


'''Gaussian models
'''

class Gaussian(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "a_1", "a_0"]
        # self.x0 = [100, 5000, 0.001, 1e-5, 1e-5]


    def model(self, x):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''

        sigma, mu, a_1, a_0 = x

        fun = (
            (a_1 * np.exp(-np.power(self.t - mu, 2.) / (2 * np.power(sigma, 2.)))) + a_0)
        return fun

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    # def plot_model(self, x):
        
    #     return self.model(x)

class GaussianStim(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "a_1", "a_2","a_3", "a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class == 2:
                stim_matrix[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class == 3:
                stim_matrix[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class == 4:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 1]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma, mu, a_1,a_2,a_3, a_4, a_0 = x
        base = (
            (np.exp(-np.power(self.t - mu, 2.) / (2 * np.power(sigma, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
            + (a_2*(self.stim_matrix[:, 1] *base.T))
            + (a_3*(self.stim_matrix[:, 2] *base.T))
            + (a_4*(self.stim_matrix[:, 3] *base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])

class GaussianStim1(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma, mu, a_1, a_0 = x
        base = (
            (np.exp(-np.power(self.t - mu, 2.) / (2 * np.power(sigma, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])

class GaussianStim2(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 2:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma, mu, a_1, a_0 = x
        base = (
            (np.exp(-np.power(self.t - mu, 2.) / (2 * np.power(sigma, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        return (np.sum(fun, axis=1)/fun.shape[1])

class GaussianStim3(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 3:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma, mu, a_1, a_0 = x
        base = (
            (np.exp(-np.power(self.t - mu, 2.) / (2 * np.power(sigma, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        return (np.sum(fun, axis=1)/fun.shape[1])
        
class GaussianStim4(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 4:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma, mu, a_1, a_0 = x
        base = (
            (np.exp(-np.power(self.t - mu, 2.) / (2 * np.power(sigma, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        return (np.sum(fun, axis=1)/fun.shape[1])

'''both presentations
'''

class GaussianBoth(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1", "a_0"]
        # self.x0 = [100, 5000, 0.001, 1e-5, 1e-5]


    def model(self, x):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''

        sigma1, mu1, sigma2, mu2, a_1, a_0 = x

        fun = (
            (a_1 * np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.))))  +
            (a_1 * np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))) + a_0)
    
        return fun

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        
        return self.model(x)

class GaussianBothPos(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1","a_2", "a_0"]
        # self.x0 = [100, 5000, 0.001, 1e-5, 1e-5]


    def model(self, x):
        '''One thing to try is to maybe pull out self.t as a kwarg in optimize, might allow jacobian to be calculated easier
        '''

        sigma1, mu1, sigma2, mu2, a_1,a_2, a_0 = x

        fun = (
            (a_1 * np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.))))  +
            (a_2 * np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))) + a_0)
    
        return fun

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def plot_model(self, x):
        
        return self.model(x)


class GaussianStimBoth(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1", "a_2","a_3", "a_4", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix_1 = np.zeros((self.spikes.shape[0], 4))
        stim_matrix_2 = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"] 
            if stim_class[0] == 1:
                stim_matrix_1[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class[0] == 2:
                stim_matrix_1[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class[0] == 3:
                stim_matrix_1[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class[0] == 4:
                stim_matrix_1[int(trial_num)][:] = [0, 0, 0, 1]
            if stim_class[1] == 1:
                stim_matrix_2[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class[1] == 2:
                stim_matrix_2[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class[1] == 3:
                stim_matrix_2[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class[1] == 4:
                stim_matrix_2[int(trial_num)][:] = [0, 0, 0, 1]
        self.stim_matrix_1 = stim_matrix_1
        self.stim_matrix_2 = stim_matrix_2
        return self.stim_matrix_1, self.stim_matrix_2

    def model(self, x, plot=False):
    
        sigma1, mu1, sigma2, mu2, a_1,a_2,a_3, a_4, a_0 = x
        fun_p1 = np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.)))
        fun_p2 = np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))
            
        fun = (
            (a_1*(self.stim_matrix_1[:, 0] * fun_p1.T + self.stim_matrix_2[:, 0] * fun_p2.T))
            + (a_2*(self.stim_matrix_1[:, 1] * fun_p1.T + self.stim_matrix_2[:, 1] * fun_p2.T))
            + (a_3*(self.stim_matrix_1[:, 2] * fun_p1.T + self.stim_matrix_2[:, 2] * fun_p2.T))
            + (a_4*(self.stim_matrix_1[:, 3] * fun_p1.T + self.stim_matrix_2[:, 3] * fun_p2.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])


class GaussianStimBothPos(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", 
            "mu1", 
            "sigma2", 
            "mu2", 
            "a_1", 
            "a_2",
            "a_3", 
            "a_4",
            "a_5",
            "a_6",
            "a_7",
            "a_8",
            "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix_1 = np.zeros((self.spikes.shape[0], 4))
        stim_matrix_2 = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"] 
            if stim_class[0] == 1:
                stim_matrix_1[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class[0] == 2:
                stim_matrix_1[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class[0] == 3:
                stim_matrix_1[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class[0] == 4:
                stim_matrix_1[int(trial_num)][:] = [0, 0, 0, 1]
            if stim_class[1] == 1:
                stim_matrix_2[int(trial_num)][:] = [1, 0, 0, 0]
            elif stim_class[1] == 2:
                stim_matrix_2[int(trial_num)][:] = [0, 1, 0, 0]
            elif stim_class[1] == 3:
                stim_matrix_2[int(trial_num)][:] = [0, 0, 1, 0]
            elif stim_class[1] == 4:
                stim_matrix_2[int(trial_num)][:] = [0, 0, 0, 1]
        self.stim_matrix_1 = stim_matrix_1
        self.stim_matrix_2 = stim_matrix_2
        return self.stim_matrix_1, self.stim_matrix_2

    def model(self, x, plot=False):
    
        sigma1, mu1, sigma2, mu2, a_1,a_2,a_3, a_4,a_5, a_6, a_7, a_8, a_0 = x
        fun_p1 = np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.)))
        fun_p2 = np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))
            
        fun = (
            (a_1*self.stim_matrix_1[:, 0] * fun_p1.T + a_5*self.stim_matrix_2[:, 0] * fun_p2.T)
            + (a_2*self.stim_matrix_1[:, 1] * fun_p1.T + a_6*self.stim_matrix_2[:, 1] * fun_p2.T)
            + (a_3*self.stim_matrix_1[:, 2] * fun_p1.T + a_7*self.stim_matrix_2[:, 2] * fun_p2.T)
            + (a_4*self.stim_matrix_1[:, 3] * fun_p1.T + a_8*self.stim_matrix_2[:, 3] * fun_p2.T)
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])



class GaussianStimBoth1(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma1, mu1, sigma2, mu2, a_1, a_0 = x
        base = (
              (np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.))))  +
            (np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])

class GaussianStimBoth2(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 2:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma1, mu1, sigma2, mu2, a_1, a_0 = x
        base = (
              (np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.))))  +
            (np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])

class GaussianStimBoth3(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 3:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma1, mu1, sigma2, mu2, a_1, a_0 = x
        base = (
              (np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.))))  +
            (np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])

class GaussianStimBoth4(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma1", "mu1", "sigma2", "mu2", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 4:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
        self.stim_matrix = stim_matrix
        return self.stim_matrix

    def model(self, x, plot=False):
    
        sigma1, mu1, sigma2, mu2, a_1, a_0 = x
        base = (
              (np.exp(-np.power(self.t - mu1, 2.) / (2 * np.power(sigma1, 2.))))  +
            (np.exp(-np.power(self.t - mu2, 2.) / (2 * np.power(sigma2, 2.)))))
        fun = (
            (a_1*(self.stim_matrix[:, 0] * base.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)

        # return self.model(x)
        return (np.sum(fun, axis=1)/fun.shape[1])

class ExponentialStim1(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["tau", "s", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 1:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        self.good_trials = np.where(stim_matrix[:, 0])
        return self.stim_matrix

    def model(self, x, plot=False):
    
        tau, s, a_1, a_0 = x
        l = 1/tau
        fun1 = np.exp((-l*self.t) + s)

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

    def plot_model(self, x):
        fun = self.model(x)


        return (np.sum(fun, axis=1)/fun.shape[1])
class ExponentialStim2(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 2:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        self.good_trials = np.where(stim_matrix[:, 0])
        return self.stim_matrix

    def model(self, x, plot=False):
    
        tau, a_1, a_0 = x
        l = 1/tau
        fun1 = np.exp(-l*self.t)

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

class ExponentialStim3(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 3:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        self.good_trials = np.where(stim_matrix[:, 0])
        return self.stim_matrix

    def model(self, x, plot=False):
    
        tau, a_1, a_0 = x
        l = 1/tau
        fun1 = np.exp(-l*self.t)

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj

class ExponentialStim4(Model):
    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["tau", "a_1", "a_0"]
        self.t = np.tile(self.t, (self.num_trials, 1))

    def info_callback(self):
        self.stims = self.info["stim_identity"]
        stim_matrix = np.zeros((self.spikes.shape[0], 4))
        if self.even_odd_trials == "even":
            trials  = list(self.stims.keys())[::2]
        elif self.even_odd_trials == "odd":
            trials = list(self.stims.keys())[1::2]
        else:
            trials  =  list(self.stims.keys())
        #rossi-pool 1 indexed trials
        trial_indices = [x-1 for x in list(map(int, trials))]

        self.t = self.t[trial_indices]
        
        for trial_num, trial in enumerate(trials):
            stim_class = self.stims[(trial)]["sample"][0]  
            if stim_class == 4:
                stim_matrix[int(trial_num)][:] = [1, 0, 0, 0]
            else:
                stim_matrix[int(trial_num)][:] = [0, 0, 0, 0]
        self.stim_matrix = stim_matrix
        self.good_trials = np.where(stim_matrix[:, 0])
        return self.stim_matrix

    def model(self, x, plot=False):
    
        tau, a_1, a_0 = x
        l = 1/tau
        fun1 = np.exp(-l*self.t)

        fun = (
            (a_1*(self.stim_matrix[:, 0] * fun1.T))
        ) + a_0

        return fun

    def objective(self, x):
        fun = self.model(x).T

        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))

        return obj