#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Time step
t0 = 0
tm = 8*24
dt = 1
#%%
# Intial number of agents
agents_i = 1000
# Transition rates
# Transition from E :NS
p = 0.01 # to E : S
q = 0.02 # to H : NS
r = 0.001 # to H : S
# Transition from E: S
l = 0.001 # to E : NS
m = 0.05 # to H : NS
n = 0.05 # to H : S
# Transition from H: NS
x = 0.005 # to H: S
y = 0.001 # to M: NS
z = 0.005 # to M: S
# Transition from H: S
a = 0.05 # to H: NS
b = 0.04 # to M: NS
c = 0.005 # to M: S
# Transition from M: NS
f = 0.005 # to M: S
# Transition from M: S
g = 0.005 # to M: NS
# Proportion in each state
prop_i = np.array([0.635,0,0.35,0,0.015,0])

#%% Switching parameters
params = np.array([
    [1-(p+q+r),p,q,r,0,0],
    [l,1-(l+m+n),m,n,0,0],
    [0,0,1-(x+y+z),x,y,z],
    [0,0,a,1-(a+b+c),b,c],
    [0,0,0,0,1-f,f],
    [0,0,0,0,g,1-g],
    ])
#%% Initial state
curr_state = np.random.choice(6,agents_i,p=prop_i)
pop_size = np.bincount(curr_state,minlength=6)
pop_array = [pop_size]
#%%
for t in np.arange(t0,tm,dt):
    # Update the number of agents in each state
    for i in range(len(curr_state)):
        curr_state[i] = np.random.choice(6, p=params[curr_state[i]])
    pop_size = np.bincount(curr_state,minlength=6)
    pop_array.append(pop_size)
# %%
pop_array = pd.DataFrame(pop_array)
pop_array.columns = ['E_NS','E_S','H_NS','H_S','M_NS','M_S']
pop_array = pop_array.div(pop_array.sum(axis=1),axis=0)*100
# %%
pop_array.to_csv('pop_array.csv')
# %%
t = np.arange(t0,tm+dt,dt)/24
fig, ax = plt.subplots(2,sharex=True,figsize=(10,15))
ax[0].plot(t,pop_array['E_NS'],label='E_NS',color='blue')
ax[0].plot(t,pop_array['H_NS'],label='H_NS',color='green')
ax[0].plot(t,pop_array['M_NS'],label='M_NS',color='red')
ax[0].legend()
ax[0].set_ylabel('Population')
ax[1].plot(t,pop_array['E_S'],label='E_S',color='blue')
ax[1].plot(t,pop_array['H_S'],label='H_S',color='green')
ax[1].plot(t,pop_array['M_S'],label='M_S',color='red')
ax[1].legend()
ax[1].set_xlabel('Time (days)')
ax[1].set_ylabel('Population')
plt.tight_layout()
plt.savefig('pop_array.svg')
# %%
