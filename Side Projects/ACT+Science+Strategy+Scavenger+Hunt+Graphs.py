
# coding: utf-8

# In[9]:

get_ipython().magic('matplotlib inline')
from matplotlib import rcParams
rcParams['savefig.dpi'] = 90
import matplotlib.pyplot as plt
import numpy as np

#Figure 1 equations
def trial_1(x):
    return 3*np.sqrt(x)

def trial_2(x):
    return 3*x

def trial_3(x):
    return np.piecewise(x, [x < 5, x >= 5], [lambda x: 1.5*x, lambda x: -3 * np.sqrt(x - 5) + 7.5])
    #if x < 5:
    #    return 1.5 * x
    #else:
    #    return 0.5 * np.exp(x + np.log(15)/2 - 5)

def trial_4(x):
    return np.piecewise(x, [x < 2, x >= 7], [lambda x: 22, lambda x: 0, lambda x: -22/5 * x + 22*7/5])
    #if x < 2.5:
    #    return 8.5
    #elif x < 7:
        #return 13*x/9 + (2 - 13 * 7/9)
    #else:
        #return 2.0

    
#plotting Figure 1
X = np.linspace(0, 10)
plt.xlabel('Amount of honey (tbs)')
plt.ylabel('Number of fruitflies')
plt.title('Figure 1')
plt.plot(X, trial_1(X), 'k-', label = 'Trial 1')
plt.plot(X, trial_2(X), 'k--', label = 'Trial 2')
plt.plot(X, trial_3(X), 'k-.', label = 'Trial 3')
plt.plot(X, trial_4(X), 'k.', label = 'Trial 4')
plt.legend(loc = 0)
plt.savefig('ACT Science Review - Figure 1.png')



# In[10]:

#Figure 2 graph
X = np.arange(5)
periods_of_rotation = [30, 75, 45, 15, 37]
plt.bar(X, periods_of_rotation)
plt.ylabel('Period of rotation (in days)')
plt.xlabel('Star')
plt.xticks(X, ('1', '2', '3', '4', '5'))
plt.title('Figure 2a')
plt.savefig('ACT Science Review - Figure 2a.png')


# In[12]:

#Figure 3 equations
def star_1(t):
    return 15 * np.exp(-t/7)

def star_2(t):
    return 22 + np.exp(-(t - 4))

def star_3(t):
    return np.sqrt(22**2 * (1 + (t/15)**2))

def star_4(t):
    return t**0 * 5

def star_5(t):
    return -t/3 + 80

X = np.linspace(0, 60)


f, (ax, ax2) = plt.subplots(2, 1, sharex = True)


ax.plot(star_1(X), 'k-', label = 'Star 1')
ax.plot(star_2(X), 'k--', label = 'Star 2')
ax.plot(star_3(X), 'k-.', label = 'Star 3')
ax.plot(star_4(X), 'k.', label = 'Star 4')
ax.plot(star_5(X), 'k:', label = 'Star 5')
ax2.plot(star_1(X), 'k-', label = 'Star 1')
ax2.plot(star_2(X), 'k--', label = 'Star 2')
ax2.plot(star_3(X), 'k-.', label = 'Star 3')
ax2.plot(star_4(X), 'k.', label = 'Star 4')
ax2.plot(star_5(X), 'k:', label = 'Star 5')
ax.set_title('Figure 2b')
ax.set_ylim(50, 90)
ax2.set_ylim(0, 25)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.tick_params(bottom = 'off', top = 'off')
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax2.legend(loc = 0)

plt.xlabel('Time (days)')
plt.ylabel('Number of Starspots')

plt.legend(loc = 0)

plt.savefig('ACT Science Review - Figure 2b.png')


# In[13]:

#Figure 4 fake data
X = np.arange(1, 5)

data_violists = np.fabs(np.random.normal(2.1, 1.0, 50))
data_violinists = np.fabs(np.random.normal(8.5, 7, 50))
new_data_violinists = np.where(data_violinists < 20, data_violinists, 9)
data_cellists = np.fabs(np.random.normal(6.2, 4.0, 50))
data_bassists = np.fabs(np.random.normal(6.1, 3.5, 50))
data_fig_4 = [new_data_violinists, data_violists, data_cellists, data_bassists]
plt.clf()
plt.boxplot(data_fig_4, 0, '')
plt.xticks(X, ['Violinists', 'Violists', 'Cellists', 'Bassists'])
plt.ylabel('Number of Hours Practiced')
plt.title('Figure 4')
plt.savefig('ACT Science Review - Figure 4.png')


# In[23]:

#Figure 5 Equations
def gal_1_v(r):
    return r**0 * 6
def gal_2_v(r):
    return 3 * np.sin(r/5 + 3) + 2
def gal_3_v(r):
    return 4*np.sqrt(r / 3)
def gal_4_v(r):
    return (r + 2) * 0.5 + 1

def gal_1_z(r):
    return r**0 * -0.45
def gal_2_z(r):
    return -0.1 * np.cos(r/5 - 2) - 0.4
def gal_3_z(r):
    return -0.2*np.sqrt(r / 4)
def gal_4_z(r):
    return -0.015 * r - 0.1


R = np.linspace(0, 40)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(left=0.2, wspace=0.6)

#Plot 1 - velocity of galaxy 1 and galaxy 2
ax1.set_title('Radial Velocity')
ax1.plot(R, gal_1_v(R), 'k-', label = 'Gal. 1')
ax1.plot(R, gal_2_v(R), 'k:', label = 'Gal. 2')
ax1.set_ylabel('Velocity (km/s)')
ax1.legend(loc = 0)

#Plot 2 - velocity of galaxy 3 and galaxy 4
ax3.plot(R, gal_3_v(R), 'k--', label = 'Gal. 3')
ax3.plot(R, gal_4_v(R), 'k-.', label = 'Gal. 4')
ax3.set_ylabel('Velocity (km/s)')
ax3.legend(loc = 0)
ax3.set_xlabel('Radius (kpc)')

#Plot 3 - redshift of galaxy 1 and galaxy 2
ax2.set_title('Redshift')
ax2.plot(R, gal_1_z(R), 'k-', label = 'Gal. 1')
ax2.plot(R, gal_2_z(R), 'k:', label = 'Gal. 2')
ax2.set_ylabel('Redshift')
ax2.legend(loc = 0)

#Plot 4 - redshift of galaxy 3 and galaxy 4
ax4.plot(R, gal_3_z(R), 'k--', label = 'Gal. 3')
ax4.plot(R, gal_4_z(R), 'k-.', label = 'Gal. 4')
ax4.set_ylabel('Redshift')
ax4.legend(loc = 0)
ax4.set_xlabel('Radius (kpc)')

plt.savefig('ACT Science Review - Figure 5.png')


# In[ ]:



