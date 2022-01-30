""" This module contains f
"""
# Standard library imports
import logging

# Third party imports
import numpy as np
from numpy.linalg import norm
from functools import partial


from scipy.integrate import solve_ivp   # Differential equation solvers
from scipy import optimize # Using Newton-Ramson method   

import matplotlib.pyplot as plt




logger = logging.getLogger(__name__)


EARTH_MOON = {
        "alpha" : 0.012153659347055779 ,   
        "m_star" : 6.0456467309e+24,   #kg
        "l_star" : 3.84400e5,     #km  
        "t_star" : 1.1888906e-2,  #years
        "m1" : 5.97217e24,        #earth Kg
        "m2" : 7.34767309e22      #moon  Kg
}

SUN_EARTH = {
        "alpha" : 3.002590280553245e-06,
        "m_star" : 1.98900597217e+30,   #kg
        "l_star" : 149.6e9,     #km  
        "t_star" : 0.15915494309189535,  #years
        "m1" : 1.989e30,        #earth
        "m2" : 5.97217e24      #moon
}

SUN_JUPITER = {
        "alpha" :  0.0009534013700475007,
        "m_star" : 1.9908981249999998e+30,   #kg
        "l_star" : 7.784120e8,    #km
        "t_star" : 1.888328,      #years
        "m1" : 1.989e30,          #sun
        "m2" : 1898.125e24        #jupiter
        
}

SUN_SATURN = {
    "alpha" : 0.00028564839676224095,
    "m_star" : 1.989568317e+30, #kg
    "l_star" : 1.422398e9,  #km
    "t_star" : 4.686932,    #years
    "m1" : 1.989e30,        #sun
    "m2" : 568.317e24       #saturn
}


# Dimension-less version of the differential equation for the rotational reference frame
def dSdt_rot(t, S, alpha):     
    """Differential equation to calculate the movement equation of the m3 body,
    i.e.
    

    Parameters
    ----------
    t : [type]
        [description]
    S : [type]
        [description]
    alpha : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    r_xyz = S[0:3]
    v_xyz = S[3:6]
    w_xyz = np.array([0,0,1])
    ac1_xyz = -2*(np.cross(w_xyz,v_xyz))  # Coriolis 
    ac2_xyz = -np.cross(w_xyz,np.cross(w_xyz,r_xyz)) #Centrifugal 
    r13_xyz = r_xyz - np.array([-alpha,0,0])
    r23_xyz = r_xyz - np.array([1-alpha,0,0])
    ac3_xyz = -(1-alpha)*r13_xyz/np.power(np.linalg.norm(r13_xyz),3) # gravity exerted by m1 on m3
    ac4_xyz = -alpha*r23_xyz/np.power(np.linalg.norm(r23_xyz),3)     # gravity exerted by m2 on m3
    acc_xyz = ac1_xyz+ ac2_xyz + ac3_xyz + ac4_xyz
    return np.concatenate((v_xyz, acc_xyz))    


# A partial funtion is defined to look for the zeros of the acceleration function 
# A one of this has to be defined for any system to study (the alpha parameter)
# The S and alpha parameters are left free
partial_acc_rot = partial(dSdt_rot, 0) 


def dSdt_rot_dim(t, S, w, G, alpha, m1, x1, x2):  
    """The dimensional version is put as a function of alpha to facilitate the root finding procedure

    Parameters
    ----------
    t : [type]
        [description]
    S : [type]
        [description]
    w : [type]
        [description]
    mu1 : [type]
        [description]
    x1 : [type]
        [description]
    mu2 : [type]
        [description]
    x2 : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    r_xyz = S[0:3]
    v_xyz = S[3:6]
    w_xyz = np.array([0,0,w])
    ac1_xyz = -2*np.cross(w_xyz,v_xyz) # Coriolis 
    ac2_xyz = -np.cross(w_xyz,np.cross(w_xyz,r_xyz)) #Centrifugal 
    r13_xyz = r_xyz - np.array([x1,0,0])
    r23_xyz = r_xyz - np.array([x2,0,0])
    mu1 = G*m1
    mu2 = mu1*(alpha/(1-alpha))
    ac3_xyz = -mu1*r13_xyz/np.power(norm(r13_xyz),3)     # gravity exerted by m1
    ac4_xyz = -mu2*r23_xyz/np.power(norm(r23_xyz),3)     # gravity exerted by m2
    acc_xyz = ac1_xyz+ ac2_xyz + ac3_xyz + ac4_xyz
    return np.concatenate((v_xyz, acc_xyz))

# With dimensions for the E
G = 6.6742e-20 # Km^3/Kg*s^2
MOON_ROTATION_PERIOD_IN_SECS=27*24*3600+7*3600+43*60+11.5
W = 2*np.pi/MOON_ROTATION_PERIOD_IN_SECS # rad/sec
X1 = -4656
X2 = 379344
t0 = 0
tf = MOON_ROTATION_PERIOD_IN_SECS*1
R = EARTH_MOON['l_star']

partial_acc_rot_dim = partial(dSdt_rot_dim, 0, w=W, G=G, m1=EARTH_MOON['m1'], x1=X1, x2=X2)


def dSdt_three_bodies_3d (t, S, G, m1, m2, m3):
    r1 = S[0:3]
    r2 = S[3:6]
    r3 = S[6:9]
    
    v1 = S[9:12]
    v2 = S[12:15]
    v3 = S[15:18]
    
    r12__3 = np.power(norm(r1-r2),3)
    r23__3 = np.power(norm(r2-r3),3)
    r13__3 = np.power(norm(r1-r3),3)
    
    a1 = G*((m2*(r2-r1)/r12__3) + (m3*(r3-r1)/r13__3))
    a2 = G*((m1*(r1-r2)/r12__3) + (m3*(r3-r2)/r23__3))
    a3 = G*((m1*(r1-r3)/r13__3) + (m2*(r2-r3)/r23__3))
    
    return np.concatenate((v1,v2,v3,a1,a2,a3))


def dSdt_inert_3body (t, S, m1, m2, G = 1):
    r1 = S[0:3]
    v1 = S[3:6]
    
    r2 = S[6:9]
    v2 = S[9:12]
    
    r3 = S[12:15]
    v3 = S[15:18]
    
    r13 = r3-r1
    r23 = r3-r2
    r12 = r2-r1
       
    a1 = G*(m2*r12/np.power(norm(r12),3))
    a2 = G*(-m1*r12/np.power(norm(r12),3))
    a3 = G*(-m1*r13/np.power(norm(r13),3) - m2*r23/np.power(norm(r23),3))
        
    return np.concatenate((v1,a1, v2, a2, v3, a3))



def un_rotate(xy_rot, ts, w=1):
    xy = np.zeros(xy_rot.shape)
    xy[:,0] = np.multiply(xy_rot[:,0],np.cos(ts)) - np.multiply(xy_rot[:,1], np.sin(ts))
    xy[:,1] = np.multiply(xy_rot[:,0],np.sin(ts)) + np.multiply(xy_rot[:,1], np.cos(ts))
    return xy


def plot_orbits(sols_st, limit, plt_style='seaborn-whitegrid'):
    plt.style.use(plt_style)
    fig, ax = plt.subplots(nrows=len(sols_st),figsize=(9,5*5))
    fig.subplots_adjust(hspace=0.2,top=0.99)
    colors= ['bo','co','ro']
    for idx, sol in enumerate(sols_st) :
        ax[idx].set_xlabel('x')
        ax[idx].set_ylabel('y')
        x_limits = (-limit,limit)
        y_limits = (-limit,limit)
        ax[idx].set_xlim(x_limits)
        ax[idx].set_ylim(y_limits)
        for color_idx, (body_name, st) in enumerate(sol.items()):
            ax[idx].plot(st[:,0:1], st[:,1:2] , colors[color_idx], ms = 1, label=body_name )
        ax[idx].set_title("L"+str(idx+1))
    ax[0].legend()    



def propagate_orbit(t0, alpha, initial_states, tspans, diff_eq, t_eval=None, method='LSODA'):
    """[summary]

    Parameters
    ----------
    t0 : [type]
        [description]
    alpha : [type]
        [description]
    initial_states : [type]
        [description]
    tspans : [type]
        [description]
    diff_eq : [type]
        [description]
    t_eval : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    sols = []
    for idx, Y0 in enumerate(initial_states):
        tf=tspans[idx]
        sols.append(solve_ivp(diff_eq, (t0, tf), Y0, args=(alpha,), method=method, rtol= 1e-13, t_eval=t_eval, dense_output=False))
        print (len(sols[idx].t))
    return sols

# These are the theoretical Lagrange points. The acceleration as seen from the rotating reference frame should be 0. However,
# this is not the case. We use the Newton method to find the zeros of the acceleration using as x0 values the theoretical lagrange 
# points
def theo_lag_points(alpha, r=1):
    """Computes the theoretical Lagrange points (they are not exact solutions)
    
    Parameters
    ----------
    alpha : [type]
        [description]
    r : float
        The distance from m1 to m2 [Km]

    Returns
    -------
    [type]
        [description]
    """
    return [np.array([r*(1-np.cbrt(alpha/3)),0,0]),
            np.array([r*(1+np.cbrt(alpha/3)),0,0]),
            np.array([-r*(1+(5*alpha/12)),0,0]),
            np.array([r*(0.5-alpha),np.sqrt(3)*r/2,0]),
            np.array([r*(0.5-alpha),-np.sqrt(3)*r/2,0])]
    
    
def opt_lag_points(func, alpha, r=1, traces=True):
    new_lag_points = []
    for idx, lag_point, in enumerate(theo_lag_points(alpha, r)):
        X0 = lag_point[0]
        y = lag_point[1]
        f = lambda x: np.linalg.norm(func(S=np.array([x,y,0,0,0,0]),alpha=alpha))
        if idx+1 <= 3 or r==1 :
            root = optimize.newton(f, X0, tol=1.48e-14)
        else :
            delta = 0.00001*r
            root = optimize.brent(f, brack=(X0-delta, X0+delta), tol=1.48e-1)    
        new_lag_points.append(np.array([root,y,0]))
        if traces :
            print (f'L{idx+1}: th:[{X0},{y}], opt:[{root},{y}] where acc={f(root)}')
    return new_lag_points



def curve_length(curve):
    ''' Assumed a shape of n,3'''
    diff_vectors = np.diff(curve, axis=0)
    return np.sum(np.linalg.norm(diff_vectors,axis=1))

def dSdt_nonrotating(t, S, G, m1, m2):
    x1 = S[0:2]
    x2 = S[2:4]
    x3 = S[4:6]
    
    v1 = S[6:8]
    v2 = S[8:10]
    v3 = S[10:12]
    
    r12__3 = np.linalg.norm(x1-x2)**3
    r23__3 = np.linalg.norm(x2-x3)**3
    r13__3 = np.linalg.norm(x1-x3)**3
    
    a1 = G*((m2*(x2-x1)/r12__3))
    a2 = G*((m1*(x1-x2)/r12__3))
    a3 = G*((m1*(x1-x3)/r13__3) + (m2*(x2-x3)/r23__3))
    #print (G*(m1*(x1-x3)/r13__3), G*(m2*(x2-x3)/r23__3), a3)
    result = np.concatenate((v1,v2,v3,a1,a2,a3))
    return result
    








if __name__ == "__main__":
    None