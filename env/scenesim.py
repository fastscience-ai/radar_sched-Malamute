#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 22:22:02 2023

@author: e23615
"""

import random
import numpy as np
import simpy


RANDOM_SEED = 45
NUM_TARGET = 2          # Number of targets
MOV_MEAN = 10.0         # Avg. target moving time in minutes
MOV_SIGMA = 2.0         # Sigma of target moving time
STA_MEAN = 15.0         # Avg. target stationary time in minutes
STA_SIGMA = 3.0         # Sigma of target stationary time

VEL_MEAN = 2.0          # Avg. target moving velocity
VEL_SIGMA = 1.0         # Sigma of target moving velocity

GRID_SIZE = 800         # Size of grid on which targets move
MID_GRID  = 400         # Center of grid
GUARD_BAND = 100        # Size of guard band in which targets move but are not detected

MAX_STRENGTH = 30       # Maximum strength of target (fixed for now)
STRENGTH_INC = 10       # Strength increase for target which is detected (fixed for now)
COAST_PEN = 1           # Penalty for coasting (i.e. not detecting) a target in a given time step

SIM_TIME = 5            # Length of simulation

def movetime():
    return random.normalvariate(MOV_MEAN, MOV_SIGMA)

def stattime():
    return random.normalvariate(STA_MEAN, STA_SIGMA)

def time_to_interrogate():
    
    done_in2 = 1
    
    return done_in2

class Target(object):
    """A target either moves at a constant speed or sits stationary.

    A target has a *name*, a *Position*, a *velocity*, and a *state* (moving or stationary).

    """
    def __init__(self, env, name, posx, posy, velx, vely, state):
        self.env = env
        self.name = name
        self.posx = posx
        self.posy = posy
        self.velx = velx
        self.vely = vely
        self.state = state       # 1 = moving, 0 = stationary        
        self.process = env.process(self.move(name, posx, posy, velx, vely, state))
        env.process(self.interrogate())

    def move(self, name, posx, posy, velx, vely, state):
        """Move from location posx posy with velx and vely until end of motion.
        
        velx and vely set by state (=assigned values when state=1, =0 when state=0)

        When motion ends, change state and repeat until simulation ends

        """
        while True:
            # Start moving
            done_in = movetime()
            while done_in:
                try:
                    # Move the target
                    start = self.env.now
                    if self.state == 1:
                        vx = velx
                        vy = vely
                    else:
                        vx = 0
                        vy = 0
                    
                    # Yield current move when time done                    
                    yield self.env.timeout(done_in)
                    
                    # Calulate position at end of move
                    posx = posx + done_in * vx
                    posy = posy + done_in * vy
                    self.posx = posx
                    self.posy = posy
                    
                    # For next move, change state
                    if self.state == 0 :
                        self.state = 1
                    else :
                        self.state = 0

                    print("Target %d done move at position %.2f %.2f time %.2f now in state %d." % (name, posx, posy, self.env.now, self.state))
                    done_in = 0  # Set to 0 to exit while loop.
             
                except simpy.Interrupt:
                    posx = posx + (self.env.now - start) * vx
                    posy = posy + (self.env.now - start) * vy
                    print("Target state at interrupt time %.2f :" %self.env.now)
                    print("Target %d state %d position %.2f %.2f" % (name, self.state, posx, posy))
                    done_in -= self.env.now - start  # How much time left?
                    self.posx = posx
                    self.posy = posy

    def interrogate(self) :
        #Interrogate evolving targets to obtain their current state and position
        while True:
            yield self.env.timeout(time_to_interrogate())

            self.process.interrupt()
        
    def __repr__(self): 
        return "Target,%d,state,%d,xpos,%.2f,ypos,%.2f" % (self.name, self.state, self.posx, self.posy) 
    
# Setup and start the simulation
print('Target Simulation')
random.seed(RANDOM_SEED)  # This helps to reproduce the results

# Create an environment and start the setup process
env = simpy.Environment()

iposx = np.zeros((NUM_TARGET))
iposy = np.zeros((NUM_TARGET))
ivelx = np.zeros((NUM_TARGET))
ively = np.zeros((NUM_TARGET))
istate = np.zeros((NUM_TARGET))

# Target state vector definition - 
# keep both current target state and prior state for comparison
# Indices:
#  0: Target ID #
#  1: Target Movement ID# - 0=moving, 1=stationary
#  2: Target detection strength
#  3: Target x position
#  4: Target y position
#  Note: while the defined position array is 800x800, 
#        only the central 600x600 is valid for locating targets (i.e. is the search areas)
#        This allows targets to enter search area without having to be specially initialized.
target_info_current = np.zeros((NUM_TARGET,5))
target_info_prior = np.zeros((NUM_TARGET,5))
#Initialize target ID #'s in both arrays - these will not change 
for i in range(NUM_TARGET) :
    target_info_current[i,0] = i
    target_info_prior[i,0] = i

# Randomly select initial values for targets
for i in range(NUM_TARGET) :
    iposx[i] = random.randint(0,GRID_SIZE)
    iposy[i] = random.randint(0,GRID_SIZE)
    ivelx[i] = random.randint(-VEL_MEAN - VEL_SIGMA, VEL_MEAN + VEL_SIGMA)
    ively[i] = random.randint(-VEL_MEAN - VEL_SIGMA, VEL_MEAN + VEL_SIGMA)
    istate[i] = random.randint(0,1)
    
targets = [Target(env, i, iposx[i], iposy[i], ivelx[i], ively[i], istate[i] )
            for i in range(NUM_TARGET)]


# Execute!

simtime = 0
targetname = np.zeros((NUM_TARGET),dtype='int')
targetmode = np.zeros((NUM_TARGET),dtype='int')
targetpos = np.zeros((NUM_TARGET,2),dtype='float')

while simtime < SIM_TIME :
  
    target_info_prior = target_info_current
    
    int_type = random.randint(0,10)
    
    if int_type < 8 :
        print("MTI mode chosen")
        done_in2 = 1
    elif int_type < 10 :
        print("SAR mode chosen")
        done_in2 = 4
    else :
        print("Self test mode chosen")
        done_in2 = 2
        
    simtime += done_in2

    env.run(until=simtime)
    print("At time = %d" % simtime)
    print([targets])
    
# Use printed target string to generate current target information
    tstring = str([targets])
    tstring2 = tstring.strip("[]")
    tsubstring = tstring2.split(",")
    tarray = np.array(tsubstring)
    
    for i in range(NUM_TARGET):
        targetname[i] = int(tarray[8*i+1])
        targetmode[i] = int(tarray[8*i+3])
        targetpos[i,0] = float(tarray[8*i+5])
        targetpos[i,1] = float(tarray[8*i+7])
 
# Interrogate targets to upate detections
    for i in range(NUM_TARGET) :
        if int_type < 2 :           #MTI Mode Lower left portion of grid
            if targetmode[i] == 0 and targetpos[i,0] >= GUARD_BAND and targetpos[i,0] <= MID_GRID and targetpos[i,1] >= GUARD_BAND and targetpos[i,1] <= MID_GRID:
                target_info_current[i,1] = 0
                target_info_current[i,3] = targetpos[i,0]
                target_info_current[i,4] = targetpos[i,1]
                if target_info_prior[i,1] == 1 :
                    target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                if target_info_current[i,2] < (MAX_STRENGTH - STRENGTH_INC) :
                    target_info_current[i,2] += STRENGTH_INC
                else :
                    target_info_current[i,2] = MAX_STRENGTH
                    
        elif int_type < 4 :         #MTI Mode Upper left portion of grid
            if targetmode[i] == 0 and targetpos[i,0] > MID_GRID and targetpos[i,0] <= (GRID_SIZE - GUARD_BAND) and targetpos[i,1] >= GUARD_BAND and targetpos[i,1] <= MID_GRID:
                target_info_current[i,1] = 0
                target_info_current[i,3] = targetpos[i,0]
                target_info_current[i,4] = targetpos[i,1]
                if target_info_prior[i,1] == 1 :
                    target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                if target_info_current[i,2] < (MAX_STRENGTH - STRENGTH_INC) :
                    target_info_current[i,2] += STRENGTH_INC
                else :
                    target_info_current[i,2] = MAX_STRENGTH
                    
        elif int_type < 6 :         #MTI Mode Lower right portion of grid
            if targetmode[i] == 0 and targetpos[i,0] >= GUARD_BAND and targetpos[i,0] <= MID_GRID and targetpos[i,1] > MID_GRID and targetpos[i,1] <= (GRID_SIZE - GUARD_BAND):
                target_info_current[i,1] = 0
                target_info_current[i,3] = targetpos[i,0]
                target_info_current[i,4] = targetpos[i,1]
                if target_info_prior[i,1] == 1 :
                    target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                if target_info_current[i,2] < (MAX_STRENGTH - STRENGTH_INC) :
                    target_info_current[i,2] += STRENGTH_INC
                else :
                    target_info_current[i,2] = MAX_STRENGTH
                    
        elif int_type < 8:          #MTI Mode Upper right portion of grid
            if targetmode[i] == 0 and targetpos[i,0] > MID_GRID and targetpos[i,0] <= (GRID_SIZE - GUARD_BAND) and targetpos[i,1] > MID_GRID and targetpos[i,1] <= (GRID_SIZE - GUARD_BAND):
                target_info_current[i,1] = 0
                target_info_current[i,3] = targetpos[i,0]
                target_info_current[i,4] = targetpos[i,1]
                if target_info_prior[i,1] == 1 :
                    target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                if target_info_current[i,2] < (MAX_STRENGTH - STRENGTH_INC) :
                    target_info_current[i,2] += STRENGTH_INC
                else :
                    target_info_current[i,2] = MAX_STRENGTH
                                                                                 
        elif int_type < 10 :
            if targetmode[i] == 1 and targetpos[i,0] >= GUARD_BAND and targetpos[i,0] <= (GRID_SIZE - GUARD_BAND) and targetpos[i,1] >= GUARD_BAND and targetpos[i,1] <= (GRID_SIZE - GUARD_BAND):
                target_info_current[i,1] = 1
                target_info_current[i,3] = targetpos[i,0]
                target_info_current[i,4] = targetpos[i,1]
                if target_info_prior[i,1] == 0 :
                    target_info_current[i,2] = 0      # if target was moving when last interrogated, reset target strength to 0
                if target_info_current[i,2] < (MAX_STRENGTH - STRENGTH_INC) :
                    target_info_current[i,2] += STRENGTH_INC
                else :
                    target_info_current[i,2] = MAX_STRENGTH
                    
# If a target not detected in this time step, apply coasting penalty
    for i in range(NUM_TARGET) :   
        if target_info_current[i,2] == target_info_prior[i,2] :
            if target_info_current[i,2] > COAST_PEN :
                target_info_current[i,2] -= COAST_PEN
            else :
                target_info_current[i,2] = 0
                 
print("All done!")
