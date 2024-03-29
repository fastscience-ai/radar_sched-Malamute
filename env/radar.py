import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random
import numpy as np
import simpy
from scipy.stats import norm

class Target(object):
    """
    Define target class with all properties necessary for target to move or remain stationary in grid.
    A target either moves at a constant speed or sits stationary.
    A target has a *name*, a *Position*, a *velocity*, and a *state* (moving or stationary).
    """
    def __init__(self, sim, name, posx, posy, velx, vely, state):
        self.sim = sim
        self.name = name
        self.posx = posx
        self.posy = posy
        self.velx = velx
        self.vely = vely
        self.state = state       # 1 = moving, 0 = stationary        
        self.process = sim.process(self.move(name, posx, posy, velx, vely))
        sim.process(self.interrogate())
        self.MOV_MEAN = None
        self.MOV_SIGMA = None
        self.STA_MEAN = None
        self.STA_SIGMA = None

    def movetime(self):
        # Time that target spends moving - can choose between normal distribution
        # and uniform distribution
        # return random.normalvariate(self.MOV_MEAN, self.MOV_SIGMA)
        return random.uniform(self.MOV_MEAN - 3*self.MOV_SIGMA, self.MOV_MEAN + 3*self.MOV_SIGMA)

    def stattime(self):
        # Time that target spends stationary - same as above
        # return random.normalvariate(self.STA_MEAN, self.STA_SIGMA)
        return random.uniform(self.STA_MEAN - 3*self.STA_SIGMA, self.STA_MEAN + 3*self.STA_SIGMA)

    def time_to_interrogate(self):
        # Set interrogation time - without this, moving targets will not have
        # their locations update.
        # This is because simpy is interrupt-driven.  Without interrrupts processes
        # do not change the state of the targets until their move or stationary
        # time is finished.
        self.done_in2 = 1
        return self.done_in2

    def move(self, name, posx, posy, velx, vely):
        """Move from location posx posy with velx and vely until end of motion.
        
        velx and vely set by state (=assigned values when state=1, =0 when state=0)

        When motion ends, change state and repeat until simulation ends

        """
        while True:
            # Begin by setting time for current action given target state
            if self.state == 1:
                done_in = self.movetime()
            else:
                done_in = self.stattime()
            while done_in:
                try:
                    # Move the target
                    start = self.sim.now
                    if self.state == 1:
                        vx = velx
                        vy = vely
                    else:
                        vx = 0
                        vy = 0

                    # Yield current move when time done                    
                    yield self.sim.timeout(done_in)

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

                    print("Target %d done move at position %.2f %.2f time %.2f now in state %d." % (name, posx, posy, self.sim.now, self.state))
                    done_in = 0  # Set to 0 to exit while loop.

                except simpy.Interrupt:
                    # when interrupt received, update target position and
                    # then reset time remaining for current action based on
                    # interrupt time
                    posx = posx + (self.sim.now - start) * vx
                    posy = posy + (self.sim.now - start) * vy
                    print("Target state at interrupt time %.2f :" %self.sim.now)
                    print("Target %d state %d position %.2f %.2f" % (name, self.state, posx, posy))
                    done_in -= self.sim.now - start  # How much time left?
                    self.posx = posx
                    self.posy = posy

    def interrogate(self) :
        # Interrogate evolving targets to obtain their current state and position
        # Here is where the interrupt is sent
        while True:
            yield self.sim.timeout(self.time_to_interrogate())

            self.process.interrupt()

    def __repr__(self):
        # Define the outputs to be sent to the simulation for further processing
        return "Target,%d,state,%d,xpos,%.2f,ypos,%.2f" % (self.name, self.state, self.posx, self.posy)



class Radar(gym.Env):
    def __init__(self):
        # Constants needed for code
        # Values set for testing
        # More typical values given in commented lines
        self.RANDOM_SEED = 45
        self.MIN_TARGETS = 2         # Minimum number of targets
        self.MAX_TARGETS = 20        # Maximum number of targets
        # self.NUM_TARGETS = random.randint(self.MIN_TARGETS, self.MAX_TARGETS) # Actual number of targets if variable
        self.NUM_TARGETS = 2          # Number of targets if fixed, for initial training, could be 20
        self.MOV_MEAN = 10         # Avg. target moving time in minutes
        self.MOV_SIGMA = 2         # Sigma of target moving time
        self.STA_MEAN = 15         # Avg. target stationary time in minutes
        self.STA_SIGMA = 3         # Sigma of target stationary time
        self.VEL_MEAN = 2            # Avg. target moving velocity
        self.VEL_SIGMA = 1           # Sigma of target moving velocity
        self.GRID_SIZE = 800         # Size of grid on which targets move
        self.MID_GRID  = 400         # Center of grid
        self.GUARD_BAND = 100        # Size of guard band in which targets move but are not detected
        self.MAX_STRENGTH = 30       # Maximum strength of target (fixed for now)
        self.STRENGTH_INC = 10       # Strength increase for target which is detected (fixed for now)
        self.COAST_PEN = 1           # Penalty for coasting (i.e. not detecting) a target in a given time step
        self.PDFIFTY = 15            # Target strength for PD = 0.50
        self.PDSCALE = 5             # Scale factor for computing PD
        self.SIM_TIME = 50            # Length of simulation1
        self.target_info_current = np.ones(shape=(self.NUM_TARGETS, 7), dtype=np.float32)
        self.target_info_prior = np.ones(shape=(self.NUM_TARGETS, 7), dtype=np.float32)

        self.viewer = None
        self.done = False
        self.num_actions = 6

        # high = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1] ], dtype=np.int32) #shape (NUM_TARGET, 5)
        # low = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0] ], dtype=np.int32)
        # self.action_space = spaces.Discrete(11) #spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        high = np.ones_like(self.target_info_current)*np.Inf
        low = np.zeros_like(self.target_info_current)
        self.action_space = spaces.Discrete(self.num_actions) #spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # self.seed()

    def set_seed(self, seed=None):
        # This helps to reproduce the results
        random.seed(seed if seed else self.RANDOM_SEED)

    def reset(self, *, seed=None, options=None):
        # set the passed in seed
        self.set_seed(seed)

        # Create a simulation and start the setup process
        self.sim = simpy.Environment()

        # simtime keeps track of the simulation elapsed time
        self.simtime = 0
        iposx = np.zeros((self.NUM_TARGETS))
        iposy = np.zeros((self.NUM_TARGETS))
        ivelx = np.zeros((self.NUM_TARGETS))
        ively = np.zeros((self.NUM_TARGETS))
        istate = np.zeros((self.NUM_TARGETS))
        # Target state vector definition -
        # keep both current target state and prior state for comparison
        # Indices:
        #  0: Target ID #
        #  1: Target Movement ID# - 1=moving, 0=stationary
        #  2: Target detection strength
        #  3: Target x position
        #  4: Target y position
        #  5: Target currently located - 0=no, 1=yes
        #  6: Target PD
        #  Note: while the defined position array is 800x800,
        #        only the central 600x600 is valid for locating targets (i.e. is the search areas)
        #        This allows targets to enter search area without having to be specially initialized.
        self.target_info_current = np.zeros_like((self.target_info_current))
        self.target_info_prior = np.zeros_like(self.target_info_prior)
        #Initialize target ID #'s in both arrays - these will not change
        for i in range(self.NUM_TARGETS) :
            self.target_info_current[i,0] = i
            self.target_info_prior[i,0] = i

        # Randomly select initial values for targets
        # Position, velocity, and state
        for i in range(self.NUM_TARGETS) :
            iposx[i] = random.randint(0,self.GRID_SIZE)
            iposy[i] = random.randint(0,self.GRID_SIZE)
            ivelx[i] = random.randint(self.VEL_MEAN - self.VEL_SIGMA, self.VEL_MEAN + self.VEL_SIGMA)
            ively[i] = random.randint(self.VEL_MEAN - self.VEL_SIGMA, self.VEL_MEAN + self.VEL_SIGMA)
            istate[i] = random.randint(0,1)

        # Initialize the targets
        self.targets = [Target(self.sim, i, iposx[i], iposy[i], ivelx[i], ively[i], istate[i] )
                        for i in range(self.NUM_TARGETS)]

        # Update the mov_mean, mov_sigma, sta_mean, sta_sigma for each target from the values here.
        for i in range(self.NUM_TARGETS):
            self.targets[i].MOV_MEAN = self.MOV_MEAN
            self.targets[i].MOV_SIGMA = self.MOV_SIGMA
            self.targets[i].STA_MEAN = self.STA_MEAN
            self.targets[i].STA_SIGMA = self.STA_SIGMA

        return self._get_obs()

    def _get_obs(self):
        return self.target_info_current

    def is_done(self):
        """
        Tells whether episode is done or not
        """
        return self.simtime >= self.SIM_TIME

    def step(self, action):
        # Step the simulation
        print("action\n\n\n\n\n", action)
        assert action <= 5, "Unknown action"
        # the following three arrays keep track of target parameters extracted from
        # interrogating the targets
        targetname = np.zeros((self.NUM_TARGETS), dtype='int')
        targetmode = np.zeros((self.NUM_TARGETS), dtype='int')
        targetpos = np.zeros((self.NUM_TARGETS, 2), dtype='float')

        # update the prior information array
        self.target_info_prior = self.target_info_current
        self.target_info_current[:, 5] = 0  # set all targets as not detected in current interrogation
        # For this simulation, pick actions for agent at random,
        # in RL system, training will focus on picking actions to maximize
        # information gain
        if action < 4 : # 0,1,2,3
            print("MTI mode chosen")
            done_in2 = 1
        elif action == 4 :
            print("SAR mode chosen")
            done_in2 = 4
        else : # action==5
            print("Self test mode chosen")
            done_in2 = 2
        # set time for simulation to be interrogated
        self.simtime += done_in2
        #SOO
        #try:
        #    self.sim.run(until=simtime)
        #except:
        #    simtime+=1
        self.sim.run(until=self.simtime)
        print("At time = %d" % self.simtime)
        # print out target information string defined in class target
        print([self.targets])
        # Use printed target string to generate current target information
        tstring = str([self.targets])
        tstring2 = tstring.strip("[]")
        tsubstring = tstring2.split(",")
        tarray = np.array(tsubstring)
        for i in range(self.NUM_TARGETS):
            targetname[i] = int(tarray[8*i+1])
            targetmode[i] = int(tarray[8*i+3])
            targetpos[i,0] = float(tarray[8*i+5])
            targetpos[i,1] = float(tarray[8*i+7])
        # Interrogate targets to upate detections
        for i in range(self.NUM_TARGETS) :
            if action == 0:           #MTI Mode Lower left portion of grid
                if targetmode[i] == 1 and targetpos[i,0] >= self.GUARD_BAND and targetpos[i,0] <= self.MID_GRID and targetpos[i,1] >= self.GUARD_BAND and targetpos[i,1] <= self.MID_GRID:
                    self.target_info_current[i,1] = 1
                    self.target_info_current[i,3] = targetpos[i,0]
                    self.target_info_current[i,4] = targetpos[i,1]
                    if self.target_info_prior[i,1] == 0 :
                        self.target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                    if self.target_info_current[i,2] < (self.MAX_STRENGTH - self.STRENGTH_INC) :
                        self.target_info_current[i,2] += self.STRENGTH_INC
                    else :
                        self.target_info_current[i,2] = self.MAX_STRENGTH
                    self.target_info_current[i,5] = 1 #mark this target as currently detected

            elif action == 1 :         #MTI Mode Upper left portion of grid
                if targetmode[i] == 1 and targetpos[i,0] > self.MID_GRID and targetpos[i,0] <= (self.GRID_SIZE - self.GUARD_BAND) and targetpos[i,1] >= self.GUARD_BAND and targetpos[i,1] <= self.MID_GRID:
                    self.target_info_current[i,1] = 1
                    self.target_info_current[i,3] = targetpos[i,0]
                    self.target_info_current[i,4] = targetpos[i,1]
                    if self.target_info_prior[i,1] == 0 :
                        self.target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                    if self.target_info_current[i,2] < (self.MAX_STRENGTH - self.STRENGTH_INC) :
                        self.target_info_current[i,2] += self.STRENGTH_INC
                    else :
                        self.target_info_current[i,2] = self.MAX_STRENGTH
                    self.target_info_current[i,5] = 1 #mark this target as currently detected

            elif action == 2 :         #MTI Mode Lower right portion of grid
                if targetmode[i] == 1 and targetpos[i,0] >= self.GUARD_BAND and targetpos[i,0] <= self.MID_GRID and targetpos[i,1] > self.MID_GRID and targetpos[i,1] <= (self.GRID_SIZE - self.GUARD_BAND):
                    self.target_info_current[i,1] = 1
                    self.target_info_current[i,3] = targetpos[i,0]
                    self.target_info_current[i,4] = targetpos[i,1]
                    if self.target_info_prior[i,1] == 0 :
                        self.target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                    if self.target_info_current[i,2] < (self.MAX_STRENGTH - self.STRENGTH_INC) :
                        self.target_info_current[i,2] += self.STRENGTH_INC
                    else :
                        self.target_info_current[i,2] = self.MAX_STRENGTH
                    self.target_info_current[i,5] = 1 #mark this target as currently detected

            elif action == 3:          #MTI Mode Upper right portion of grid
                if targetmode[i] == 1 and targetpos[i,0] > self.MID_GRID and targetpos[i,0] <= (self.GRID_SIZE - self.GUARD_BAND) and targetpos[i,1] > self.MID_GRID and targetpos[i,1] <= (self.GRID_SIZE - self.GUARD_BAND):
                    self.target_info_current[i,1] = 1
                    self.target_info_current[i,3] = targetpos[i,0]
                    self.target_info_current[i,4] = targetpos[i,1]
                    if self.target_info_prior[i,1] == 0 :
                        self.target_info_current[i,2] = 0      # if target was stationary when last interrogated, reset target strength to 0
                    if self.target_info_current[i,2] < (self.MAX_STRENGTH - self.STRENGTH_INC) :
                        self.target_info_current[i,2] += self.STRENGTH_INC
                    else :
                        self.target_info_current[i,2] = self.MAX_STRENGTH
                    self.target_info_current[i,5] = 1 #mark this target as currently detected

            elif action == 4:         # SAR Mode, applies to entire grid
                if targetmode[i] == 0 and targetpos[i,0] >= self.GUARD_BAND and targetpos[i,0] <= (self.GRID_SIZE - self.GUARD_BAND) and targetpos[i,1] >= self.GUARD_BAND and targetpos[i,1] <= (self.GRID_SIZE - self.GUARD_BAND):
                    self.target_info_current[i,1] = 0
                    self.target_info_current[i,3] = targetpos[i,0]
                    self.target_info_current[i,4] = targetpos[i,1]
                    if self.target_info_prior[i,1] == 1 :
                        self.target_info_current[i,2] = 0      # if target was moving when last interrogated, reset target strength to 0
                    if self.target_info_current[i,2] < (self.MAX_STRENGTH - self.STRENGTH_INC) :
                        self.target_info_current[i,2] += self.STRENGTH_INC
                    else :
                        self.target_info_current[i,2] = self.MAX_STRENGTH
                    self.target_info_current[i,5] = 1 #mark this target as currently detected

        for i in range(self.NUM_TARGETS) :
            # If a target not detected in this time step, apply coasting penalty
            if self.target_info_current[i,5] == 0 :
                if self.target_info_current[i,2] > self.COAST_PEN :
                    self.target_info_current[i,2] -= self.COAST_PEN
                else :
                    self.target_info_current[i,2] = 0
            # Calculate PD based on current detection strength and formula using erfc
            arg = (self.PDFIFTY - self.target_info_current[i,2])/self.PDSCALE
            self.target_info_current[i,6] = norm.sf(arg)

        # reward = sum(self.target_info_current[:,2]-self.target_info_prior[:,2])
        reward = sum(self.target_info_current[:,2] >= self.MAX_STRENGTH/2)
        # reward = sum(self.target_info_current[:,6]) # maximize prob. of detect (PD) of all targets.
        return self._get_obs(), reward, self.is_done(), {}

    def render(self, mode='human'):
        pass
        """
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(600,600)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
       
        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

