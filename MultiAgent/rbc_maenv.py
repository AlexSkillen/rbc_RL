from pettingzoo import ParallelEnv
import gymnasium as gym
import numpy as np
import functools
import dedalus.public as d3
import logging


class DedalusRBC_Env(ParallelEnv):
    metadata = {'render_modes' : 'human'}
    
    Lx = np.pi
    Lz = 1.

    def __init__(
            self,
            nagents=10,
            actionsPerEp=256,
            actionDuration=1.5,
            magPenFactor=0.,
            DiscardTime=100,
            Ni=100,
            Nk=64,
            ObsNi=30,
            ObsNk=8,
            Ra=1e4,
            Pr=0.71,
            render_mode=None
    ):
        self.nagents=nagents
        self.actionsPerEp=actionsPerEp
        self.actionDuration=actionDuration
        self.magPenFactor=magPenFactor
        self.DiscardTime=DiscardTime
        self.Ni=Ni
        self.Nk=Nk
        self.ObsNi=ObsNi
        self.ObsNk=ObsNk
        self.Ra=Ra
        self.Pr=Pr

        self.possible_agents = ["agent_" + str(i) for i in range(self.nagents)]
        self.render_mode = render_mode
        self.agent_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.observation_spaces = {agent: self.observation_space( agent ) for agent in self.possible_agents[:]}
        self.action_spaces = {agent: self.action_space( agent ) for agent in self.possible_agents[:]}


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Box(-0.5, 1.8, shape=(self.ObsNk*self.ObsNi,))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Box(-1, 1, shape=(1,))

    def reset(self, seed=None, options={}):
        self.problem, self.solver, self.CFL, self.fp, self.g, self.x = self._D3_RBC_setup(np.random.randint(100000))
        
        self.agents = self.possible_agents[:]
        
        observations = {agent: self._extractObs( agent ) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions=None):
        if actions is not None:
            self._setBC(actions)

        for _ in range(int(self.actionDuration / self.timestep)):
            self.solver.step(self.timestep)

        observations = {agent: self._extractObs( agent ) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        rewards = self._computeReward( actions ) 
        truncations = {agent: not self.solver.proceed for agent in self.agents}
        terminations = {agent: False for agent in self.agents}

        if all(truncations.values()):
            self.agents = []
            
        return observations, rewards, terminations, truncations, infos

    def render(self):
        print( self.fp.max('Nu') )  
    
    def _computeReward(self, actions):
        #extract as array
        action = np.array(list(actions.values()))
        r = np.zeros_like(action)
        r -= (np.average(self.fp.properties['Nu']['g'].flatten()[-5:]) - self.Nu0)
        r -= self.magPenFactor*np.absolute(action)

        r_dict = {agent: float(r[i]) for i, agent in enumerate(self.agents)}
        return r_dict
        
    def _setBC(self, actions):
        #extract as array
        action = np.array(list(actions.values()))
        #lower BC
        Tp = action - np.mean(action)

        Tp /= max(1., np.abs(Tp).max()/0.75)

        Tp = np.repeat(Tp, 3)
        
        #copy last action due to periodicity
        Tp = np.append(Tp, Tp[0])
        xp=np.linspace(0, self.Lx, len(Tp))

        T = np.interp(self.x, xp, Tp)
       
        self.g['g'] = T
        self.g['g'] += 1.0


    def _extractObs(self, agent):
        ID = self.agent_mapping[agent]
        agent_x = self.Lx/self.nagents*(0.5+ID)
        
        x=np.linspace(0, self.Lx, self.ObsNi+1)
        z=np.linspace(0.1, self.Lz-0.1, self.ObsNk)
        
        #remove the last x due to periodicity
        x=x[:-1]
        
        X, Z = np.meshgrid(x, z)
        obs = np.zeros_like(X, dtype=np.float32)

        for i in range(self.ObsNi):
            for k in range(self.ObsNk):
                xp = X[k][i] + agent_x
                if xp>self.Lx:
                    xp -= self.Lx
                obs[k][i] = np.squeeze(self.problem.variables[1](x=xp, z=Z[k][i]).evaluate()['g'])
       
        return obs.flatten()
    
    def _D3_RBC_setup(self, seed):
        logger = logging.getLogger(__name__)
        
        # Parameters
        Lx, Lz = self.Lx, self.Lz
        Nx, Nz = self.Ni, self.Nk
        Rayleigh = self.Ra
        Prandtl = self.Pr 

        dealias = 3/2
        stop_sim_time = self.actionsPerEp*self.actionDuration + self.DiscardTime
        timestepper = d3.RK222
        max_timestep = 0.125
        dtype = np.float64

        # Bases
        coords = d3.CartesianCoordinates('x', 'z')
        dist = d3.Distributor(coords, dtype=dtype)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)


        # Fields
        p = dist.Field(name='p', bases=(xbasis,zbasis))
        b = dist.Field(name='b', bases=(xbasis,zbasis))
        u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
        g = dist.Field(bases=xbasis)
        tau_p = dist.Field(name='tau_p')
        tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
        tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
        
        # Substitutions
        kappa = (Rayleigh * Prandtl)**(-1/2)
        nu = (Rayleigh / Prandtl)**(-1/2)
        x, z = dist.local_grids(xbasis, zbasis)
        ex, ez = coords.unit_vector_fields(dist)
        lift_basis = zbasis.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
        grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

        #bottom boundary
        g['g'] = 1.0
        
        # Problem
        # First-order form: "div(f)" becomes "trace(grad_f)":
        # First-order form: "lap(f)" becomes "div(grad_f)"
        problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
        problem.add_equation("b(z=0) = g")
        problem.add_equation("u(z=0) = 0")
        problem.add_equation("b(z=Lz) = 0")
        problem.add_equation("u(z=Lz) = 0")
        problem.add_equation("integ(p) = 0") # Pressure gauge

        # Solver
        solver = problem.build_solver(timestepper)
        solver.stop_sim_time = stop_sim_time

        fp = d3.GlobalFlowProperty(solver)
        fp.add_property((d3.Average((ez@u)*b, 'x') - kappa*d3.Differentiate(d3.Average(b, 'x'), coords[1]))/kappa, name='Nu')
            
        # Initial conditions
        b.fill_random('g', seed=seed, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls
        b['g'] += Lz - z # Add linear background


        # CFL
        CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
        CFL.add_velocity(u)

        while True:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if solver.sim_time >= self.DiscardTime:
                self.timestep = timestep
                break

        #get initial Nu for normalisation based on top of domain
        self.Nu0 = np.average(fp.properties['Nu']['g'].flatten()[-5:])
        
        return problem, solver, CFL, fp, g, x
