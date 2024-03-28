import gymnasium as gym
import numpy as np
import dedalus.public as d3
import logging

class DedalusRBC_Env(gym.Env):
    metadata = {'render_modes' : 'human'}
    obs_metadata = {"Ni" : 30, "Nk" : 8 }
    sim_metadata = {"Lx" : np.pi, "Lz" : 1.0, "Ra" : 1e4, "Pr" : 0.71, "Ni" : 100, "Nk" : 64, "DiscardTime" : 80}
    act_metadata = {"actionDuration" : 1.5, "actionsPerEp" : 256, "magPenFactor" : 0.1}
    
    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(-0.5, 1.5, shape=(self.obs_metadata['Nk']*self.obs_metadata['Ni'],))
        self.action_space = gym.spaces.Box(-1, 1, shape=(10,))
        self.render_mode = render_mode

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self._D3_init()
        obs = self._extractObs()
        info = {}

        return obs, info

    def step(self, action=None):
        if action is not None:
            self._setBC(action)

        for _ in range(int(self.act_metadata['actionDuration'] / self.timestep)):
            self.solver.step(self.timestep)

        obs = self._extractObs()
        info = {}
        reward = self._computeReward( action )
        term = not self.solver.proceed
        trun = term

        return obs, reward, term, trun, info

    def _computeReward(self, action):
        mainReward = -(np.average(self.fp.properties['Nu']['g'].flatten()[-5:]) - self.Nu0)
        mag_pen    = -self.act_metadata['magPenFactor']*np.average(np.absolute(action))
        return mainReward + mag_pen

    def _extractObs(self):
        x=np.linspace(0, self.sim_metadata['Lx'], self.obs_metadata['Ni']+1)
        z=np.linspace(0.1, self.sim_metadata['Lz']-0.1, self.obs_metadata['Nk'])
        
        #remove the last x due to periodicity
        x=x[:-1]
        
        X, Z = np.meshgrid(x, z)
        obs = np.zeros_like(X, dtype=np.float32)

        for i in range(self.obs_metadata['Ni']):
            for k in range(self.obs_metadata['Nk']):
                obs[k][i] = np.squeeze(self.problem.variables[1](x=X[k][i], z=Z[k][i]).evaluate()['g'])

        return obs.flatten()

    def _setBC(self, action):
        #lower BC
        Tp = action - np.mean(action)

        Tp /= max(1., np.abs(Tp).max()/0.75)

        Tp = np.repeat(Tp, 3)
        
        #copy last action due to periodicity
        Tp = np.append(Tp, Tp[0])
        xp=np.linspace(0, self.sim_metadata['Lx'], len(Tp))

        T = np.interp(self.x, xp, Tp)
       
        self.g['g'] = T
        self.g['g'] += 1.0
    
    def _D3_init(self):
        self.problem, self.solver, self.CFL, self.fp, self.g, self.x = self._D3_RBC_setup(np.random.randint(100000))

        while True:
            self.timestep = self.CFL.compute_timestep()
            self.solver.step(self.timestep)
            if self.solver.sim_time >= self.sim_metadata['DiscardTime']:
                break

        #get initial Nu for normalisation based on top of domain
        self.Nu0 = np.average(self.fp.properties['Nu']['g'].flatten()[-5:])
        
    
    def _D3_RBC_setup(self, seed):
        logger = logging.getLogger(__name__)
        
        # Parameters
        Lx, Lz = self.sim_metadata['Lx'], self.sim_metadata['Lz']
        Nx, Nz = self.sim_metadata['Ni'], self.sim_metadata['Nk']
        Rayleigh = self.sim_metadata['Ra']
        Prandtl = self.sim_metadata['Pr'] 
        dealias = 3/2
        stop_sim_time = self.act_metadata['actionsPerEp']*self.act_metadata['actionDuration']+self.sim_metadata['DiscardTime']
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


        return problem, solver, CFL, fp, g, x

    def render(self):
        print( self.fp.max('Nu') )
        


