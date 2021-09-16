import numpy as np
from scipy.signal import cont2discrete
import casadi as ca



class Variables:

    '''
    var: Symbolic variables
    ub: Upper bounds for symbolic variables
    lb: Lower bounds for symbolic variables
    add_inequality(constraints, ub, lb): Adds symbolic variables and its bounds.
    add_equality(constraints): Adds symbolic variables. Bounds of the variables are set to be 0
    '''

    def __init__(self, name=None, var=None, ub=None, lb=None):

        self.name = name

        if type(var)==type(None):
            self.var = ca.SX.zeros(0)
            self.ub = np.array([])
            self.lb = np.array([])
        else:
            self.var = var
            if type(ub)==type(None):
                self.ub = float('inf')*np.ones(var.shape)
            else:
                self.ub = ub
            if type(lb)==type(None):
                self.lb = -float('inf')*np.ones(var.shape)
            else:
                self.lb = lb


    def __add__(self, other):

        temp = Variables(self.name + '+' + other.name)
        temp.var = ca.vertcat(self.var, other.var)
        temp.ub = np.concatenate([self.ub, other.ub])
        temp.lb = np.concatenate([self.lb, other.lb])

        return temp


    def __getitem__(self, key):

        return {
            'param':self.var[key],
            'ub':self.ub[key],
            'lb':self.lb[key]
        }


    def add_inequality(self, constraints, ub=None, lb=None):

        self.var = ca.vertcat(self.var, constraints)
        if type(ub)==type(None):
            self.ub = np.concatenate([self.ub, float('inf')*np.ones([constraints.shape[0]])])
        else:
            self.ub = np.concatenate([self.ub, ub])
        if type(lb)==type(None):
            self.lb = np.concatenate([self.lb, -float('inf')*np.ones([constraints.shape[0]])])
        else:
            self.lb = np.concatenate([self.lb, lb])


    def add_equality(self, constraints):

        self.var = ca.vertcat(self.var, constraints)
        self.ub = np.concatenate([self.ub, np.zeros([constraints.shape[0]])])
        self.lb = np.concatenate([self.lb, np.zeros([constraints.shape[0]])])



class MPC:

    def __init__(self, state_space, Q, R, N=10, Ts=0.1):
        #[z dz phi dphi theta dtheta z_LF dz_LF z_RF dz_RF z_LR dz_LR z_RR dz_RR]
        # 1 is front, 2 is rear
        self.n_states = state_space.n_states
        self.n_controls = state_space.n_controls
        self.N = N # PredictionHorizon
        self.Ts = Ts # Step time

        sysd = cont2discrete(
            (
                state_space.A,
                state_space.B,
                np.eye(state_space.n_states),
                np.zeros_like(state_space.B)
            ),
            self.Ts
        )

        Ad = ca.reshape(sysd[0], self.n_states, self.n_states)
        Bd = ca.reshape(sysd[1], self.n_states, self.n_controls)
        Q = ca.reshape(Q, self.n_states, self.n_states)
        R = ca.reshape(R, self.n_controls, self.n_controls)

        P = ca.SX.sym('P', self.n_states)
        X = Variables('X')
        U = Variables('U')
        g = Variables('g')

        obj = 0 # Objective function
        x = ca.SX.sym('x0', self.n_states)
        X.add_inequality(x) # initial state
        g.add_equality(x-P[0:self.n_states]) # initial condition constraints
        for k in range(self.N):
            u = ca.SX.sym('u'+str(k), self.n_controls)
            U.add_inequality(u)
            delta_fl_dot =  state_space.W/2*x[3]*ca.cos(x[2]) - state_space.L/2*x[5]*ca.cos(x[4]) + x[1] - x[7]
            delta_fr_dot = -state_space.W/2*x[3]*ca.cos(x[2]) - state_space.L/2*x[5]*ca.cos(x[4]) + x[1] - x[9]
            delta_rl_dot =  state_space.W/2*x[3]*ca.cos(x[2]) + state_space.L/2*x[5]*ca.cos(x[4]) + x[1] - x[11]
            delta_rr_dot = -state_space.W/2*x[3]*ca.cos(x[2]) + state_space.L/2*x[5]*ca.cos(x[4]) + x[1] - x[13]
            dynamic_yield_force = ca.vertcat(
                ca.sign(delta_fl_dot) * (u[0] - state_space.C_min * delta_fl_dot),
                ca.sign(delta_fr_dot) * (u[1] - state_space.C_min * delta_fr_dot),
                ca.sign(delta_rl_dot) * (u[2] - state_space.C_min * delta_rl_dot),
                ca.sign(delta_rr_dot) * (u[3] - state_space.C_min * delta_rr_dot),
            )
            g.add_inequality(dynamic_yield_force, ub=300*np.ones((self.n_controls)), lb=np.zeros((self.n_controls)))
            obj += ca.mtimes(x.T, ca.mtimes(Q, x))
            obj += ca.mtimes(u.T, ca.mtimes(R, u))
            x_next = ca.SX.sym('x'+str(k+1), self.n_states)
            X.add_inequality(x_next)
            x_predict = ca.mtimes(Ad, x) + ca.mtimes(Bd, u)
            g.add_equality(x_next - x_predict)
            x = x_next


        OPT_variables = X + U

        qp = {'f':obj, 'x':OPT_variables.var, 'g':g.var, 'p':P}
        self.solver = ca.nlpsol('solver', 'ipopt', qp)

        self.args = {'lbg':g.lb,'ubg':g.ub,'lbx':OPT_variables.lb,'ubx':OPT_variables.ub} 
        self.u0 = np.zeros((self.N * self.n_controls, 1))

    def solve(self, x_init):
        self.args['p'] = ca.reshape(x_init, self.n_states, 1)
        self.args['x0'] = ca.vertcat(
            ca.repmat(np.reshape(x_init, (self.n_states, 1)), self.N+1, 1),
            self.u0

        )
        sol = self.solver(
            x0  = self.args['x0'],
            lbx = self.args['lbx'],
            ubx = self.args['ubx'],
            lbg = self.args['lbg'],
            ubg = self.args['ubg'],
            p   = self.args['p']
        )
        u = np.array(
            sol['x'][
                self.n_states * (self.N+1):self.n_states * (self.N+1) + self.n_controls
            ]
        )

        return u
