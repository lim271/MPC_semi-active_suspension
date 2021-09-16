import numpy as np
from fullcar_state_space import FullcarStateSpace
from mpc import MPC

if __name__=='__main__':
    
    ss = FullcarStateSpace()

    Q11 = 40*10000*np.eye(6)
    Q12 = np.zeros((6, ss.n_states-6))
    Q21 = np.zeros((ss.n_states-6, 6))
    Q22 = np.zeros((ss.n_states-6, ss.n_states-6))
    Q = np.block(
        [
            [Q11, Q12],
            [Q21, Q22]
        ]
    )
    R = 0.01*np.eye(ss.n_controls)

    mpc = MPC(ss, Q, R, N=10, Ts=0.1)

    u = mpc.solve(np.random.randn(14))
    print(u)