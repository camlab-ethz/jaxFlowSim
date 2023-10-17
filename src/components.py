import jax.numpy as jnp
import numpy as np
from jax import tree_util

#@jax.dataclass
class Heart:
    inlet_type: str
    cardiac_T: float
    input_data: jnp.ndarray
    inlet_number: int
    def __init__(self, inlet_type, cardiac_T, input_data, inlet_number):
        self.inlet_type = inlet_type
        self.cardiac_T = cardiac_T
        self.input_data = input_data
        self.inlet_number = inlet_number
    
    def _tree_flatten(self):
        children = ()
        aux_data = {'inlet_type': self.inlet_type, 'cardiac_T': self.cardiac_T, 'input_data': self.input_data, 'inlet_number': self.inlet_number} 
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(Heart,
                               Heart._tree_flatten,
                               Heart._tree_unflatten)

#@jax.dataclass
class Blood:
    mu: float
    rho: float
    rho_inv: float
    def __init__(self, mu, rho, rho_inv):
        self.mu = mu
        self.rho = rho
        self.rho_inv = rho_inv
    
    def _tree_flatten(self):
        children = ()
        aux_data = {'mu': self.mu, 'rho': self.rho, 'rho_inv': self.rho_inv} 
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(Blood,
                               Blood._tree_flatten,
                               Blood._tree_unflatten)

#@jax.dataclass
class Vessel:
    A: jnp.ndarray
    Q: jnp.ndarray
    u: jnp.ndarray
    c: jnp.ndarray
    P: jnp.ndarray
    A_t: jnp.ndarray
    Q_t: jnp.ndarray
    u_t: jnp.ndarray
    c_t: jnp.ndarray
    P_t: jnp.ndarray
    A_l: jnp.ndarray
    Q_l: jnp.ndarray
    u_l: jnp.ndarray
    c_l: jnp.ndarray
    P_l: jnp.ndarray
    W1M0: float
    W2M0: float
    U00A: float
    U00Q: float
    U01A: float
    U01Q: float
    UM1A: float
    UM1Q: float
    UM2A: float
    UM2Q: float
    last_P_name: str
    last_Q_name: str
    last_A_name: str
    last_c_name: str
    last_u_name: str
    out_P_name: str
    out_Q_name: str
    out_A_name: str
    out_c_name: str
    out_u_name: str
    node2: int
    node3: int
    node4: int
    Rt: float
    R1: float
    R2: float
    Cc: float
    dU: jnp.ndarray
    Pc: float
    outlet: str

    def __init__(self, A, Q, u, c, P, 
                 #A_t, Q_t, u_t, c_t, P_t, 
                 #A_l, Q_l, u_l, c_l, P_l, 
                 P_t, P_l, 
                 W1M0, W2M0, U00A, 
                 U00Q, U01A, U01Q, UM1A, UM1Q, UM2A, UM2Q):
        self.A = A
        self.Q = Q
        self.u = u
        self.c = c
        self.P = P
        #self.A_t = A_t
        #self.Q_t = Q_t
        #self.u_t = u_t
        #self.c_t = c_t
        self.P_t = P_t
        #self.A_l = A_l
        #self.Q_l = Q_l
        #self.u_l = u_l
        #self.c_l = c_l
        self.P_l = P_l
        self.W1M0 = W1M0
        self.W2M0 = W2M0
        self.U00A = U00A
        self.U00Q = U00Q
        self.U01A = U01A
        self.U01Q = U01Q
        self.UM1A = UM1A
        self.UM1Q = UM1Q
        self.UM2A = UM2A
        self.UM2Q = UM2Q
    
    def _tree_flatten(self):
        children = (self.A, self.Q, self.u, self.c, self.P, 
                    self.P_t, self.P_l, 
                    #self.A_t, self.Q_t, self.u_t, self.c_t, self.P_t, 
                    #self.A_l, self.Q_l, self.u_l, self.c_l, self.P_l, 
                    self.W1M0, self.W2M0, self.U00A, self.U00Q, self.U01A, 
                    self.U01Q, self.UM1A, self.UM1Q, self.UM2A, self.UM2Q, 
                   )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

class Vessel_const:
    label: str
    heart: Heart
    M: int
    wallVa: np.ndarray
    wallVb: np.ndarray
    last_P_name: str
    last_Q_name: str
    last_A_name: str
    last_c_name: str
    last_u_name: str
    out_P_name: str
    out_Q_name: str
    out_A_name: str
    out_c_name: str
    out_u_name: str
    node2: int
    node3: int
    node4: int
    Pc: float

    def __init__(self, label, heart, M,
                 wallVa, wallVb, last_P_name, 
                 last_Q_name, last_A_name, last_c_name, last_u_name, 
                 out_P_name, out_Q_name, out_A_name, out_c_name, 
                 out_u_name, node2, node3, node4):
        self.label = label
        self.heart = heart
        self.M = M
        self.wallVa = wallVa
        self.wallVb = wallVb
        self.last_P_name = last_P_name
        self.last_Q_name = last_Q_name
        self.last_A_name = last_A_name
        self.last_c_name = last_c_name
        self.last_u_name = last_u_name
        self.out_P_name = out_P_name
        self.out_Q_name = out_Q_name
        self.out_A_name = out_A_name
        self.out_c_name = out_c_name
        self.out_u_name = out_u_name
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
    
    #def _tree_flatten(self):
    #    children = (self.A, self.Q, self.u, self.c, self.P, self.A_t, 
    #                self.Q_t, self.u_t, self.c_t, self.P_t, self.A_l, 
    #                self.Q_l, self.u_l, self.c_l, self.P_l, self.W1M0, 
    #                self.W2M0, self.U00A, self.U00Q, self.U01A, 
    #                self.U01Q, self.UM1A, self.UM1Q, self.UM2A, self.UM2Q, 
    #                self.dU,
    #               self.Pc,)
    #    aux_data = {'label': self.label, 'ID': self.ID, 
    #                'sn': self.sn, 'tn': self.tn, 'inlet': self.inlet, 
    #                'heart': self.heart, 'M': self.M, 'dx': self.dx, 
    #                'invDx': self.invDx, 'halfDx': self.halfDx, 'beta': self.beta, 
    #                'gamma': self.gamma, 's_15_gamma': self.s_15_gamma, 
    #                'gamma_ghost': self.gamma_ghost, 'A0': self.A0, 
    #                's_A0': self.s_A0, 'inv_A0': self.inv_A0, 's_inv_A0': self.s_inv_A0, 
    #                'Pext': self.Pext, 'viscT': self.viscT, 'wallE': self.wallE, 
    #                'wallVa': self.wallVa, 'wallVb': self.wallVb, 'last_P_name': self.last_P_name, 
    #                'last_Q_name': self.last_Q_name, 'last_A_name': self.last_A_name, 
    #                'last_c_name': self.last_c_name, 'last_u_name': self.last_u_name, 
    #                'out_P_name': self.out_P_name, 'out_Q_name': self.out_Q_name, 
    #                'out_A_name': self.out_A_name, 'out_c_name': self.out_c_name, 
    #                'out_u_name': self.out_u_name, 'node2': self.node2, 'node3': self.node3, 
    #                'node4': self.node4, 'Rt': self.Rt, 'R1': self.R1, 'R2': self.R2, 
    #                'Cc': self.Cc, 'outlet': self.outlet}
    #    return (children, aux_data)

tree_util.register_pytree_node(Vessel,
                               Vessel._tree_flatten,
                               Vessel._tree_unflatten)

class Edges:
    edges: np.ndarray
    inlets: np.ndarray
    outlets: np.ndarray
    
    def __init__(self,edges,inlets,outlets):
        self.edges = edges
        self.inlets = inlets
        self.outlets = outlets

    #def __hash__(self):
    #    return hash(np.array2string(self.edges))

    #def __eq__(self,other):
    #    return (self.__hash__() == other.__hash__())