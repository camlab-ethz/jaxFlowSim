class Blood:
    mu: float
    rho: float
    rho_inv: float
    def __init__(self, mu, rho, rho_inv):
        self.mu = mu
        self.rho = rho
        self.rho_inv = rho_inv
