import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynHouseholdLaborModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.beta = 0.98 # discount factor

        par.rho_01 = 0.05 # weight on labor dis-utility of men
        par.rho_02 = 0.05 # weight on labor dis-utility of women
        par.rho_11 = 0.1
        par.rho_12 = 0.1

        par.eta = -1.5 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # kids
        par.p_birth = 0.1

        # income
        par.wage_const_1 = np.log(10_000.0) # constant, men
        par.wage_const_2 = np.log(10_000.0) # constant, women
        par.wage_K_1 = 0.1 # return on human capital, men
        par.wage_K_2 = 0.1 # return on human capital, women

        par.delta = 0.1 # depreciation in human capital

        # taxes
        par.tax_scale = 2.1   #2.278029 # from Borella et al. (2023), singles: 1.765038
        par.tax_pow = 0.0861765 # from Borella et al. (2023), singles: 0.0646416

        # child transfers
        par.uncon_uni = 0.0
        par.means_level = 0.0
        par.means_slope = 0.0
        par.cond = 0.0
        par.cond_high = 0.0

        # grids        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 #30 # number of grid points in wealth grid
        par.Nn = 2    

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        # reform
        par.joint_tax = True


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # kids grid
        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Nk,par.Nk)
        sol.h1 = np.nan + np.zeros(shape)
        sol.h2 = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.h1 = np.nan + np.zeros(shape)
        sim.h2 = np.nan + np.zeros(shape)
        sim.k1 = np.nan + np.zeros(shape)
        sim.k2 = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)
        
        sim.c = np.nan + np.zeros(shape)
        sim.income1 = np.nan + np.zeros(shape)
        sim.income2 = np.nan + np.zeros(shape)
        sim.income_hh = np.nan + np.zeros(shape)
        sim.tax_hh = np.nan + np.zeros(shape)
        sim.child_tran = np.nan + np.zeros(shape)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.k1_init = np.zeros(par.simN)
        sim.k2_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: human capital for each household member
            for i_n, kids in enumerate(par.n_grid):
                for i_k1,capital1 in enumerate(par.k_grid):
                    for i_k2,capital2 in enumerate(par.k_grid):
                        idx = (t,i_n,i_k1,i_k2)
                        
                        # ii. find optimal consumption and hours at this level of wealth in this period t.
                        if t==(par.T-1): # last period
                            obj = lambda x: -self.util(x[0],x[1],capital1,capital2,kids)

                        else:
                            obj = lambda x: - self.value_of_choice(x[0],x[1],capital1,capital2,kids,t)  

                        # call optimizer
                        bounds = [(0,np.inf) for i in range(2)]
                        
                        init_h = np.array([0.1,0.1])
                        if i_k1>0: 
                            init_h[0] = sol.h1[t,i_n,i_k1-1,i_k2]
                        if i_k2>0: 
                            init_h[1] = sol.h2[t,i_n,i_k1,i_k2-1]

                        res = minimize(obj,init_h,bounds=bounds) 

                        # store results
                        sol.h1[idx] = res.x[0]
                        sol.h2[idx] = res.x[1]
                        sol.V[idx] = -res.fun
 

    def value_of_choice(self,hours1,hours2,capital1,capital2,kids,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. current utility
        util = self.util(hours1,hours2,capital1,capital2,kids)
        
        # c. continuation value
        k1_next = (1.0-par.delta)*capital1 + hours1
        k2_next = (1.0-par.delta)*capital2 + hours2

        # no birth
        kids_next = kids
        V_next = sol.V[t+1,kids_next]
        V_next_no_birth = interp_2d(par.k_grid,par.k_grid,V_next,k1_next,k2_next)

        # birth
        if (kids>=(par.Nn-1)):
            # cannot have more children
            V_next_birth = V_next_no_birth

        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,kids_next]
            V_next_birth = interp_2d(par.k_grid,par.k_grid,V_next,k1_next,k2_next)

        EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth

        # d. return value of choice
        return util + par.beta*EV_next


    # relevant functions
    def consumption(self,hours1,hours2,capital1,capital2,kids):
        par = self.par

        income1 = self.wage_func(capital1,1) * hours1
        income2 = self.wage_func(capital2,2) * hours2
        income_hh = income1+income2

        child_tran = self.child_tran(hours1,hours2,income_hh,kids)

        if par.joint_tax:
            tax_hh = self.tax_func(income_hh,child_tran)
        else:
            tax_hh = self.tax_func(income1,child_tran/2) + self.tax_func(income2,child_tran/2)
        

        return income_hh - tax_hh + child_tran

    def wage_func(self,capital,sex):
        # before tax wage rate
        par = self.par

        constant = par.wage_const_1
        return_K = par.wage_K_1
        if sex>1:
            constant = par.wage_const_2
            return_K = par.wage_K_2

        return np.exp(constant + return_K * capital)

    def tax_func(self,income,child_tran):
        par = self.par

        rate = 1.0 - par.tax_scale*((income+child_tran)**(-par.tax_pow))
        return rate*(income+child_tran)
    
    def child_tran(self,hours1,hours2,income_hh,kids):
        par = self.par
        if kids<1:
            return 0.0
        else:
            C1 = par.uncon_uni
            C2 = np.fmax(par.means_level - par.means_slope*income_hh,0.0)
            both_work = (hours1>0) * (hours2>0)
            C3 = par.cond*both_work
            C4 = par.cond_high*both_work*(income_hh>0.5)
        
        return C1 + C2 + C3 + C4

    def util(self,hours1,hours2,capital1,capital2,kids):
        par = self.par

        cons = self.consumption(hours1,hours2,capital1,capital2,kids)

        util_cons = 2*(cons/2)**(1.0+par.eta) / (1.0+par.eta)
        util_hours1 = (par.rho_01 + par.rho_11*kids)*(hours1)**(1.0+par.gamma) / (1.0+par.gamma)
        util_hours2 = (par.rho_02 + par.rho_12*kids)*(hours2)**(1.0+par.gamma) / (1.0+par.gamma)

        return util_cons - util_hours1 - util_hours2

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.k1[i,0] = sim.k1_init[i]
            sim.k2[i,0] = sim.k2_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal hours
                idx_sol = (t,sim.n[i,t])
                sim.h1[i,t] = interp_2d(par.k_grid,par.k_grid,sol.h1[idx_sol],sim.k1[i,t],sim.k2[i,t])
                sim.h2[i,t] = interp_2d(par.k_grid,par.k_grid,sol.h2[idx_sol],sim.k1[i,t],sim.k2[i,t])

                # store income
                sim.income1[i,t] = self.wage_func(sim.k1[i,t],1)*sim.h1[i,t]
                sim.income2[i,t] = self.wage_func(sim.k2[i,t],2)*sim.h2[i,t]

                sim.income_hh[i,t] = sim.income1[i,t] + sim.income2[i,t]
                sim.child_tran[i,t] = self.child_tran(sim.h1[i,t],sim.h2[i,t],sim.income_hh[i,t],sim.n[i,t])
                
                if par.joint_tax:
                    sim.tax_hh[i,t] = self.tax_func(sim.income_hh[i,t],sim.child_tran[i,t])
                else:
                    sim.tax_hh[i,t] = self.tax_func(sim.income1[i,t],sim.child_tran[i,t]/2) + self.tax_func(sim.income2[i,t],sim.child_tran[i,t]/2)

                sim.c[i,t] = sim.income_hh[i,t] + sim.child_tran[i,t] - sim.tax_hh[i,t]

                # iii. store next-period states
                if t<par.simT-1:
                    sim.k1[i,t+1] = (1.0-par.delta)*sim.k1[i,t] + sim.h1[i,t]
                    sim.k2[i,t+1] = (1.0-par.delta)*sim.k2[i,t] + sim.h2[i,t]
                    
                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    


