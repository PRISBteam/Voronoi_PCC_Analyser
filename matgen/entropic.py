"""
Software module for the Entropic analysis paper
"""
import math
import logging
import numpy as np

class EntropicMixin:
    """
    """
    def __repr__(self) -> str:
        """
        """
        return self.__str__()
    
    def get_triv_entropy(self, p):
        """
        theoretical trivial entropy
        """
        if p == 0 or p == 1:
            return 0
        q = 1 - p
        return - (p * math.log2(p) + q * math.log2(q))
    
    def get_entropy(self, tuple_):
        """
        config entropy
        """
        array_ = np.array(tuple_)
        nonzeros = array_[array_ > 0]
        return - np.sum(nonzeros * np.log2(nonzeros))
    
    def get_mean_part(self, tuple_):
        """
        the mean part of the config entropy
        """
        array_ = np.array(tuple_)
        if np.any(array_ == 0):
            return np.inf        
        return - np.log2(np.prod(array_)) / len(array_)

    def get_deviat_part(self, tuple_):
        """
        the deviatoric part of the config entropy
        """
        array_ = np.array(tuple_)
        if np.any(array_ == 0):
            return - np.inf
        Ss = 0
        for k in range(len(array_)):
            jk = array_[k]
            for l in range(k + 1, len(array_)):
                jl = array_[l]
                Ss += (jk - jl) * math.log2(jk / jl)
        Ss = Ss / len(array_)
        return - Ss

    def get_metastability(self, Smax, Smin):
        """
        """
        Srand = self.S_rand
        S = self.S
        if S >= Srand and Smax != Srand:
            return (S - Srand) / (Smax - Srand)
        elif S < Srand and Smin != Srand:
            return (S - Srand) / (Srand - Smin)
    
    def get_properties(self, attr_list: list = []) -> dict:
        """
        attr_list - custom list of properties to return 
        """
        if not attr_list:
            return self.__dict__

        try:
            return {attr_name: getattr(self, attr_name) for attr_name in attr_list}
        except:
            logging.exception('Check properties!')


class TripleJunctionSet(EntropicMixin):
    """
    structure representation
    """
    def __init__(self, p, j_tuple):
        """
        p - empiric SGBs fraction
        j_tuple - TJ fractions j0, j1, j2, j3
        """
        if p > 1 or p < 0:
            logging.warning('SGBs fraction is not valid (must be < 1 and > 0)')
        self.p = p
        
        if not math.isclose(sum(j_tuple), 1):
            logging.warning('TJs fraction sum is not equal to 1')
        self.j0, self.j1, self.j2, self.j3 = j_tuple

        self.q = 1 - p
        
        self.Sp = self.get_Sp()
        self.Sp_m = self.get_Sp_m()
        self.Sp_s = self.get_Sp_s()
        
        self.p_expected = (self.j1 + 2*self.j2 + 3*self.j3) / 3
        self.delta_p = abs(self.p_expected - self.p)

        self.S = self.get_S()
        self.S_m = self.get_S_m()
        self.S_d = self.get_S_d()
        self.kappa = self.S_m / self.S_d if self.S_d != 0 else 0
        self.delta_S = self.Sp - self.S

        self.d1 = self.j1 / (1 - self.j0) if self.j0 != 1 else 0
        self.d2 = self.j2 / (1 - self.j0) if self.j0 != 1 else 0
        self.d3 = self.j3 / (1 - self.j0) if self.j0 != 1 else 0

        self.sigma = self.get_sigma()
        self.chi = self.get_chi()

    def __str__(self):
        cell_str = (self.__class__.__name__ + 
                    f"(p={self.p}, j0={self.j0}, j1={self.j1}, " +
                    f"j2={self.j2}, j3={self.j3})")
        return cell_str
    
    def __repr__(self) -> str:
        """
        """
        return self.__str__()

    @property
    def j_tuple(self):
        """
        """
        return (self.j0, self.j1, self.j2, self.j3)
    
    @property
    def j_tuple_random(self):
        """
        """
        jr0 = (1 - self.p)**3
        jr1 = 3 * self.p * (1 - self.p)**2
        jr2 = 3 * self.p**2 * (1 - self.p)
        jr3 = self.p**3

        return (jr0, jr1, jr2, jr3)

    def get_Sp(self):
        """
        theoretical trivial entropy for a given p
        """
        if self.p == 0 or self.p == 1:
            return 0

        p = self.p
        q = 1 - self.p
        return - (p * math.log2(p) + q * math.log2(q))

    def get_Sp_m(self):
        """
        the mean part of theoretical entropy
        """
        if self.p == 0 or self.p == 1:
            return np.inf
        return - math.log2(self.p * (1 - self.p)) / 2

    def get_Sp_s(self):
        """
        the deviatoric part of theoretical entropy
        """
        if self.p == 0 or self.p == 1:
            return - np.inf
        
        p = self.p
        q = 1 - self.p
        return - (p - q) / 2 * math.log2(p / q)

    def get_S(self):
        """
        structural entropy
        """
        j_array = np.array(self.j_tuple)

        nonzeros = j_array[j_array > 0]
        return - np.sum(nonzeros * np.log2(nonzeros))

    def get_S_m(self):
        """
        the mean part of the structural entropy
        """
        j_array = np.array(self.j_tuple)
        if np.any(j_array == 0):
            return np.inf
        
        return - np.log2(np.prod(j_array)) / len(j_array)

    def get_S_d(self):
        """
        the skrew part of the structural entropy
        """
        j_array = np.array(self.j_tuple)
        if np.any(j_array == 0):
            return - np.inf

        Ss = 0
        for k in range(len(j_array)):
            jk = j_array[k]
            for l in range(k + 1, len(j_array)):
                jl = j_array[l]
                Ss += (jk - jl) * math.log2(jk / jl)
        Ss = Ss / len(j_array)
        return - Ss
    
    def get_sigma(self):
        """
        sigma differentiates states of segregation from ordering
        Notes
        -----
        See IV.A. Definition of correlation functions in 
        Frary and Schuh. Phys. Rev. E 76, 041108
        DOI: 10.1103/PhysRevE.76.041108
        """
        if self.p == 0 or self.p == 1: # J1r = J2r = 0
            return 0
        
        j_tuple = self.j_tuple
        j_tuple_r = self.j_tuple_random

        if j_tuple[0] == 0 and j_tuple[3] == 0:
            return -1
        j_sigma = (
            (j_tuple[1] + j_tuple[2]) / (j_tuple[0] + j_tuple[3]) *
            (j_tuple_r[0] + j_tuple_r[3]) / (j_tuple_r[1] + j_tuple_r[2])
        )

        if j_sigma <= 1:
            return (1 - j_sigma)
        else:
            return 1 / j_sigma - 1
    
    def get_chi(self):
        """
        differentiates the tendency for bonds to form either compact 
        or elongated clusters
        """
        if self.p == 0 or self.p == 1 or (self.j1 == 0 and self.j2 == 0):
            return 0
        
        if self.j1 == 0:
            return -1

        j_tuple_r = self.j_tuple_random
        j_chi = self.j2 / self.j1 * j_tuple_r[1] / j_tuple_r[2]

        if j_chi <= 1:
            return (1 - j_chi)
        else:
            return 1 / j_chi - 1

    @property
    def S_rand(self):
        """
        """
        # j_array = np.array(self.j_tuple_random)

        # nonzeros = j_array[j_array > 0]
        # return - np.sum(nonzeros * np.log2(nonzeros))
        return self.get_entropy(self.j_tuple_random)


class GrainGammaSet(EntropicMixin):
    """
    structure representation
    __dict__ doesn't show properties, but attributes
    """
    def __init__(self, g, gamma_tuple):
        if g > 1 or g < 0:
            logging.warning('g fraction is not valid (must be < 1 and > 0)')
        self.g = g

        if not math.isclose(sum(gamma_tuple), 1):
            logging.warning('TJs fraction sum is not equal to 1')
        self.gamma0, self.gamma1, self.gamma2 = gamma_tuple

        self.delta1 = self.gamma1/(1 - self.gamma0) if self.gamma0 != 1 else 0
        self.delta2 = self.gamma2/(1 - self.gamma0) if self.gamma0 != 1 else 0

        self.S_g = self.get_S_g()
        self.S = self.get_S()
        self.S_m = self.get_S_m()
        self.S_d = self.get_S_d()
        self.kappa = self.S_m / self.S_d if self.S_d != 0 else 0
        self.delta_S = self.S - self.S_rand

    def __str__(self):
        cell_str = (self.__class__.__name__ + 
                    f"(g={self.g}, gamma0={self.gamma0}, " +
                    f"gamma1={self.gamma1}, gamma2={self.gamma2})")
        return cell_str

    @property
    def gamma_tuple(self):
        """
        """
        return (self.gamma0, self.gamma1, self.gamma2)
    
    @property
    def gamma_tuple_random(self):
        """
        """
        gammar0 = (1 - self.g)**2
        gammar1 = 2 * self.g * (1 - self.g)
        gammar2 = self.g**2
        return (gammar0, gammar1, gammar2)

    def get_omega(self):
        """
        """
        if self.g == 0 or self.g == 1:
            w = 0 # TODO: check
        elif self.gamma1 <= self.gammar1:
            w = 1 - self.gamma1 / self.gammar1
        elif self.gamma1 > self.gammar1:
            w = self.gamma0 * self.gamma2 / self.gammar0 / self.gammar2 - 1
        return w

    def get_S_g(self):
        """
        theoretical trivial entropy for a given g
        """
        return self.get_triv_entropy(self.g)

    def get_S(self):
        """
        configurational entropy
        """
        return self.get_entropy(self.gamma_tuple)
    
    def get_S_m(self):
        """
        the mean part of the configurational entropy
        """
        return self.get_mean_part(self.gamma_tuple)

    def get_S_d(self):
        """
        the deviat part of the configurational entropy
        """
        return self.get_deviat_part(self.gamma_tuple)
    
    @property
    def S_rand(self):
        """
        """
        return self.get_entropy(self.gamma_tuple_random)