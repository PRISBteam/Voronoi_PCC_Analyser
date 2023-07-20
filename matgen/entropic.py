"""
Software module for the Entropic analysis paper
"""
import math
from typing import Iterable
import logging
import numpy as np

class TripleJunctionSet:
    """
    structure representation
    """
    def __init__(self, p, j_tuple) -> None:
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
        self.S_s = self.get_S_s()
        self.kappa = self.S_m / self.S_s if self.S_s != 0 else 0
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

    def get_S_s(self):
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
        j_array = np.array(self.j_tuple_random)

        nonzeros = j_array[j_array > 0]
        return - np.sum(nonzeros * np.log2(nonzeros))

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
