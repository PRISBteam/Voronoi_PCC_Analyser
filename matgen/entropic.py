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
        self.j_tuple = j_tuple
        self.j0, self.j1, self.j2, self.j3 = j_tuple

    @property
    def q(self):
        """
        q = 1 - p - empiric OGBs fraction
        """
        return 1 - self.p
    
    @property
    def Sp(self):
        """
        theoretical trivial entropy for a given p
        """
        if self.p == 0 or self.p == 1:
            return 0

        p = self.p
        q = 1 - self.p
        return - (p * math.log2(p) + q * math.log2(q))
    
    @property
    def Sp_m(self):
        """
        the mean part of theoretical entropy
        """
        if self.p == 0 or self.p == 1:
            return np.inf
        return - math.log2(self.p * (1 - self.p)) / 2

    @property
    def Sp_s(self):
        """
        the deviatoric part of theoretical entropy
        """
        if self.p == 0 or self.p == 1:
            return -np.inf
        
        p = self.p
        q = 1 - self.p
        return - (p - q) / 2 * math.log2(p / q)

    @property
    def p_expected(self):
        """
        """
        return (self.j1 + 2*self.j2 + 3*self.j3) / 3
        
    @property
    def delta_p(self):
        """
        """
        return abs(self.p_expected - self.p)

    @property
    def S(self):
        """
        """
        return matutils.entropy(*self.j_tuple)

    @property
    def S_m(self):
        """
        """
        if np.any(np.array(self.j_tuple) == 0):
            return np.inf
        else:
            return matutils.entropy_m(*self.j_tuple)

    @property
    def S_s(self):
        """
        """
        if np.any(np.array(self.j_tuple) == 0):
            return - np.inf
        else:
            return matutils.entropy_s(*self.j_tuple)

    @property
    def kappa(self):
        """
        """
        if self.S_s == 0:
            return 0
        else:
            return self.S_m / self.S_s

    @property
    def delta_S(self):
        """
        """
        return self.Sp - self.S

    @property
    def d_tuple(self):
        """
        """
        return matutils.get_d_tuple(self.j_tuple)

    @property
    def d1(self):
        """
        """
        return self.d_tuple[0]

    @property
    def d2(self):
        """
        """
        return self.d_tuple[1]

    @property
    def d3(self):
        """
        """
        return self.d_tuple[2]

    def get_property(self, attr):
        """
        """
        return getattr(self, attr)
    
    def get_properties(self, attr_list: list = []) -> Dict:
        """
        """
        if not attr_list:
            attr_list = [
                'p',
                'q',
                'Sp',
                'Sp_m',
                'Sp_s',
                'S',
                'S_m',
                'S_s',
                'kappa',
                'delta_S',
                'd1',
                'd2',
                'd3'
            ]

        try:
            return {attr_name: getattr(self, attr_name) for attr_name in attr_list}
        except:
            logging.exception('Check properties!')

        # values = [getattr(self, attr_name) for attr_name in attr_list]
        # return pd.DataFrame([values], columns = attr_list)


def entropy(*args):
    """
    S
    """
    # input arguments may be a scalar, a tuple or several scalars
    if len(args) == 1 and isinstance(args[0], Iterable):
        j_array = np.array(args[0])
    else:
        j_array = np.array(args)

    # check sum of input parameters
    if len(j_array) > 1 and not math.isclose(j_array.sum(), 1):
        logging.warning('Sum is not equal to 1')

    # calculate entropy
    if len(j_array) == 1:
        p = j_array[0]
        if p == 0 or p == 1:
            return 0
        elif p > 0 and p < 1:
            return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
    elif len(j_array) > 1:
        nonzeros = j_array[j_array > 0]
        return - np.sum(nonzeros * np.log2(nonzeros))
    
def entropy_m(*args):
    """
    mean part of S
    """
    # input arguments may be a scalar, a tuple or several scalars
    if len(args) == 1 and isinstance(args[0], Iterable):
        j_array = np.array(args[0])
    else:
        j_array = np.array(args)

    # check sum of input parameters
    if len(j_array) > 1 and not math.isclose(j_array.sum(), 1):
        logging.warning('Sum is not equal to 1')

    # check zero elements
    if np.any(j_array == 0):
        logging.warning('One or more j is equal to 0')
        return np.inf

    # calculate mean entropy
    if len(j_array) == 1:
        p = j_array[0]
        if p == 1:
            return np.inf
        elif p > 0 and p < 1:
            return - math.log2(p * (1 - p)) / 2
    elif len(j_array) > 1:
        return - np.log2(np.prod(j_array)) / len(j_array)
    
def entropy_s(*args):
    """
    deviatoric part of S
    """
    # input arguments may be a scalar, a tuple or several scalars
    if len(args) == 1 and isinstance(args[0], Iterable):
        j_array = np.array(args[0])
    else:
        j_array = np.array(args)

    # check sum of input parameters
    if len(j_array) > 1 and not math.isclose(j_array.sum(), 1):
        logging.warning('Sum is not equal to 1')

    # check zero elements
    if np.any(j_array == 0):
        logging.warning('One or more j is equal to 0')
        return - np.inf
    
    # calculate deviatoric entropy
    if len(j_array) == 1:
        p = j_array[0]
        if p == 1:
            return - np.inf
        elif p > 0 and p < 1:
            q = 1 - p
            return - (p - q) / 2 * math.log2(p / q)
    elif len(j_array) > 1:
        Ss = 0
        for k in range(len(j_array)):
            jk = j_array[k]
            for l in range(k + 1, len(j_array)):
                jl = j_array[l]
                Ss += (jk - jl) * math.log2(jk / jl)
        Ss = Ss / len(j_array)
        return - Ss