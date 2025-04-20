# File: /sgwb3d/sgwb3d/gw3d.py

import numpy as np
from scipy.special import jv, legendre, gamma, hyp2f1
import matplotlib.pyplot as plt

class GW3D:
    """
    Compute angular power spectra and correlation functions for gravitational waves
    in 3D space with different polarization modes.
    
    This implementation follows the formulas in:
    "Subluminal stochastic gravitational waves in pulsar-timing arrays and astrometry"
    """
    
    def __init__(self, l_max=20):
        """
        Initialize with maximum multipole moment to compute
        
        Parameters:
        -----------
        l_max : int
            Maximum multipole moment
        """
        self.l_max = l_max
        self.ell = np.arange(0, l_max+1)
        
    def I_l_n(self, ell, n, v):
        """
        Compute the integral I_l^(n)(v) defined in Eq. (A.3) of the paper
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        n : int
            Order of the integral
        v : float
            Group velocity (0 < v ≤ 1)
        
        Returns:
        --------
        complex
            Value of the integral
        """
        if v == 1:
            # Luminal case - Eq. (A.4)
            if n == 0:
                # This case diverges for SL mode
                return np.inf
            elif ell + 1 > n > 0:
                return (1j)**(ell+1-n) * 2**(n-1) * gamma(ell-n+1) / gamma(ell+n+1) * gamma(n)
            else:
                return 0
        elif v < 1:
            # Subluminal case - Eq. (A.5)
            if ell + 1 > n >= 0:
                factor = ((1j * v)/2)**(ell+1-n) * np.sqrt(np.pi) / (2**n)
                factor *= gamma(ell+1-n) / gamma(ell + 3/2)
                factor *= hyp2f1((ell+1-n)/2, (ell+2-n)/2, ell+3/2, v**2)
                return factor
            else:
                return 0
        else:
            raise ValueError("Group velocity must be 0 < v ≤ 1")
    
    def N_l(self, ell):
        """
        Compute the normalization factor N_l defined in the paper
        
        Parameters:
        -----------
        ell : int
            Multipole moment
            
        Returns:
        --------
        float
            Normalization factor
        """
        if ell >= 2:
            return np.sqrt((ell+2)*gamma(ell+3)/(2*gamma(ell-1)))
        else:
            return 0
    
    def F_z(self, ell, alpha, v):
        """
        Projection factor for PTA observable (Table 1 of the paper)
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        alpha : str
            Polarization mode ('ST', 'SL', 'VE', 'VB', 'TE', 'TB')
        v : float
            Group velocity (0 < v ≤ 1)
        
        Returns:
        --------
        complex
            Projection factor
        """
        if alpha == 'ST':
            # Scalar-Transverse mode
            if ell == 0:
                return -1/(2*np.sqrt(2)) * (1/v**2)
            elif ell == 1:
                return -1/(2*np.sqrt(2)) * (1j/(3*v))
            else:
                return -1/(2*np.sqrt(2)) * (1j * (1-v**2)/(v**3)) * self.I_l_n(ell, 0, v)
                
        elif alpha == 'SL':
            # Scalar-Longitudinal mode
            if ell == 0:
                return 1/2 * (1/v**2)
            elif ell == 1:
                return 1/2 * (1j/(3*v))
            else:
                return 1/2 * (1j/v**3) * self.I_l_n(ell, 0, v)
                
        elif alpha == 'VE':
            # Vector-E mode
            if ell == 1:
                return -(1j/(3*v))
            elif ell >= 2:
                return (1/v**2) * np.sqrt(ell*(ell+1)/2) * self.I_l_n(ell, 1, v)
            else:
                return 0
                
        elif alpha == 'TE':
            # Tensor-E mode
            if ell >= 2:
                N_l = self.N_l(ell)
                return (1j/(2*v)) * N_l * self.I_l_n(ell, 2, v)
            else:
                return 0
                
        elif alpha in ['VB', 'TB']:
            # B-modes don't contribute to z
            return 0
            
        else:
            raise ValueError(f"Unknown polarization mode: {alpha}")
    
    def F_E(self, ell, alpha, v):
        """
        Projection factor for astrometry E-mode (Table 1 of the paper)
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        alpha : str
            Polarization mode ('ST', 'SL', 'VE', 'VB', 'TE', 'TB')
        v : float
            Group velocity (0 < v ≤ 1)
        
        Returns:
        --------
        complex
            Projection factor
        """
        if alpha == 'ST':
            if ell == 1:
                return 1j/(6*v)
            elif ell >= 2:
                return -(1-v**2)/(2*v**2) * np.sqrt(ell*(ell+1)/2) * self.I_l_n(ell, 1, v)
            else:
                return 0
                
        elif alpha == 'SL':
            if ell == 1:
                return -1j/(3*np.sqrt(2)*v)
            elif ell >= 2:
                return (1/(2*v**2)) * np.sqrt(ell*(ell+1)) * self.I_l_n(ell, 1, v)
            else:
                return 0
                
        elif alpha == 'VE':
            if ell == 1:
                return 2j/(3*np.sqrt(2)*v)
            elif ell >= 2:
                return (-1/(np.sqrt(2)*v**2)) * self.I_l_n(ell, 1, v) + (1j*(1-v**2)/(np.sqrt(2)*v**3)) * self.I_l_n(ell, 0, v)
            else:
                return 0
                
        elif alpha == 'TE':
            if ell >= 2:
                N_l = self.N_l(ell)
                return -N_l/np.sqrt(ell*(ell+1)) * ((1j/v) * self.I_l_n(ell, 2, v) + ((1-v**2)/(2*v**2)) * self.I_l_n(ell, 1, v))
            else:
                return 0
                
        elif alpha == 'VB' or alpha == 'TB':
            # These don't contribute to E-mode
            return 0
            
        else:
            raise ValueError(f"Unknown polarization mode: {alpha}")
    
    def F_B(self, ell, alpha, v):
        """
        Projection factor for astrometry B-mode (Table 1 of the paper)
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        alpha : str
            Polarization mode ('ST', 'SL', 'VE', 'VB', 'TE', 'TB')
        v : float
            Group velocity (0 < v ≤ 1)
        
        Returns:
        --------
        complex
            Projection factor
        """
        if alpha == 'VB':
            if ell == 1:
                return 1j/(3*np.sqrt(2))
            elif ell >= 2:
                return -1/(np.sqrt(2)*v) * self.I_l_n(ell, 1, v)
            else:
                return 0
                
        elif alpha == 'TB':
            if ell >= 2:
                N_l = self.N_l(ell)
                return -1j*N_l/np.sqrt(ell*(ell+1)) * self.I_l_n(ell, 2, v)
            else:
                return 0
                
        elif alpha in ['ST', 'SL', 'VE', 'TE']:
            # These don't contribute to B-mode
            return 0
            
        else:
            raise ValueError(f"Unknown polarization mode: {alpha}")
    
    def C_l(self, ell, X, X_prime, alpha, v):
        """
        Compute power spectrum C_l^{XX'} for observables X, X'
        
        Parameters:
        -----------
        ell : int or array
            Multipole moment(s)
        X : str
            First observable ('z', 'E', 'B')
        X_prime : str
            Second observable ('z', 'E', 'B')
        alpha : str
            Polarization mode ('ST', 'SL', 'VE', 'VB', 'TE', 'TB')
        v : float
            Group velocity (0 < v ≤ 1)
        
        Returns:
        --------
        float or array
            Power spectrum value(s)
        """
        if np.isscalar(ell):
            if X == 'z':
                F_X = self.F_z(ell, alpha, v)
            elif X == 'E':
                F_X = self.F_E(ell, alpha, v)
            elif X == 'B':
                F_X = self.F_B(ell, alpha, v)
            else:
                raise ValueError(f"Unknown observable: {X}")
                
            if X_prime == 'z':
                F_X_prime = self.F_z(ell, alpha, v)
            elif X_prime == 'E':
                F_X_prime = self.F_E(ell, alpha, v)
            elif X_prime == 'B':
                F_X_prime = self.F_B(ell, alpha, v)
            else:
                raise ValueError(f"Unknown observable: {X_prime}")
                
            return 32 * np.pi**2 * np.abs(F_X * np.conjugate(F_X_prime))
        else:
            return np.array([self.C_l(l, X, X_prime, alpha, v) for l in ell])
    
    def normalize_spectrum(self, C_l, ell_norm=2):
        """
        Normalize spectrum by the value at ell_norm
        
        Parameters:
        -----------
        C_l : array
            Power spectrum
        ell_norm : int
            Multipole to normalize by (default: 2 for quadrupole)
        
        Returns:
        --------
        array
            Normalized power spectrum
        """
        if C_l[ell_norm] == 0:
            # If the normalization value is zero, find the first non-zero value
            for i in range(len(C_l)):
                if C_l[i] != 0:
                    return C_l / C_l[i]
            return C_l  # All zeros
        else:
            return C_l / C_l[ell_norm]
    
    def correlation_function(self, alpha, v, X, X_prime, angles=None):
        """
        Compute correlation function C(θ) from power spectrum
        
        Parameters:
        -----------
        alpha : str
            Polarization mode ('ST', 'SL', 'VE', 'VB', 'TE', 'TB')
        v : float
            Group velocity (0 < v ≤ 1)
        X : str
            First observable ('z', 'E', 'B')
        X_prime : str
            Second observable ('z', 'E', 'B')
        angles : array, optional
            Angular separations in degrees
            
        Returns:
        --------
        angles : array
            Angular separations in degrees
        corr : array
            Correlation function values
        """
        if angles is None:
            angles = np.linspace(0, 180, 181)
        
        # Get power spectrum
        C_l_values = self.C_l(self.ell, X, X_prime, alpha, v)
        
        # Normalize by quadrupole
        C_l_values = self.normalize_spectrum(C_l_values)
        
        # Calculate correlation function
        corr = np.zeros_like(angles, dtype=float)
        cos_theta = np.cos(np.radians(angles))
        
        for i, ct in enumerate(cos_theta):
            for l, cl in enumerate(C_l_values):
                if cl != 0:
                    leg = legendre(l)
                    corr[i] += (2*l + 1) / (4*np.pi) * cl * leg(ct)
        
        return angles, corr

def hellings_downs(theta):
    """
    Compute the Hellings-Downs curve
    
    Parameters:
    -----------
    theta : array
        Angular separations in degrees
        
    Returns:
    --------
    array
        Hellings-Downs correlation values
    """
    x = np.cos(np.radians(theta))
    
    # Initialize result array
    result = np.zeros_like(x, dtype=float)
    
    # Special case for θ=0 (self-correlation)
    mask_zero = np.isclose(x, 1.0)
    result[mask_zero] = 0.5
    
    # For all other angles, use the Hellings-Downs formula
    mask_nonzero = ~mask_zero
    if np.any(mask_nonzero):
        x_nonzero = x[mask_nonzero]
        result[mask_nonzero] = 0.5 * (1 + x_nonzero)
        result[mask_nonzero] += 0.5 * (1 - x_nonzero) * np.log((1 - x_nonzero) / 2)
    
    return result