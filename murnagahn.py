#!/usr/bin/env python
import numpy as np
from scipy.optimize import curve_fit


class fitMurnagahn:
    '''
        General class for fitting a Murnagahn curve.
        :methods:
            fitVignet - fits the Vignet equation of state to the data
            printParameters - outputs the parameters to screen
    '''
    def __init__(self, volume=[], energy=[], B0guess=190., B0pguess=3.):
        '''
        :params:
            volume - array of volumes in Angstrom^3
            energy - array of energies in eV
            B0guess - initial guess for B0 in GPa
            B0pguess - initial guess for B0'
        :return:
            self.B0 - bulk modulus in GPa
            self.V0 - equilibrium volume in Angtrom^3
            self.E0 - minimum energy in eV
            self.B0p - fitted B0'
        '''
        self.volume = volume
        self.energy = energy

        self.B0 = []
        self.V0 = []
        self.E0 = []
        self.B0p = []

        self.volumeAnaly = []
        self.energyAnaly = []

        # conversion factor
        self.eVA3toGpa = (1.6E-19)*(1.E30)*(1.E-9)

        # initial guess
        self.B0guess = B0guess / self.eVA3toGpa
        self.B0pguess = B0pguess

        return

    def fit_Vignet(self):
        '''
            Fits the Vinet EOS from the data
        '''

        def eqVignet(volume, B0, V0, E0, B0p):
            '''
                Vignet EOS
            '''
            eta = (volume/V0)**(1./3.)
            energy = E0 + (2.*B0*V0)/(B0p-1)**2. * \
                (2. - (5. + 3.*B0p*(eta-1.)-3.*eta) *
                 np.exp(-3.*(B0p-1.)*(eta-1.)/2.))
            return energy

        # define fitting variables
        varx = self.volume
        vary = self.energy

        # define initial guesses
        idmin = np.nonzero(vary == np.min(vary))[0]
        B0guess = self.B0guess
        V0guess = varx[idmin[0]]
        E0guess = vary[idmin[0]]
        B0pguess = self.B0pguess

        init_guess = np.array([B0guess, V0guess, E0guess, B0pguess])
        eqVignet(varx, B0guess, V0guess, E0guess, B0pguess)

        # fit Vignet
        params = curve_fit(eqVignet, varx, vary, init_guess)
        self.B0 = params[0][0] * self.eVA3toGpa
        self.V0 = params[0][1]
        self.E0 = params[0][2]
        self.B0p = params[0][3]

        self.volumeAnaly = np.linspace(varx.min(), varx.max(), 1001)
        self.energyAnaly = eqVignet(self.volumeAnaly, params[0][0],
                                    params[0][1], params[0][2], params[0][3])

        return params

    def get_energy(self, volume):
        ''' Returns the energy given the volume'''
        B0 = self.B0 / self.eVA3toGpa
        eta = (volume/self.V0)**(1./3.)
        energy = self.E0 + (2.*B0*self.V0)/(self.B0p-1)**2. * \
            (2. - (5. + 3.*self.B0p*(eta-1.)-3.*eta) *
             np.exp(-3.*(self.B0p-1.)*(eta-1.)/2.))

        return energy

    def print_parameters(self):
        '''
            Prints out fitted parameters
        '''
        print('Fitted parameters')
        print('B0  = %10.6f GPa' % (self.B0))
        print('V0  = %10.6f A^3' % (self.V0))
        print('E0  = %10.6f eV ' % (self.E0))
        print('B0p = %10.6f    ' % (self.B0p))

        return
