import numpy as np
import pandas as pd
from collections import namedtuple

import crystaltools.vasprun_parser as vp
import crystaltools.fetch_tools as ft
import crystaltools.crystal_generator as cg
import crystaltools.murnagahn as murn

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')


def fxn_aggregator(file_path):
    """
        Function aggregator to extract additional information
        from hdf5 file defined by file_path.  This aggregator
        function is meant for calculations with 1 solute.

        :params:
            file_path - file path of hdf5 file

        :return:
            pd.Series containing:
            - a0: lattice paramter
            - dnn: distance of the nearest neighbors
            - Nnn: number of nearest neighbors
            - ismear: ISMEAR parameter from INCAR
            - sigma: SIGMA parameter from INCAR
            - magmom: boolean stating whether the MAGMOM parameter
                      exists in the INCAR or not

    """
    with vp.VasprunHDFParser(directory='', filename=file_path) as vasprun:
        cellvec = vasprun.get_cell_vectors()
        element = vasprun.get_atoms()['atomtype']
        positions = vasprun.get_positions(state='final', coords='relative')
        rrel = np.array(positions[['rx', 'ry', 'rz']])

        # construct crystal
        crys = cg.Crystal()
        crys.cellvec = cellvec
        crys.itype = np.array(element).astype('int')
        crys.rrel = rrel

        # calculate lattice parameter
        if len(crys.xcar[crys.itype == 1]) in [4*(n)**(3.)
                                               for n in xrange(10)]:
            nsc = (len(crys.xcar[crys.itype == 1])/4)**(1./3.)
        elif len(crys.xcar[crys.itype == 1]) in [2*(n)**(3.)
                                                 for n in xrange(10)]:
            nsc = (len(crys.xcar[crys.itype == 1])/2)**(1./3.)
        a0 = np.linalg.norm(cellvec[:, 0]) / nsc

        # center cell on the interstitial
        r0 = crys.rcar[crys.itype == 2]
        crys.translate_cartesian(r0)
        crys.center_atoms()

        # calculate nearest neighbor distances
        eps = 0.00001
        mask = (crys.itype == 1)
        dist = (crys.xcar[mask]**2. + crys.ycar[mask]**2. +
                crys.zcar[mask]**2.)**(1./2.)/a0
        dnn = min(dist)
        Nnn = len(dist[dist <= dnn+eps])

        # get ISMEAR and SIGMA parameters
        incar = vasprun.get_incar()
        incar = vasprun.get_incar()
        if 'ISMEAR' in incar.keys():
            ismear = incar.ISMEAR
        else:
            ismear = np.nan

        if 'SIGMA' in incar.keys():
            sigma = incar.SIGMA
        else:
            sigma = np.nan

        if 'MAGMOM' in incar.keys():
            magmom = True
        else:
            magmom = False

        # construct pandas series
        return pd.Series([file_path, a0, dnn, Nnn, ismear, sigma, magmom],
                         index=['path', 'a0', 'dnn', 'Nnn',
                                'ismear', 'sigma', 'magmom'])


def fxn_aggregator_pure(file_path):
    """
        Function aggregator to extract additional information
        from hdf5 file defined by file_path.  This aggregator
        function is meant for calculations containing only 1
        element.

        :params:
            file_path - file path of hdf5 file

        :return:
            pd.Series containing:
            - a0: lattice paramter
            - dnn: distance of the nearest neighbors
            - Nnn: number of nearest neighbors
            - ismear: ISMEAR parameter from INCAR
            - sigma: SIGMA parameter from INCAR
            - magmom: boolean stating whether the MAGMOM parameter
                      exists in the INCAR or not

    """
    with vp.VasprunHDFParser(directory='', filename=file_path) as vasprun:
        cellvec = vasprun.get_cell_vectors()
        element = vasprun.get_atoms()['atomtype']
        positions = vasprun.get_positions(state='final', coords='relative')
        rrel = np.array(positions[['rx', 'ry', 'rz']])

        # construct crystal
        crys = cg.Crystal()
        crys.cellvec = cellvec
        crys.itype = np.array(element).astype('int')
        crys.rrel = rrel

        # calculate lattice parameter
        nsc = (len(crys.xcar[crys.itype == 1])/4)**(1./3.)
        a0 = np.linalg.norm(cellvec[:, 0]) / nsc

        # get ISMEAR and SIGMA parameters
        incar = vasprun.get_incar()
        incar = vasprun.get_incar()
        if 'ISMEAR' in incar.keys():
            ismear = incar.ISMEAR
        else:
            ismear = np.nan

        if 'SIGMA' in incar.keys():
            sigma = incar.SIGMA
        else:
            sigma = np.nan

        if 'MAGMOM' in incar.keys():
            magmom = True
        else:
            magmom = False

        # construct pandas series
        return pd.Series([file_path, a0, ismear, sigma, magmom],
                         index=['path', 'a0', 'ismear', 'sigma', 'magmom'])


def fit_vignet(df, mask, ax=[], plot=True):
    """
        Fits Vignet parameters from data in pandas dataframe df.
        The dataframe must contain columns ['volume', e_fr_energy].

        :params:
            df - pandas dataframe containing the volume-energy data
            mask - mask applied to df

        :return:
            pd.Series containing:
                E0, V0, B0, B0p - Vignet parameters
    """
    Color = namedtuple('Color', ['red', 'blue', 'orange',
                                 'cyan', 'green', 'yellow'])
    sol = Color(red='#dc322f', blue='#268bd2', orange='#cb4b16',
                cyan='#2aa198', green='#859900', yellow='#b58900')

    vignet = murn.fitMurnagahn()
    volume = df[mask].volume.as_matrix()
    energy = df[mask].e_fr_energy.as_matrix()
    vignet = murn.fitMurnagahn(volume, energy)
    vignet.fit_Vignet()

    if plot:
        vanaly = np.linspace(volume.min(), volume.max(), 1001)
        Eanaly = vignet.get_energy(vanaly)

        ax.plot(volume, energy, 'ok', markerfacecolor=sol.blue, zorder=2)
        ax.plot(vanaly, Eanaly, '--', color=sol.red, zorder=1)
        ax.set_xlabel('Volume (\AA$^3$)')
        ax.set_ylabel('Energy (eV)')

    return pd.Series([vignet.E0, vignet.V0, vignet.B0, vignet.B0p],
                     index=['E0', 'V0', 'B0', 'B0p'])


def get_vignet(df_all=[], matrix=[], solname=[], n1=[], n2=[],
               kx=[], isif=[], cubic=[], ismear=[], magnetic=[],
               plot=False, verbose=False):
    """
        Fit's Vignet parameters for subset of data in df_all defined
        by the paramters. See 'Vignet_parameter_fitting_example.ipynb'.

        :params:
            df_all - dataframe containg data
            matrix - matrix name
            solname - name of solute
            n1 - number of matrix atoms
            n2 - number of interstitial atoms
            kx - kx parameter
            isif - ISIF paramter
            cubic - boolean whether the caclulation is cubic or not
            ismear - ISMEAR parameter,
            magnetic - boolean whether the MAGMOM keyword is in INCAR
    """
    # grab subset of data for systems containing a solute
    dfsub = df_all[(df_all.el1 == matrix) & (df_all.el2 == solname) &
                   (df_all.n1 == n1) & (df_all.n2 == n2) & (df_all.kx == kx) &
                   (df_all.isif == isif)].copy()
    dfsub.reset_index(inplace=True)

    # grab subset of data for the pure matrix
    dfpure = df_all[(df_all.el1 == matrix) & (df_all.el2 == 'None') &
                    (df_all.n1 == n1) & (df_all.n2 == 0) & (df_all.kx == kx) &
                    (df_all.isif == isif) & (df_all.e_fr_energy < 0)].copy()
    dfpure.reset_index(inplace=True)

    # determine crystal type
    if (cubic) & (n1 in [4*(n)**3. for n in xrange(10)]):
        crystal_type = 'fcc'
    elif (cubic) & (n1 in [2*(n)**3. for n in xrange(10)]):
        crystal_type = 'bcc'
    elif cubic:
        crystal_type = 'cubic'
    else:
        crystal_type = 'non-cubic'

    if crystal_type == 'fcc':
        dict_Nnn = {'oct': 6, 'tet': 4}
    elif crystal_type == 'bcc':
        dict_Nnn = {'oct': 2, 'tet': 1}

    dfsub['crystal_type'] = crystal_type
    dfpure['crystal_type'] = crystal_type

    # Get additonal information from hdf5 data for cells with solutes
    if verbose:
        print('Getting additional data for calculations with solutes:')
    path_list = list(dfsub.path)
    dfnn, bad_path = ft.get_aggregate_data(fxn_aggregator,
                                           path_list, verbose=verbose)

    dfsub2 = pd.merge(dfsub, dfnn, how='inner', on='path')

    # Get additional information from hdf5 data for matrix
    if verbose:
        print('\nGetting additional data for the pure matrix:')
    path_list = list(dfpure.path)
    df_extra, bad_path = ft.get_aggregate_data(fxn_aggregator_pure,
                                               path_list, verbose=verbose)
    dfpure = pd.merge(dfpure, df_extra, how='inner', on='path')

    # create figure and axis for plots
    if plot:
        fig = plt.figure(figsize=(10, 3))
        fig.clf()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
    else:
        ax1 = []
        ax2 = []
        ax3 = []

    # Fit parameters for the octahedral site
    site_type = 'oct'
    mask = (dfsub2.Nnn == dict_Nnn[site_type]) & (dfsub2.ismear == ismear) & \
        (dfsub2.magmom == magnetic)
    vignet_oct = fit_vignet(dfsub2, mask, ax=ax1, plot=plot)

    # Fit parameters for the tetrahedral site
    site_type = 'tet'
    mask = (dfsub2.Nnn == dict_Nnn[site_type]) & (dfsub2.ismear == ismear) & \
        (dfsub2.magmom == magnetic)
    vignet_tet = fit_vignet(dfsub2, mask, ax=ax2, plot=plot)

    # Fit parameter for the matrix
    mask = (dfpure.ismear == 2) & (dfpure.magmom == magnetic)
    vignet_pure = fit_vignet(dfpure, mask, ax=ax3, plot=plot)

    if plot:
        ax1.set_title(matrix + '-' + solname + ' (oct)')
        ax2.set_title(matrix + '-' + solname + ' (tet)')
        ax3.set_title(matrix + ' (Pure)')
        fig.tight_layout()

    return vignet_oct, vignet_tet, vignet_pure
