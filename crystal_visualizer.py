# import os
# import sys
# import time
import copy
import numpy as np
# import scipy as sp
import crystaltools as ct
import matplotlib.pyplot as plt
# import matplotlib        as mpl
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


class CrystalPlot2D():
    def __init__(self, ms=180., lc='k', lw=1.5, mfc='r', marker='o',
                 fwidth=4., fheight=4., bool_interactive=True):
        self.ms = ms
        self.marker = marker
        self.lc = lc
        self.lw = lw
        self.mfc = mfc
        self.fwidth = fwidth
        self.fheight = fheight

        self.bool_interactive = bool_interactive

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def plot_simple(self, crystal, v1=[], v2=[], fignum=1):
        """
            Simple 2D plot of the crystal
            crys - crsytal
            will be projected along v1 and (v2 - (v2*v1)v1)
        """
        # create a copy of the crystal
        crys = copy.deepcopy(crystal)

        # set default values for v1 and v2 if not defined
        if len(v1) == 0:
            v1 = crys.lattvec()[0, :]
        if len(v2) == 0:
            v2 = crys.lattvec()[1, :]

        # calculate rotation matrix
        Rmat = ct.create_rotation_matrix(v1, v2)

        # rotate crystal
        crys.rotate_crystal(Rmat)
        crys.reflect_atoms()

        if self.bool_interactive:
            plt.ion()

        fig = plt.figure(fignum, figsize=(self.fwidth, self.fheight))
        fig.clf()
        ax = fig.add_subplot(111)
        hf2 = self._plot_crystal_cell(crys, ax)
        hf1 = self._scatter_basis(crys, ax, zorder=5)

        plt.axis('equal')

        return crys, fig, ax

    def _scatter_basis(self, crys, ax, zorder=2):
        xcar = crys.xcar
        ycar = crys.ycar
        # hfig = ax.scatter(xcar, ycar, marker=self.marker, s=self.ms, \
        #         facecolors=self.mfc, zorder=zorder)

        for ii in range(len(xcar)):
            x = xcar[ii]
            y = ycar[ii]
            z = int(crys.zcar[ii] // 1)
            hfig = ax.scatter(x, y, marker=self.marker, s=self.ms,
                              facecolors=self.mfc, zorder=z)

            print z

        return hfig

    def _plot_crystal_cell(self, crys, ax):
        p0 = np.zeros(2)
        p1 = crys.lattvec[0, 0:2] * crys.scale
        p2 = crys.lattvec[1, 0:2] * crys.scale
        p3 = crys.lattvec[2, 0:2] * crys.scale

        hfig = self._plot_line(ax, p0      , p1      , zorder=-100)
        hfig = self._plot_line(ax, p1      , p1+p2   , zorder=-100)
        hfig = self._plot_line(ax, p1+p2   , p2      , zorder=-100)
        hfig = self._plot_line(ax, p2      , p0      , zorder=-100)
        hfig = self._plot_line(ax, p0   +p3, p1   +p3, zorder=-100)
        hfig = self._plot_line(ax, p1   +p3, p1+p2+p3, zorder=-100)
        hfig = self._plot_line(ax, p1+p2+p3, p2   +p3, zorder=-100)
        hfig = self._plot_line(ax, p2   +p3, p0   +p3, zorder=-100)
        hfig = self._plot_line(ax, p0      , p0   +p3, zorder=-100)
        hfig = self._plot_line(ax, p1      , p1   +p3, zorder=-100)
        hfig = self._plot_line(ax, p1+p2   , p1+p2+p3, zorder=-100)
        hfig = self._plot_line(ax, p2      , p2   +p3, zorder=-100)

        # hfig = self._plot_line(ax, p0, p2, zorder=3)
        # hfig = self._plot_line(ax, p0, p3, zorder=3)
        # hfig = self._plot_line(ax, p0, p3, zorder=3)

    def _plot_line(self, ax, p1, p2, zorder=1):
        """ Plots a line from p1 to p2 """
        hfig = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                       lw=self.lw, color=self.lc)

        return hfig

class CrystalPlot3D():
    def __init__(self, ms=40., lc='k', lw=1.5, mfc='r', marker='o', \
            fwidth = 4., fheight=4., bool_interactive=True):
        self.ms = ms
        self.marker = marker
        self.lc = lc
        self.lw = lw
        self.mfc = mfc
        self.fwidth = fwidth
        self.fheight = fheight

        self.bool_interactive = bool_interactive

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def plot_simple(self, crystal, v1=[], v2=[], fignum=1 ):
        """
            Simple 2D plot of the crystal
            crys - crsytal
            will be projected along v1 and (v2 - (v2*v1)v1)
        """
        # create a copy of the crystal
        crys = copy.deepcopy(crystal)

        # set default values for v1 and v2 if not defined
        if len(v1)==0:
            v1 = crys.lattvec()[0,:]
        if len(v2)==0:
            v2 = crys.lattvec()[1,:]

        # calculate rotation matrix
        Rmat = ct.create_rotation_matrix(v1,v2)

        # rotate crystal
        crys.rotate_crystal(Rmat)
        crys.reflect_atoms()

        if self.bool_interactive:
            plt.ion()

        fig = plt.figure(fignum, figsize=(self.fwidth, self.fheight))
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        hf2 = self._plot_crystal_cell(crys, ax)
        hf1 = self._scatter_basis(crys, ax, zorder=5)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.set_size_inches(self.fwidth, self.fheight)

        return crys, fig, ax

    def _scatter_basis(self, crys, ax, zorder=2):
        xcar = crys.xcar
        ycar = crys.ycar
        zcar = crys.zcar

        hfig = ax.scatter(xcar, ycar, zcar,  marker=self.marker, s=self.ms,
                          facecolors=self.mfc, zorder=zorder)

        max_range = np.array(
                [xcar.max()-xcar.min(), ycar.max()-ycar.min(),
                 zcar.max()-zcar.min()]).max() / 2.0 * 1.5

        mean_x = xcar.mean()
        mean_y = ycar.mean()
        mean_z = zcar.mean()
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y - max_range, mean_y + max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)

        return hfig

    def _plot_crystal_cell(self, crys, ax):
        p0 = np.zeros(3)
        p1 = crys.lattvec[0,:] * crys.scale
        p2 = crys.lattvec[1,:] * crys.scale
        p3 = crys.lattvec[2,:] * crys.scale

        hfig = self._plot_line(ax, p0      , p1      , zorder=1)
        hfig = self._plot_line(ax, p1      , p1+p2   , zorder=1)
        hfig = self._plot_line(ax, p1+p2   , p2      , zorder=1)
        hfig = self._plot_line(ax, p2      , p0      , zorder=1)

        hfig = self._plot_line(ax, p0   +p3, p1   +p3, zorder=1)
        hfig = self._plot_line(ax, p1   +p3, p1+p2+p3, zorder=1)
        hfig = self._plot_line(ax, p1+p2+p3, p2   +p3, zorder=1)
        hfig = self._plot_line(ax, p2   +p3, p0   +p3, zorder=1)

        hfig = self._plot_line(ax, p0      , p0   +p3, zorder=1)
        hfig = self._plot_line(ax, p1      , p1   +p3, zorder=1)
        hfig = self._plot_line(ax, p1+p2   , p1+p2+p3, zorder=1)
        hfig = self._plot_line(ax, p2      , p2   +p3, zorder=1)

        #hfig = self._plot_line(ax, p0, p2, zorder=3)
        #hfig = self._plot_line(ax, p0, p3, zorder=3)
        #hfig = self._plot_line(ax, p0, p3, zorder=3)


    def _plot_line(self, ax, p1, p2, zorder=1):
        """ Plots a line from p1 to p2 """
        hfig = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],\
                '-', lw=self.lw, color=self.lc)

        return hfig

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-1.e-10,zback]])

if __name__ == "__main__":
    proj3d.persp_transformation = orthogonal_proj

    unit = ct.get_unit_cell('fcc primitive', 2.8)
    # unit = ct.get_unit_cell('bcc cubic', 2.8)
    # unit = ct.get_unit_cell('fcc cubic', 2.8)
    # sc = unit.create_supercell(10,10,10, name='supercell')
    sc = unit.create_supercell(5, 5, 5, name='supercell')
    # sc = unit.create_supercell(10,10,10, name='supercell')

    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, 1, 1])
    # v2 = np.array([1,1,-2])

    # v1 = np.array([1,0,0])
    # v2 = np.array([0,1,0])

    # v1 = np.array([1,1,0])
    # v2 = np.array([1,0,1])

    myplot = CrystalPlot2D(lc='0.5')
    myplot.ms = 40
    crys, fig, ax = myplot.plot_simple(sc, v1, v2)

    # v1 = np.array([1,0,0])
    # v2 = np.array([0,1,0])
    myplot = CrystalPlot3D(lc='0.5', ms=80)
    crys, fig, ax = myplot.plot_simple(sc, v1, v2, fignum=2)

    #for ii in range(1):
    #    unit = cg.get_unit_cell('fcc primitive', 2.8)

    #    v1 = np.array([-1., 1., 0.])
    #    v2 = np.array([ 1., 1., 1.])

    #    #v1 = np.array([-1., 1., 0.])
    #    #v2 = np.array([ 0., 0., 1.])

    #    v3 = np.cross(v1,v2)

    #    #v1 = np.array([ 1., 1., 0.])
    #    #v2 = np.array([ 1., 0., 1.])
    #    #v3 = np.array([ 0., 1., 1.])

    #    sc = unit.create_supercell(4,4,4, name='supercell')
    #    newcell = unit.create_new_orientation(v1, v2, v3, name='new orientation')
    #    newcell.reorient()

    #    dist = sc.find_distance_peroidic()

    #    a0 = sc.scale()
    #    ep = a0 * 1.e-5
    #    d0 = 0.
    #    d1 = a0*(1./np.sqrt(2.))
    #    d2 = a0

    #    idn1 = sc.get_neighbor_list(dmax=d1+ep, dmin=d0+ep)
    #    idn2 = sc.get_neighbor_list(dmax=d2+ep, dmin=d1+ep)
