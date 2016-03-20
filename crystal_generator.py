import numpy as np
import sys
# import time
# from itertools import izip


class Crystal(object):
    """
        This is a general crystal class along with
        convenient methods to manipulate the crystal
    """
    def __init__(self, name='crystal'):
        self._name = name
        self._name = ''            # name of the crystal
        self._rcar = []            # Cartesian coodinates
        self._rrel = []            # relative coordinates
        self._scale = []           # scale of the lattice vectors
        self._lattvec = np.eye(3)  # lattice vectors
        self._itype = []           # integer identifier of the basis atoms
        self._natoms = []          # number of atoms
        self._ntypes = []          # number of types of atoms
        self._dict_types = {}      # dictionary of atom types
        self._ndist = []           # neighbor distances

    @property
    def rcar(self):
        """ Returns the [x,y,z] Cartesian coordinates of the basis atoms """
        return self._rcar

    @rcar.setter
    def rcar(self, new_rcar):
        self._rcar = new_rcar
        self.update_rrel_from_rcar()

    @property
    def xcar(self):
        """ Returns the x Cartesian coordinates of the basis atoms """
        return self.rcar[:, 0]

    @xcar.setter
    def xcar(self, new_xcar):
        self._rcar[:, 0] = new_xcar
        self.update_rrel_from_rcar()

    @property
    def ycar(self):
        """ Returns the y Cartesian coordinates of the basis atoms """
        return self._rcar[:, 1]

    @ycar.setter
    def ycar(self, new_ycar):
        self._rcar[:, 1] = new_ycar
        self.update_rrel_from_rcar()

    @property
    def zcar(self):
        """ Returns the y Cartesian coordinates of the basis atoms """
        return self._rcar[:, 2]

    @zcar.setter
    def zcar(self, new_zcar):
        self._rcar[:, 2] = new_zcar
        self.update_rrel_from_rcar()

    @property
    def rrel(self):
        """ Returns the relative [x,y,z] coordinates of the basis atoms """
        return self._rrel

    @rrel.setter
    def rrel(self, new_rrel):
        self._rrel = new_rrel
        self.update_rcar_from_rrel()

    @property
    def xrel(self):
        """ Returns the x reltesian coordinates of the basis atoms """
        return self.rrel[:, 0]

    @xrel.setter
    def xrel(self, new_xrel):
        self._rrel[:, 0] = new_xrel
        self.update_rcar_from_rrel()

    @property
    def yrel(self):
        """ Returns the relative y coordinates of the basis atoms """
        return self._rrel[:, 1]

    @yrel.setter
    def yrel(self, new_yrel):
        self._rrel[:, 1] = new_yrel
        self.update_rcar_from_rrel()

    @property
    def zrel(self):
        """ Returns the relative z coordinates of the basis atoms """
        return self._rrel[:, 2]

    @zrel.setter
    def zrel(self, new_zrel):
        self._rrel[:, 2] = new_zrel
        self.update_rcar_from_rrel()

    @property
    def itype(self):
        """ Return interger type identifier of basis atoms """
        return self._itype

    @property
    def stype(self):
        """ Return string types of basis atoms according to _dict_types """
        return [self._dict_types[itype] for itype in self.itype]

    @property
    def natoms(self):
        """ returns the number of basis atoms in the crystal """
        return len(self.xcar)

    @property
    def scale(self):
        """ Returns the scale for the lattice vectors """
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        """ Sets the new scale for the crystal """
        self._scale = new_scale
        self.update_rcar_from_rrel()

    @property
    def lattvec(self):
        """ Returns the lattice vectors of the crystal """
        return self._lattvec

    @lattvec.setter
    def lattvec(self, new_lattvec):
        """ Sets the lattice vectors of the crystal """
        self._lattvec = new_lattvec
        self.update_rcar_from_rrel()

    @property
    def dict_types(self):
        """ Returns dictionary for atom types (itype) """
        return self._dict_types

    @dict_types.setter
    def dict_types(self, new_dict_types):
        self._dict_types = new_dict_types

    def update_rrel_from_rcar(self):
        """ Updates relative coordinates  based on Cartesian coordinates """
        cellvec_inv = np.linalg.inv(self._scale*self._lattvec)
        self._rrel = np.dot(self._rcar, cellvec_inv)

    def update_rcar_from_rrel(self):
        """ Updates Cartesian coordinates based on relative coordinates """
        self._rcar = np.dot(self._rrel, self._scale*self._lattvec)

    def _consolidate_details(self):
        """ Considates the variables in the class and makes them consistent """
        self._natoms = int(len(self.xcar))
        itype_unique = np.unique(self._itype)
        self._ntypes = len(itype_unique)

    def rotate_crystal(self, Rmatrix):
        """
            Rotates the crystal by Rmatrix
            Definitions:
                unit vectors in the old coodinate system -> ex , ey , ez
                unit vectors in the new coodinate system -> ex', ey', ez'
                Rmat = [[ex'*ex, ex'*ey, ex'*ez],
                        [ey'*ex, ey'*ey, ey'*ez],
                        [ez'*ex, ez'*ey, ez'*ez]]
        """
        self.lattvec = np.dot(self.lattvec, Rmatrix.T)
        self.update_rcar_from_rrel()
        self.clean_coordinates()

    def delete_atom(self, idnull):
        """ Delete atoms in the crystal  """
        self.rcar = np.delete(self._rcar, idnull, 0)
        self._consolidate_details()

    def translate_cartesian(self, rtrans_cart):
        """ Translate the cartensian coodinates of the basis by rtrans_cart """
        self.rcar = self.rcar - rtrans_cart

    def translate_relative(self, rtrans_rel):
        """ Translate the cartensian coodinates of the basis by rtrans_rel """
        self.rrel = self._rrel - rtrans_rel

    def reflect_atoms(self):
        """ Reflects atoms inside the primary lattice cell """
        mask = self.rrel >= 1.
        self._rrel[mask] = self._rrel[mask] - 1.
        mask = self.rrel < 0.
        self._rrel[mask] = self._rrel[mask] + 1.
        self.update_rcar_from_rrel()

    def center_atoms(self):
        """ Centers atoms around the origin """
        mask = self.rrel >= 0.5
        self._rrel[mask] = self._rrel[mask] - 1.
        mask = self.rrel < -0.5
        self._rrel[mask] = self._rrel[mask] + 1.
        self.update_rcar_from_rrel()

    def reset_crystal(self):
        """ Resents the crystal """
        self._rcar = []
        self._rrel = []
        self._scale = []
        self._lattvec = np.eye(3)
        self._itype = []
        self._natoms = []
        self._ntypes = []
        self._dict_types = []

    def clean_coordinates(self, ndecimals=12):
        """ Rounds the relative coodinates to the nearest ndecimal value """
        self.lattvec = np.round(self.lattvec, ndecimals)
        self.rrel = np.round(self.rrel, ndecimals)
        self.update_rcar_from_rrel()

    def create_supercell(self, n1, n2, n3, name=''):
        """
            Creates an n1xn2xn3 supercell of unit
            Inputs: n1, n2, n3 - number of translations in the v1, v2 and v3
                                 directions, respectively
                    name       - name of the supercell
            Returns supercell
        """
        supercell = Crystal(name=name)
        supercell.dict_types = self._dict_types

        # define new lattice vectors
        supercell._scale = self.scale
        new_lattvec = np.zeros([3, 3])
        new_lattvec[0, :] = self._lattvec[0, :] * n1
        new_lattvec[1, :] = self._lattvec[1, :] * n2
        new_lattvec[2, :] = self._lattvec[2, :] * n3
        supercell._lattvec = new_lattvec

        # create translation matrix
        var1 = np.mgrid[0:n1, 0:n2, 0:n3]
        transmat = np.zeros([n1*n2*n3, 3])
        transmat[:, 0] = var1[0, :, :].flatten(1)
        transmat[:, 1] = var1[1, :, :].flatten(1)
        transmat[:, 2] = var1[2, :, :].flatten(1)

        # calculate supercell lattice positions
        sclattpos = np.dot(transmat, self._scale*self._lattvec)

        # calculate the cartesian coordinates of the supercell basis
        for ii in xrange(self.natoms):
            if len(supercell._rcar) == 0:
                supercell._rcar = sclattpos + self._rcar[ii, :]
                supercell._itype = (np.ones(n1*n2*n3) *
                                    self._itype[ii]).astype('int')
            else:
                supercell._rcar = \
                    np.append(supercell._rcar,
                              sclattpos+self._rcar[ii, :], axis=0)
                supercell._itype = \
                    np.append(supercell._itype,
                              (np.ones(n1*n2*n3) *
                               self._itype[ii]).astype('int'), axis=0)

        # update cell data
        supercell.update_rrel_from_rcar()
        supercell._consolidate_details()
        supercell.clean_coordinates(ndecimals=12)

        return supercell

    # TODO: Special consideration must be added if the basis of the crystal
    #       are not equivalent
    def create_new_orientation(self, v1, v2, v3, n1=20, n2=20, n3=20,
                               name='', tolerance=1.E-10):
        """
            Creates a cell with cellvectors in the new orientation with the
            minimum volume.
            Requirement: All the basis atoms in base_crystal must be equivalent
            Inputs:
                base_crystal  - base crystal for the new unit cell in the given
                                orientation
                v1, v2, v3    - new cell vector directions
                n1, n2, n3    - defines the size of the supercell of
                                base_crystal that will be used to find the
                                output crystal. (default: n1=n2=n3=20)
                tolerance     - defines tolerance due to numerical errors
                                (default=1.e-10)
            Returns newcell
        """
        # create normalized lattice vectors
        lattvec_norm = np.array([v1/np.linalg.norm(v1),
                                 v2/np.linalg.norm(v2),
                                 v3/np.linalg.norm(v3)])

        # creaste a supercell of base_crystal
        newcell = self.create_supercell(n1, n2, n3, name=name)

        # translate the supercell such that an atom lies in the origin
        newcell.translate_cartesian(newcell.rcar[0, :])

        # center atoms about the origin
        newcell.center_atoms()

        # normalize cartesian coordinates to get unit vectors
        rcar_norm = np.linalg.norm(newcell.rcar, axis=1)
        rcar_norm_vec = np.repeat(rcar_norm, 3).reshape(newcell.natoms, 3)
        idv, = np.where(rcar_norm != 0.)
        ercar = np.zeros(newcell.rcar.shape)
        ercar[idv, :] = newcell.rcar[idv, :] / rcar_norm_vec[idv, :]

        # initialize lattice vectors of the output crystal
        lattvec = np.zeros([3, 3])

        # calculate dot product of normalized lattice vectors with the
        # normalized cartesian coordinates
        for ii in xrange(3):
            # define unit vector of the new lattice vector
            ev = lattvec_norm[ii, :]
            # find ids of atoms whose cartesian coordinates are paralell to ev
            idv, = np.where(np.dot(ercar, ev) > 1.-tolerance)
            # calculate the length of the vector from the origin to the atoms
            # parallel to the lattice vectors
            d = np.linalg.norm(newcell.rcar[idv, :], axis=1)
            idv2, = np.where(d == d.min())
            lattvec[ii, :] = newcell.rcar[idv[idv2], :]

        # define the new cell's lattice vectors
        newcell._lattvec = lattvec / newcell.scale
        # update relative coordinates based on new lattice vectors
        newcell.update_rrel_from_rcar()
        # shift the relative coordinates (to prevent some atoms
        # lying on the cell boundary
        rrel_trans = -np.ones(3)*1.e-2
        newcell.translate_relative(rrel_trans)
        # delete atoms lying outside the cell boundary
        iddel, = np.where(
                (newcell.xrel < 0.) | (newcell.xrel >= 1.) |
                (newcell.yrel < 0.) | (newcell.yrel >= 1.) |
                (newcell.zrel < 0.) | (newcell.zrel >= 1.))
        newcell.delete_atom(iddel)
        # revert to the original relative coordinates
        newcell.translate_relative(-rrel_trans)
        # clean the coodinates
        newcell.clean_coordinates()

        return newcell

    def reorient(self, dir1=0, dir2=1, nroll=0):
        """
            Reorients the crystal
        """
        dict_dir = {0: 0, 1: 1, 2: 2, 'x': 0, 'y': 1, 'z': 2}
        v1 = self.lattvec[dict_dir[dir1], :]
        v1 = v1 / np.linalg.norm(v1)
        v2 = self.lattvec[dict_dir[dir2], :]
        v2 = v2 - np.dot(v1, v2) * v1
        Rmat = create_rotation_matrix(v1, v2, nroll)
        self.rotate_crystal(Rmat)

        return Rmat

    def find_distance(self):
        """
            Finds distance between the points in rcar with each other
            Input:   rcar (Nx3 array) - cartesian position of points
            Returns: dist (NxN array) - the distance of the points

            dist[ii,:] gives the distance of the points with respect to
            atom ii.
        """
        diff = self.rcar.reshape(self.natoms, 1, 3) - self.rcar
        dist = np.sqrt((diff**2.).sum(2))
        idiag = np.arange(self.natoms)
        dist[idiag, idiag] = np.inf

        return dist

    def find_distance_peroidic(self):
        """
            Finds distance between the points in rcar with each other.
            Takes into accout perodicity of the crystal
            Input:   rcar (Nx3 array) - cartesian position of points
            Returns: dist (NxN array) - the distance of the points

            dist[ii,:] gives the distance of the points with respect to
            atom ii.
        """
        # define the relative shifts to take care of periodicity
        xrel_shift, yrel_shift, zrel_shift = np.meshgrid(
                np.arange(0., 1.5, 0.5), np.arange(0., 1.5, 0.5),
                np.arange(0., 1.5, 0.5))
        rrel_shift_array = np.array([xrel_shift.flatten(),
                                     yrel_shift.flatten(),
                                     zrel_shift.flatten()]).T
        nshifts = len(rrel_shift_array)

        self._ndist = np.ones([self.natoms, self.natoms]) * np.inf

        for ii, rrel_shift in enumerate(rrel_shift_array):
            sys.stdout.write(
                    "\rGetting neighbor distances (periodic): %3d%%"
                    % ((ii + 1.) / nshifts * 100))
            sys.stdout.flush()
            # shift the crystal
            self.translate_relative(rrel_shift)
            # center atom around the shift
            self.center_atoms()
            # find the distance of the atoms
            new_dist = self.find_distance()
            # update distance matrix
            mask = new_dist < self._ndist
            self._ndist[mask] = new_dist[mask]

        print "\n"
        return self._ndist

    @property
    def ndist(self):
        """ Returns the neighbor distance list """
        return self._ndist

    def get_neighbor_list(self, dmax, dmin=0.):
        """
            Returns a list of atoms that with distance dmin < d < dmax
            from each atom in the cell.
        """
        neighbor_list = []

        # iterate over all atoms
        for ndist in self.ndist:
            idneighbors, = np.where((ndist > dmin) &
                                    (ndist < dmax))
            neighbor_list.append(idneighbors)

        return neighbor_list


def get_unit_cell(uc_name, a=[], b=[], c=[]):
    """
        A crystal subclass with predefined unit cells of
        common crystals
    """
    # Initialize the relative coordinates
    rrel = []

    if uc_name == 'simple cubic':
        natoms = 1
        dict_types = {1: 'A'}
        scale = a
        lattvec1 = np.array([1., 0., 0.])
        lattvec2 = np.array([0., 1., 0.])
        lattvec3 = np.array([0., 0., 1.])
        rrel = np.zeros([natoms, 3])
        itype = np.zeros(natoms).astype('int')
        #
        ii = 0
        rrel[ii, :] = np.array([0., 0., 0.])
        itype[ii] = 1
        #
    elif uc_name == 'bcc cubic':
        natoms = 2
        dict_types = {1: 'A'}
        scale = a
        lattvec1 = np.array([1., 0., 0.])
        lattvec2 = np.array([0., 1., 0.])
        lattvec3 = np.array([0., 0., 1.])
        rrel = np.zeros([natoms, 3])
        itype = np.zeros(natoms).astype('int')
        #
        ii = 0
        rrel[ii, :] = np.array([0.0, 0.0, 0.0])
        itype[ii] = 1
        #
        ii = 1
        rrel[ii, :] = np.array([0.5, 0.5, 0.5])
        itype[ii] = 1
        #
    elif uc_name == 'fcc cubic':
        natoms = 4
        dict_types = {1: 'A'}
        scale = a
        lattvec1 = np.array([1., 0., 0.])
        lattvec2 = np.array([0., 1., 0.])
        lattvec3 = np.array([0., 0., 1.])
        rrel = np.zeros([natoms, 3])
        itype = np.zeros(natoms).astype('int')
        #
        ii = 0
        rrel[ii, :] = np.array([0.0, 0.0, 0.0])
        itype[ii] = 1
        #
        ii = 1
        rrel[ii, :] = np.array([0.5, 0.5, 0.0])
        itype[ii] = 1
        #
        ii = 2
        rrel[ii, :] = np.array([0.5, 0.0, 0.5])
        itype[ii] = 1
        #
        ii = 3
        rrel[ii, :] = np.array([0.0, 0.5, 0.5])
        itype[ii] = 1
        #
    elif uc_name == 'fcc primitive':
        natoms = 1
        dict_types = {1: 'A'}
        scale = a
        lattvec1 = np.array([0.5, 0.5, 0.0])
        lattvec2 = np.array([0.5, 0.0, 0.5])
        lattvec3 = np.array([0.0, 0.5, 0.5])
        rrel = np.zeros([natoms, 3])
        itype = np.zeros(natoms).astype('int')
        #
        ii = 0
        rrel[ii, :] = np.array([0.0, 0.0, 0.0])
        itype[ii] = 1
        #
        # TODO: hexagonal unit cells
        #       - additional support for hexagonal structures
        #           > hex indices to conventional miller indices
        # elif uc_name=='simple hexagonal':

    unit_cell = Crystal(name=uc_name)
    unit_cell._scale = scale
    unit_cell._lattvec = np.array([lattvec1, lattvec2, lattvec3])
    unit_cell._dict_types = dict_types
    unit_cell.rrel = rrel
    unit_cell._itype = itype
    unit_cell._consolidate_details()

    return unit_cell


def create_rotation_matrix(v1, v2, nroll=0):
    """
        Creates a right hand rotation matrix with with directions
        v1, v2, v1xv2
        Inputs: v1, v2 - vectors defining the orientation of the rotation
                nroll  - rolls the resulting matrix
        Returns Rmat - the rotation matrix
    """
    ev1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(v2, ev1)*ev1
    ev2 = v2 / np.linalg.norm(v2)
    ev3 = np.cross(ev1, ev2)

    return np.roll(np.array([ev1, ev2, ev3]), nroll, axis=0)


def find_distances(rcar):
    """
        Finds distance between the points in rcar with each other
        Input:   rcar (Nx3 array) - cartesian position of points
        Returns: dist (NxN array) - the distance of the points

        dist[ii,:] gives the distance of the points with respect to
        atom ii.
    """
    diff = rcar.reshape(rcar.shape[0], 1, 3) - rcar
    dist = np.sqrt((diff**2.).sum(2))
    ii = np.arange(rcar.shape[0])
    dist[ii, ii] = np.inf

    return dist

if __name__ == "__main__":
    unit = get_unit_cell('fcc primitive', 2.8)
    # unit = get_unit_cell('bcc cubic', 2.8)
    # unit = get_unit_cell('fcc cubic', 2.8)

    v1 = np.array([-1., 1., 0.])
    v2 = np.array([1., 1., 1.])

    # v1 = np.array([-1., 1., 0.])
    # v2 = np.array([0., 0., 1.])
    v3 = np.cross(v1, v2)

    v1 = np.array([1., 0., 0.])
    v2 = np.array([0., 1., 0.])
    v3 = np.cross(v1, v2)

    sc = unit.create_supercell(10, 10, 10, name='supercell')
    # sc = unit.create_supercell(4,4,4, name='supercell')
    newcell = unit.create_new_orientation(v1, v2, v3, name='new orientation')
    newcell.reorient()

    # Rmat = create_rotation_matrix(v1, v2)
    # newcell.rotate_crystal(Rmat)

    dist = sc.find_distance_peroidic()

    a0 = sc.scale
    ep = a0 * 1.e-5
    d0 = 0.
    d1 = a0*(1./np.sqrt(2.))
    d2 = a0

    idn1 = sc.get_neighbor_list(dmax=d1+ep, dmin=d0+ep)
    idn2 = sc.get_neighbor_list(dmax=d2+ep, dmin=d1+ep)

    rcar = sc.rcar
