#!/usr/bin/env python
import os
import stat
# import sys
# import numpy as np
# import copy as copy
from shutil import copyfile


class VASP_job(object):
    """
        Creates a VASP input deck.
    """
    def __init__(self, crystal, root_dir, subdir_name, jobname='VASP job',
                 path_to_INCAR=[], path_to_POTCAR=[], path_to_KPOINTS=[],
                 path_to_support_dir=[], support_file_list=[]):
        """
            :parameters:
                crystal:(crystal):
                    crystal class to generate the POSCAR file
                root_dir:(string):
                    root directory of the VASP job
                subdir_name:(string):
                    name of job subdirectory in root_dir
                jobname:(string):
                    name of the job
                path_to_INCAR:(string):
                    path to the INCAR file
                path_to_POTCAR:(string):
                    path to the POTCAR file
                path_to_KPOINTS:(string):
                    path to the KPOINTS file
                path_to_support_directory:(string):
                    path to the directory of any supporting files that would be
                    included in the job directory
                support_file_list:(string):
                    list of support files in path_to_support_directory
        """

        self.crystal = crystal
        self.root_dir = root_dir
        self.subdir_name = subdir_name
        self.jobname = jobname
        self.path_to_POTCAR = path_to_POTCAR
        self.path_to_INCAR = path_to_INCAR
        self.path_to_KPOINTS = path_to_KPOINTS
        self.path_to_support_dir = path_to_support_dir
        self.support_file_list = support_file_list

    def generate_input_deck(self):
        """
            Creates VASP input deck at self.root_dir/self.subdir_name
            contaning the POSCAR, INCAR, POTCAR, KPOINTS and files
            defined by self.support_file_list.
        """
        # create working directory if it doesn't exist
        workingdir = os.path.join(self.root_dir, self.subdir_name)
        if not os.path.exists(workingdir):
            os.makedirs(workingdir)

        # create imput deck
        self.crystal.to_POSCAR(workingdir, 'POSCAR', title=self.jobname)
        copyfile(self.path_to_INCAR, os.path.join(workingdir, 'INCAR'))
        copyfile(self.path_to_POTCAR, os.path.join(workingdir, 'POTCAR'))
        copyfile(self.path_to_KPOINTS, os.path.join(workingdir, 'KPOINTS'))

        for filename in self.support_file_list:
            path_to_file = os.path.join(self.path_to_support_dir, filename)
            copyfile(path_to_file, os.path.join(workingdir, filename))
            os.chmod(os.path.join(workingdir, filename), 0777)

        return
