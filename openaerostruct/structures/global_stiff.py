from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from openaerostruct.utils.testing import view_mat

class GlobalStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.ny = ny = surface['mesh'].shape[1]

        self.more_dof = 0
        if surface['name'] == 'wing':
            self.more_dof = 1
        
        size = 6 * ny + 6 + self.more_dof

        self.add_input('nodes', shape=(ny, 3), units='m')
        self.add_input('local_stiff_transformed', shape=(ny - 1, 12, 12))
        self.add_output('K', shape=(size, size), units='N/m')

        arange = np.arange(ny - 1)

        rows = np.empty((ny - 1, 12, 12), int)
        for i in range(12):
            for j in range(12):
                mtx_i = 6 * arange + i
                mtx_j = 6 * arange + j
                rows[:, i, j] = size * mtx_i + mtx_j
        rows = rows.flatten()
        cols = np.arange(144 * (ny - 1))
        self.declare_partials('K', 'local_stiff_transformed', val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        surface = self.options['surface']

        ny = self.ny

        size = 6 * ny + 6

        arange = np.arange(ny - 1)

        outputs['K'] = 0.
        for i in range(12):
            for j in range(12):
                outputs['K'][6 * arange + i, 6 * arange + j] += inputs['local_stiff_transformed'][:, i, j]

        # Find constrained nodes based on closeness to central point
        nodes = inputs['nodes']
        dist = nodes - np.array([5., 0, 0])
        idx = (np.linalg.norm(dist, axis=1)).argmin()
        index = 6 * idx
        num_dofs = 6 * ny

        arange = np.arange(6)

        outputs['K'][index + arange, num_dofs + arange] = 1.e9
        outputs['K'][num_dofs + arange, index + arange] = 1.e9

        if surface['name'] == 'wing':
            wingbox_loc = -2.8
            dist = nodes[:,1] - wingbox_loc
            diff = np.abs(dist)
            idx = diff.argmin()

            index = 6 * idx
            num_dofs = 6 * ny + 4 # why is it +4?!
            arange = 2
            outputs['K'][index + arange, num_dofs + arange] = 1.e9
            outputs['K'][num_dofs + arange, index + arange] = 1.e9