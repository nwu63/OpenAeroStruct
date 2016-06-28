""" Manipulate geometry mesh based on high-level design parameters """

from __future__ import division
import numpy
from numpy import cos, sin, tan

from openmdao.api import Component

from b_spline import get_bspline_mtx
from crm_data import crm_base_mesh
from b_spline import get_bspline_mtx


def rotate(mesh, thetas):
    """ Computes rotation matricies given mesh and rotation angles in degress """
    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25*te + 0.75*le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    rad_thetas = thetas * numpy.pi / 180.

    mats = numpy.zeros((ny, 3, 3), dtype="complex")
    mats[:, 0, 0] = cos(rad_thetas)
    mats[:, 0, 2] = sin(rad_thetas)
    mats[:, 1, 1] = 1
    mats[:, 2, 0] = -sin(rad_thetas)
    mats[:, 2, 2] = cos(rad_thetas)

    for ix in range(nx):
        row = mesh[ix]
        row[:] = numpy.einsum("ikj, ij -> ik", mats, row - quarter_chord)
        row += quarter_chord
    return mesh

def sweep(mesh, angle):
    """ Shearing sweep angle. Positive sweeps back. """

    num_x, num_y, _ = mesh.shape
    ny2 = (num_y-1)/2

    le = mesh[0]
    te = mesh[-1]

    y0 = le[ny2, 1]

    tan_theta = tan(numpy.radians(angle))
    dx_right = (le[ny2:, 1] - y0) * tan_theta
    dx_left = -(le[:ny2, 1] - y0) * tan_theta
    dx = numpy.hstack((dx_left, dx_right))

    for i in xrange(num_x):
        mesh[i, :, 0] += dx

    return mesh

def dihedral(mesh, angle):
    """ Dihedral angle. Positive bends up. """

    num_x, num_y, _ = mesh.shape
    ny2 = (num_y-1)/2

    le = mesh[0]
    te = mesh[-1]

    y0 = le[ny2, 1]

    tan_theta = tan(numpy.radians(angle))
    dx_right = (le[ny2:, 1] - y0) * tan_theta
    dx_left = -(le[:ny2, 1] - y0) * tan_theta
    dx = numpy.hstack((dx_left, dx_right))

    for i in xrange(num_x):
        mesh[i, :, 2] += dx

    return mesh


def stretch(mesh, length):
    """ Strech mesh in span-wise direction to reach specified length"""

    le = mesh[0]
    te = mesh[-1]

    num_x, num_y, _ = mesh.shape

    span = le[-1, 1] - le[0, 1]
    dy = (length - span) / (num_y - 1) * numpy.arange(1, num_y)

    for i in xrange(num_x):
        mesh[i, 1:, 1] += dy

    return mesh

def taper(mesh, taper_ratio):
    """ Change the spanwise chord to produce a tapered wing"""

    le = mesh[0]
    te = mesh[-1]
    num_x, num_y, _ = mesh.shape
    ny2 = int((num_y+1)/2)

    tele = te - le
    center_chord = .5 * te + .5 * le
    span = le[-1, 1] - le[0, 1]
    taper = numpy.linspace(1, taper_ratio, ny2)[::-1]

    jac = get_bspline_mtx(ny2, ny2, mesh, order=2)
    taper = jac.dot(taper)

    dx = numpy.hstack((taper, taper[::-1][1:]))

    for i in xrange(num_x):
        for ind in xrange(3):
            mesh[i, :, ind] = (mesh[i, :, ind] - center_chord[:, ind]) * \
                dx + center_chord[:, ind]

    return mesh


def mirror(mesh, right_side=True):
    """ Takes a half geometry and mirrors it across the symmetry plane.
    If right_side==True, it mirrors from right to left,
    assuming that the first point is on the symmetry plane. Else
    it mirrors from left to right, assuming the last point is on the
    symmetry plane.
    """

    num_x, num_y, _ = mesh.shape

    new_mesh = numpy.empty((num_x, 2 * num_y - 1, 3))

    mirror_y = numpy.ones(mesh.shape)
    mirror_y[:, :, 1] *= -1.0

    if right_side:
        new_mesh[:, :num_y, :] = mesh[:, ::-1, :] * mirror_y
        new_mesh[:, num_y:, :] = mesh[:,   1:, :]
    else:
        new_mesh[:, :num_y, :] = mesh[:, ::-1, :]
        new_mesh[:, num_y:, :] = mesh[:,   1:, :] * mirror_y[:, 1:, :]

    # shift so 0 is at the left wing tip (structures wants it that way)
    y0 = new_mesh[0, 0, 1]
    new_mesh[:, :, 1] -= y0

    return new_mesh


def gen_crm_mesh(n_points_inboard=2, n_points_outboard=2, num_x=3, mesh=crm_base_mesh):
    """ Builds the right hand side of the CRM wing with specified number
    of inboard and outboard panels
    """

    # LE pre-yehudi
    s1 = (mesh[0, 1, 0] - mesh[0, 0, 0]) / (mesh[0, 1, 1] - mesh[0, 0, 1])
    o1 = mesh[0, 0, 0]

    # TE pre-yehudi
    s2 = (mesh[1, 1, 0] - mesh[1, 0, 0]) / (mesh[1, 1, 1] - mesh[1, 0, 1])
    o2 = mesh[1, 0, 0]

    # LE post-yehudi
    s3 = (mesh[0, 2, 0] - mesh[0, 1, 0]) / (mesh[0, 2, 1] - mesh[0, 1, 1])
    o3 = mesh[0, 2, 0] - s3 * mesh[0, 2, 1]

    # TE post-yehudi
    s4 = (mesh[1, 2, 0] - mesh[1, 1, 0]) / (mesh[1, 2, 1] - mesh[1, 1, 1])
    o4 = mesh[1, 2, 0] - s4 * mesh[1, 2, 1]

    n_points_total = n_points_inboard + n_points_outboard - 1
    half_mesh = numpy.zeros((2, n_points_total, 3))

    # generate inboard points
    dy = (mesh[0, 1, 1] - mesh[0, 0, 1]) / (n_points_inboard - 1)
    for i in xrange(n_points_inboard):
        y = half_mesh[0, i, 1] = i * dy
        half_mesh[0, i, 0] = s1 * y + o1 # le point
        half_mesh[1, i, 1] = y
        half_mesh[1, i, 0] = s2 * y + o2 # te point

    yehudi_break = mesh[0, 1, 1]
    # generate outboard points
    dy = (mesh[0, 2, 1] - mesh[0, 1, 1]) / (n_points_outboard - 1)
    for j in xrange(n_points_outboard):
        i = j + n_points_inboard - 1
        y = half_mesh[0, i, 1] = j * dy + yehudi_break
        half_mesh[0, i, 0] = s3 * y + o3 # le point
        half_mesh[1, i, 1] = y
        half_mesh[1, i, 0] = s4 * y + o4 # te point

    full_mesh = mirror(half_mesh)
    full_mesh = add_chordwise_panels(full_mesh, num_x)
    return full_mesh

def add_chordwise_panels(mesh, num_x):
    """ Divides the wing into multiple chordwise panels. """
    le = mesh[ 0, :, :]
    te = mesh[-1, :, :]

    new_mesh = numpy.zeros((num_x, mesh.shape[1], 3))
    new_mesh[ 0, :, :] = le
    new_mesh[-1, :, :] = te

    for i in xrange(1, num_x-1):
        w = float(i) / (num_x - 1)
        new_mesh[i, :, :] = (1 - w) * le + w * te

    return new_mesh

def gen_mesh(num_x, num_y, span, chord, amt_of_cos=0.):
    mesh = numpy.zeros((num_x, num_y, 3))
    ny2 = (num_y + 1) / 2
    beta = numpy.linspace(0, numpy.pi/2, ny2)

    # mixed spacing with w as a weighting factor
    cosine = .5 * numpy.cos(beta) #  cosine spacing
    uniform = numpy.linspace(0, .5, ny2)[::-1] #  uniform spacing
    half_wing = cosine * amt_of_cos + (1 - amt_of_cos) * uniform
    full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * span

    for ind_x in xrange(num_x):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, full_wing[ind_y], 0]
    return mesh

class GeometryMesh(Component):
    """ Changes a given mesh with span, sweep, and twist
    des-vars. Takes in a half mesh with symmetry plane about
    the middle and outputs a full symmetric mesh.
    """

    def __init__(self, mesh, mesh_ind, num_twist):
        super(GeometryMesh, self).__init__()

        self.num_twist = num_twist

        self.ny = mesh_ind[0, 1]
        self.nx = mesh_ind[0, 0]
        self.n = self.nx * self.ny
        self.wing_mesh = mesh[:self.n, :].reshape(self.nx, self.ny, 3).astype('complex')

        self.add_param('span', val=58.7630524)
        self.add_param('sweep', val=0.)
        self.add_param('dihedral', val=0.)
        self.add_param('twist', val=numpy.zeros(num_twist))
        self.add_param('taper', val=1.)
        self.add_output('mesh', val=mesh)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        jac = get_bspline_mtx(self.num_twist, self.ny, self.wing_mesh)
        h_cp = params['twist']
        h = jac.dot(h_cp)

        stretch(self.wing_mesh, params['span'])
        sweep(self.wing_mesh, params['sweep'])
        rotate(self.wing_mesh, h)
        dihedral(self.wing_mesh, params['dihedral'])
        taper(self.wing_mesh, params['taper'])
        unknowns['mesh'][:self.n, :] = self.wing_mesh.reshape(self.n, 3)


class LinearInterp(Component):
    """ Linear interpolation used to create linearly varying parameters """

    def __init__(self, num_y, name):
        super(LinearInterp, self).__init__()

        self.add_param('linear_'+name, val=numpy.zeros(2))
        self.add_output(name, val=numpy.zeros(num_y))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

        self.num_y = num_y
        self.vname = name

    def solve_nonlinear(self, params, unknowns, resids):
        a, b = params['linear_'+self.vname]

        if self.num_y % 2 == 0:
            imax = int(self.num_y/2)
        else:
            imax = int((self.num_y+1)/2)
        for ind in xrange(imax):
            w = 1.0*ind/(imax-1)

            unknowns[self.vname][ind] = a*(1-w) + b*w
            unknowns[self.vname][-1-ind] = a*(1-w) + b*w

if __name__ == "__main__":
    """ Test mesh generation and view results in .html file """

    import plotly.offline as plt
    import plotly.graph_objs as go

    from plot_tools import wire_mesh, build_layout

    thetas = numpy.zeros(20)
    thetas[10:] += 10

    mesh = gen_crm_mesh(3, 3)

    # new_mesh = rotate(mesh, thetas)

    # new_mesh = sweep(mesh, 20)

    new_mesh = stretch(mesh, 100)

    # wireframe_orig = wire_mesh(mesh)
    wireframe_new = wire_mesh(new_mesh)
    layout = build_layout()

    fig = go.Figure(data=wireframe_new, layout=layout)
    plt.plot(fig, filename="wing_3d.html")
