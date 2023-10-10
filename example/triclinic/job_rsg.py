import sys
from os.path import dirname
sys.path.append('/home/zhan3386/rsg')

from wavesolver import AnisotropySolver

# NBVAL_IGNORE_OUTPUT
# Adding ignore due to (probably an np notebook magic) bug
import numpy as np
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from sympy import init_printing, latex
from stiffnessoperator import *
from model import AnisoSeismicModel
from devito import info
from examples.seismic.utils import AcquisitionGeometry
import pickle

init_printing(use_latex='mathjax')

import matplotlib.pyplot as plt
# //////////////////////////////////////////////////////////////////////////////
# NBVAL_IGNORE_OUTPUT
from examples.seismic import plot_velocity

so = 8 # cannot be too high will cause the result to be "nan"
to = 2

# Define a physical size
shape = (501, 501, 501)  # Number of grid point (nx, nz)
# Grid spacing in m. The domain size is now 5km by 5km by 5km
spacing = (5., 5., 5.)
nbl = 20

dtype = np.float32

# What is the location of the top left corner. This is necessary to define
origin = (0., 0., 0.)
# the absolute location of the source and receivers

# //////////////////////////////////////////////////////////////////////////////


vp = 2.4
vs = 0.9
density = 1.0
buo = np.ones(shape) * 1 / density

# Triclinic with scale
scale = 1
stiffnessMat = StiffnessMatrix(C11=scale * 10.0, C12=scale*3.5, C13=scale*2.5, C14=scale *-5.0, C15=scale*0.1, C16=scale*0.3, 
C22=scale*8.0, C23=scale*1.5, C24=scale*0.2, C25 = scale *-0.1, C26 = scale*-0.15, C33=scale*6.0, C34 = scale*1.0, C35 = scale*0.4, C36 =scale* 0.24, C44 = scale*5.0, C45=scale*0.35, C46=scale*0.525, C55=scale*4.0, C56=scale*-1.0, C66=scale*3.0)


c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66 = stiffnessMat.getArray(shape)

model = AnisoSeismicModel(space_order=so, vp=vp, b = buo, origin=origin, shape=shape,
                     spacing=spacing, nbl=nbl, c11=c11,c12=c12,c13=c13,c14=c14,c15=c15,c16=c16,
                                                      c22=c22,c23=c23,c24=c24,c25=c25,c26=c26,
                                                              c33=c33,c34=c34,c35=c35,c36=c36,
                                                                      c44=c44,c45=c45,c46=c46,
                                                                              c55=c55,c56=c56,
                                                                                      c66=c66)

t0, tn = 0., 630.  # in ms
dt = 0.5

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.006 #kHz

# specify the source location
src_coordinates = np.array([1250., 1250., 1250.])
rec_coordinates = np.array([[1250, 1250, x] for x in range(100, 2200, 100)])
# NBVAL_SKIP

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0=t0, tn=tn, f0=f0, src_type='Ricker')
geometry._dt = dt

solver = AnisotropySolver(model=model, geometry=geometry, space_order=so,timer_on=True)

info('Space order is : '+str(so))
info('Shape is '+ str(shape))
rec, v, tau, summary = solver.forward(comp = 'vall',moment='z')

print('max value =')
print(v[0].data[0].max())

mid_x = int(0.5 * (v[0].data.shape[1] - 1))
mid_y = int(0.5 * (v[0].data.shape[2] - 1))
mid_z = int(0.5 * (v[0].data.shape[3] - 1))

midzdata =  v[2].data[1, :, :, mid_z]
np.savetxt("vz_data_midz.csv",midzdata,delimiter=",")
vmax = 0.8 * np.max(midzdata)
vmin = -vmax
plt.imsave("midz.png",np.transpose(midzdata),vmin = vmin, vmax= vmax, cmap="seismic")

midxdata =  v[2].data[1, mid_x, :, :]
np.savetxt("vz_data_midx.csv",midxdata,delimiter=",")
vmax = 0.8 * np.max(midxdata)
vmin = -vmax
plt.imsave("midx.png",np.transpose(midxdata),vmin = vmin, vmax= vmax, cmap="seismic")

midxdata =  v[2].data[1, :, mid_y, :]
np.savetxt("vz_data_midy.csv",midxdata,delimiter=",")
vmax = 0.8 * np.max(midxdata)
vmin = -vmax
plt.imsave("midy.png",np.transpose(midxdata),vmin = vmin, vmax= vmax, cmap="seismic")