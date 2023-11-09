import numpy as np
from devito.logger import info
from examples.seismic import plot_velocity
from stiffnessoperator import *
from examples.seismic import Receiver
from devito import Operator
from devito import div, grad, diag


def ForwardOperator(v, tau, model, stencil, geometry, rec1,rec2,rec3, space_order=8, save=False, moment="iso", comp="vall", **kwargs):

    """
    Construct method for the forward modelling operator in an anisotropic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    comp : string
        components to return flag:
            "vall" - return vx, vy, vz on rec1, rec2, rec3 repsectively
            "divv" - return the divergence of velocities in three orthogonal direction on rec 1.
            "p" - return the average of peasure on rec1
            "s" - return the average of shear stress on rec1
            "ps" - return the average of peasure on rec1 and the average of shear stress on rec2
            ""
    rec1, rec2, rec3 : rec
        spare receiver objects.
    moment : string
        seismic source moment flag:
            "iso" - isotropic moment source (explosive)
    """
    
    dt = geometry._dt

    src_term = getsrc(v=v, tau=tau, model=model, geometry=geometry, moment = moment, pos = "around", dt = dt)
    # pos is set to be "around" by default to avoid checkerboard effect.

    rec_term = getrec(v, tau, model, geometry, rec1,rec2,rec3 , comp)
    
    return Operator(stencil + src_term + rec_term, name="ForwardAnisoElastic", **kwargs) 
    # This order matters, stencil has to be the 1st one.

def getrec(v, tau, model, geometry, rec1,rec2,rec3, comp='vall'):

    rec_term = []

    if comp == "divv":
        rec_term.append(rec1.interpolate(expr=div(v))) # average velocity along three directions.
    elif comp == "vall" :
        rec_term.append(rec1.interpolate(expr=v[0])) # listen to the x component
        rec_term.append(rec2.interpolate(expr=v[-1])) # listen to the vertical component
        if model.grid.dim == 3 and rec3 :
            rec_term.append(rec3.interpolate(expr=v[1])) # listen to the y component
    elif comp == "p" :
        if model.grid.dim == 3:
            rec_term.append(rec1.interpolate(expr=((tau[0,0] + tau[1,1] + tau[2,2])/3)))
        elif model.grid.dim == 2:
            rec_term.append(rec1.interpolate(expr=((tau[0,0] + tau[1,1])/2)))
    elif comp == "s":
        if model.grid.dim == 3:
            rec_term.append(rec1.interpolate(expr=((tau[0,1] + tau[0,2] + tau[1,2])/3)))
        elif model.grid.dim == 2:
            rec_term.append(rec1.interpolate(expr=((tau[0,1]))))
    elif comp == "ps":
        if model.grid.dim == 3:
            rec_term.append(rec1.interpolate(expr=((tau[0,0] + tau[1,1] + tau[2,2])/3)))
            rec_term.append(rec2.interpolate(expr=((tau[0,1] + tau[0,2] + tau[1,2])/3)))
        elif model.grid.dim == 2:
            rec_term.append(rec1.interpolate(expr=((tau[0,0] + tau[1,1])/2)))
            rec_term.append(rec2.interpolate(expr=((tau[0,1]))))
    else:
        raise TypeError("Wrong objective component provided")
        
    return rec_term
    
    
def getsrc(v, tau, model, geometry, moment, pos, dt):
    
    if model.grid.dim == 3:
        if pos == "center":
            return getsrcterm_3d_center(geometry.src,tau,model.grid,dt,moment)
        elif pos == "around":
            return getsrcterm_3d_around(geometry.src,tau,model.grid,dt,moment)
    elif model.grid.dim == 2:
        if pos == "center":
            return getsrcterm_2d_center(geometry.src,tau,model.grid,dt,moment)
        elif pos == "around":
            return getsrcterm_2d_around(geometry.src,tau,model.grid,dt,moment)
    info('No source injected.')
    return
    
    # in construction e.g. moment = "dc" (double-couple) or moment = "clvd" (compensated linear vector dipole)    

def getsrcterm_2d_center(src, tau, grid, dt, moment):
    if moment == "iso":
        src_xx1 = src.inject(field=tau[0, 0].forward, expr=src * dt)
        src_zz1 = src.inject(field=tau[1, 1].forward, expr=src * dt)
        src_term = [src_xx1, src_zz1]
    if moment == "z":
        src_zz1 = src.inject(field=tau[1, 1].forward, expr=src * dt)
        src_term = [src_zz1]
    return src_term

def getsrcterm_2d_around(src, tau, grid, dt, moment):
    
    [x,y] = grid.dimensions
    if moment == "iso":
        src_xx1 = src.inject(field=tau[0, 0].forward.shift(x,x.spacing), expr=1/8 * src * dt)
        src_xx2 = src.inject(field=tau[0, 0].forward.shift(y,y.spacing), expr=1/8 * src * dt)
        src_xx3 = src.inject(field=tau[0, 0].forward.shift(y,y.spacing).shift(x,x.spacing), expr=1/8 * src * dt)
        src_xx4 = src.inject(field=tau[0, 0].forward.shift(x,-x.spacing), expr=1/8 * src * dt)
        src_xx5 = src.inject(field=tau[0, 0].forward.shift(y,-y.spacing), expr=1/8 * src * dt)
        src_xx6 = src.inject(field=tau[0, 0].forward.shift(y,-y.spacing).shift(x,-x.spacing), expr=1/8 * src * dt)
        src_xx7 = src.inject(field=tau[0, 0].forward.shift(y,y.spacing).shift(x,-x.spacing), expr=1/8 * src * dt)
        src_xx8 = src.inject(field=tau[0, 0].forward.shift(y,-y.spacing).shift(x,x.spacing), expr=1/8 * src * dt)

        src_zz1 = src.inject(field=tau[1, 1].forward.shift(x,x.spacing), expr=1/8 * src * dt)
        src_zz2 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing), expr=1/8 * src * dt)
        src_zz3 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing).shift(x,x.spacing), expr=1/8 * src * dt)
        src_zz4 = src.inject(field=tau[1, 1].forward.shift(x,-x.spacing), expr=1/8 * src * dt)
        src_zz5 = src.inject(field=tau[1, 1].forward.shift(y,-y.spacing), expr=1/8 * src * dt)
        src_zz6 = src.inject(field=tau[1, 1].forward.shift(y,-y.spacing).shift(x,-x.spacing), expr=1/8 * src * dt)
        src_zz7 = src.inject(field=tau[1, 1].forward.shift(y,-y.spacing).shift(x,x.spacing), expr=1/8 * src * dt)
        src_zz8 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing).shift(x,-x.spacing), expr=1/8 * src * dt)

        src_term = [src_xx1, src_xx2, src_xx3, src_xx4, src_xx5, src_xx6, src_xx7, src_xx8,
                src_zz1, src_zz2, src_zz3, src_zz4, src_zz5, src_zz6, src_zz7, src_zz8]
    if moment == "z":
        src_zz1 = src.inject(field=tau[1, 1].forward.shift(x,x.spacing), expr=1/8 * src * dt)
        src_zz2 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing), expr=1/8 * src * dt)
        src_zz3 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing).shift(x,x.spacing), expr=1/8 * src * dt)
        src_zz4 = src.inject(field=tau[1, 1].forward.shift(x,-x.spacing), expr=1/8 * src * dt)
        src_zz5 = src.inject(field=tau[1, 1].forward.shift(y,-y.spacing), expr=1/8 * src * dt)
        src_zz6 = src.inject(field=tau[1, 1].forward.shift(y,-y.spacing).shift(x,-x.spacing), expr=1/8 * src * dt)
        src_zz7 = src.inject(field=tau[1, 1].forward.shift(y,-y.spacing).shift(x,x.spacing), expr=1/8 * src * dt)
        src_zz8 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing).shift(x,-x.spacing), expr=1/8 * src * dt)

        src_term = [src_zz1, src_zz2, src_zz3, src_zz4, src_zz5, src_zz6, src_zz7, src_zz8]
    
    return src_term
    
    
def getsrcterm_3d_center(src, tau, grid, dt, moment):
    [x,y,z] = grid.dimensions
    src_term = []
    if moment == "iso":
        src_term += src.inject(field=tau[0, 0].forward, expr= src * dt)
        src_term += src.inject(field=tau[1, 1].forward, expr= src * dt)
        src_term += src.inject(field=tau[2, 2].forward, expr= src * dt)
        
    elif moment == "z":
        src_term += src.inject(field=tau[2, 2].forward, expr= src * dt)
        
    return src_term

def getsrcterm_3d_around(src, tau, grid, dt, moment):
    [x,y,z] = grid.dimensions
        
    if moment == "iso":
        src_xx1 = src.inject(field=tau[0, 0].forward, expr= 1/8 *src * dt)
        src_xx2 = src.inject(field=tau[0, 0].forward.shift(x,x.spacing), expr=1/8*src * dt)
        src_xx3 = src.inject(field=tau[0, 0].forward.shift(y,y.spacing), expr=1/8*src * dt)
        src_xx4 = src.inject(field=tau[0, 0].forward.shift(z,z.spacing), expr=1/8*src * dt)
        src_xx5 = src.inject(field=tau[0, 0].forward.shift(x,x.spacing).shift(y,y.spacing), expr=1/8*src * dt)
        src_xx6 = src.inject(field=tau[0, 0].forward.shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_xx7 = src.inject(field=tau[0, 0].forward.shift(x,x.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_xx8 = src.inject(field=tau[0, 0].forward.shift(x,x.spacing).shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)

        src_yy1 = src.inject(field=tau[1, 1].forward, expr= 1/8 *src * dt)
        src_yy2 = src.inject(field=tau[1, 1].forward.shift(x,x.spacing), expr=1/8*src * dt)
        src_yy3 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing), expr=1/8*src * dt)
        src_yy4 = src.inject(field=tau[1, 1].forward.shift(z,z.spacing), expr=1/8*src * dt)
        src_yy5 = src.inject(field=tau[1, 1].forward.shift(x,x.spacing).shift(y,y.spacing), expr=1/8*src * dt)
        src_yy6 = src.inject(field=tau[1, 1].forward.shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_yy7 = src.inject(field=tau[1, 1].forward.shift(x,x.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_yy8 = src.inject(field=tau[1, 1].forward.shift(x,x.spacing).shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)

        src_zz1 = src.inject(field=tau[2, 2].forward, expr= 1/8 *src * dt)
        src_zz2 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing), expr=1/8*src * dt)
        src_zz3 = src.inject(field=tau[2, 2].forward.shift(y,y.spacing), expr=1/8*src * dt)
        src_zz4 = src.inject(field=tau[2, 2].forward.shift(z,z.spacing), expr=1/8*src * dt)
        src_zz5 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing).shift(y,y.spacing), expr=1/8*src * dt)
        src_zz6 = src.inject(field=tau[2, 2].forward.shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_zz7 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_zz8 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing).shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)

        src_term = [src_xx1,src_xx2,src_xx3,src_xx4,src_xx5,src_xx6,src_xx7,src_xx8,
                   src_yy1,src_yy2,src_yy3,src_yy4,src_yy5,src_yy6,src_yy7,src_yy8,
                   src_zz1,src_zz2,src_zz3,src_zz4,src_zz5,src_zz6,src_zz7,src_zz8]
        
    elif moment == "z":
        src_zz1 = src.inject(field=tau[2, 2].forward, expr= 1/8 *src * dt)
        src_zz2 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing), expr=1/8*src * dt)
        src_zz3 = src.inject(field=tau[2, 2].forward.shift(y,y.spacing), expr=1/8*src * dt)
        src_zz4 = src.inject(field=tau[2, 2].forward.shift(z,z.spacing), expr=1/8*src * dt)
        src_zz5 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing).shift(y,y.spacing), expr=1/8*src * dt)
        src_zz6 = src.inject(field=tau[2, 2].forward.shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_zz7 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing).shift(z,z.spacing), expr=1/8*src * dt)
        src_zz8 = src.inject(field=tau[2, 2].forward.shift(x,x.spacing).shift(y,y.spacing).shift(z,z.spacing), expr=1/8*src * dt)

        src_term = [src_zz1,src_zz2,src_zz3,src_zz4,src_zz5,src_zz6,src_zz7,src_zz8]

    return src_term