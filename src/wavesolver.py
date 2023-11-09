from devito.tools import memoized_meth
from devito import VectorTimeFunction, TensorTimeFunction
from devito import info, warning
from devito import NODE
import numpy as np
from sympy import Array
from devito.finite_differences.operators import div, grad
from operators import ForwardOperator
from sympy import Array, tensorproduct, Matrix, transpose, tensorcontraction


class AnisotropySolver(object):
    
        def __init__(self, model, geometry, space_order=4, save=False, timer_on=False,**kwargs):
            """
            Solver object that provides operators for seismic forward modeling
            the wave propagation in general anisotropic media encapsulates the
            time and space discretization for a given problem setup.

            Parameters
            ----------
            model : AnisoSeismicModel
                Physical model with domain parameters.
            geometry : AcquisitionGeometry
                Geometry object that contains the source (SparseTimeFunction) and
                receivers (SparseTimeFunction) and their position.
            space_order : int, optional
                Order of the spatial stencil discretisation. Defaults to 4.
            save : int or Buffer
                Saving flag, True saves all time steps, False saves three buffered
                indices (last three time steps). Defaults to False.
            timer_on : Boolean
                Timer flag, True will info the time spent on setup stencil and modeling.
            """

            self.model = model
            self.geometry = geometry
            if space_order % 2 == 1: # space order has to be even number.
                space_order += 1 
            self.space_order = self.so = space_order 
            self.time_order = self.to = 1
            self.timer_on = timer_on
            self._kwargs = kwargs

            if self.timer_on:
                import time as timer
                set_start = timer.time()
                
            if model.grid.dim == 3:
                [x,y,z] = model.grid.dimensions
                self.v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=1,save=geometry.nt if save else None, staggered=((x,y,z),(x,y,z),(x,y,z)))
                self.tau = TensorTimeFunction(name='t', grid=model.grid, space_order=self.space_order, time_order=1,save=geometry.nt if save else None, staggered=((NODE,NODE,NODE),(NODE,NODE,NODE),(NODE,NODE,NODE)))
            elif model.grid.dim == 2:
                [x,y] = model.grid.dimensions
                self.v = VectorTimeFunction(name='v', grid=model.grid, space_order=space_order, time_order=1,save=geometry.nt if save else None, staggered=((x,y),(x,y)))
                self.tau = TensorTimeFunction(name='t', grid=model.grid, space_order=space_order, time_order=1,save=geometry.nt if save else None, staggered=((NODE,NODE),(NODE,NODE)))

            self.stencil = self.getStencil()
            
            if self.timer_on:
                set_end = timer.time()
                info('Take '+ str(set_end - set_start)+ ' (sec) to get stencil.')
                
        @property
        def dt(self):
            return self.geometry._dt
            
        def forward(self, src=None, rec1=None, rec2=None, rec3 = None, v=None, tau=None, model=None, save=None, moment = "iso", comp = "vall", **kwargs):

            rec1 = rec1 or self.geometry.new_rec(name='rec1')
            rec2 = rec2 or self.geometry.new_rec(name='rec2')
            rec3 = rec3 or self.geometry.new_rec(name='rec3')
            
            save_t = src.nt if save else None

            model = self.model

            if self.timer_on:
                import time as timer
                set_start = timer.time()
            
            # Execute operator and return wavefield and receiver data
            summary = (self.op_fwd(save=save,moment=moment,comp = comp,rec1=rec1, rec2=rec2, rec3=rec3)).apply(dt=kwargs.pop('dt', self.dt), **kwargs)
            
            # handle the case when three receivers are needed (e.g. three components)
            rec = [rec1, rec2, rec3]
            
            if self.timer_on:
                set_end = timer.time()
                info('Take '+ str(set_end - set_start)+ ' (sec) to compute.')
            return rec, self.v, self.tau, summary
        
        @memoized_meth
        def op_fwd(self, moment, comp,rec1,rec2,rec3 ,save=None):
            """Cached operator for forward runs with buffered wavefield"""
            return ForwardOperator(v=self.v, tau=self.tau, model=self.model, stencil = self.stencil, geometry=self.geometry, rec1=rec1,rec2=rec2,rec3=rec3, space_order=self.space_order, save=save, moment=moment, comp = comp, **self._kwargs)
        
        def getStencil(self):
            from devito.types.equation import Eq
            from sympy import Array, tensorproduct, Matrix, transpose, tensorcontraction

            if self.model.grid.dim == 3:
                strain_matrix = getstrainmatrix_3d(self.v, self.model.grid, self.so)
                divtau = getdivtau_3d(self.tau, self.model.grid, self.so)

                elastic_66_matrix = self.model.get2Dstiffness()
                C_mul_strain = tensorproduct(elastic_66_matrix, strain_matrix) # shape-> (6,6,6)
                sec_contract = tensorcontraction(C_mul_strain, (1, 2))  # shape-> (6,1)

                C_mul_strain_expr = getCmulstrain_3d(sec_contract) # shape (6,1) -> (3,3)

            elif self.model.grid.dim == 2:
                strain_matrix = getstrainmatrix_2d(self.v, self.model.grid, self.so)
                divtau = getdivtau_2d(self.tau, self.model.grid, self.so)
                
                elastic_66_matrix = self.model.get2Dstiffness()
                elastic_33_matrix = Array([[elastic_66_matrix[0,0], elastic_66_matrix[0,2], elastic_66_matrix[0,4]],
                                          [elastic_66_matrix[2,0], elastic_66_matrix[2,2], elastic_66_matrix[2,4]],
                                          [elastic_66_matrix[4,0], elastic_66_matrix[4,2], elastic_66_matrix[4,4]]])# 6 by 6 stiffness need to be 3 by 3 stiffness now. 

                C_mul_strain = tensorproduct(elastic_33_matrix, strain_matrix) # shape-> (3,3) * (3 * 1) 
                sec_contract = tensorcontraction(C_mul_strain, (1, 2))  # shape-> (3,3) * (3 * 1)  -> (3,1)

                C_mul_strain_expr = getCmulstrain_2d(sec_contract) # shape (3,1) -> (2,2)
            u_v = Eq(self.v.forward,  self.model.damp * (self.v + self.dt * self.model.b * divtau))
            u_t = Eq(self.tau.forward,  self.model.damp * (self.tau + self.dt * C_mul_strain_expr))
            
            return [u_v] + [u_t]
              
def getstrainmatrix_3d(v, grid, so):
    [x,y,z] = grid.dimensions
    from devito.finite_differences.operators import grad, div
    from sympy import Array
    
    vx_d1 = dv_3d(v.forward[0], grid, 1, so).evalf()
    vx_d2 = dv_3d(v.forward[0], grid, 2, so).evalf()
    vx_d3 = dv_3d(v.forward[0], grid, 3, so).evalf()
    vx_d4 = dv_3d(v.forward[0], grid, 4, so).evalf()
    
    vx_dx = 1 / (4 * x.spacing) * (vx_d1 + vx_d2 + vx_d3 + vx_d4)
    vx_dy = 1 / (4 * y.spacing) * (vx_d1 + vx_d2 - vx_d3 - vx_d4)
    vx_dz = 1 / (4 * z.spacing) * (vx_d1 - vx_d2 + vx_d3 - vx_d4)
    
    vy_d1 = dv_3d(v.forward[1], grid, 1, so).evalf()
    vy_d2 = dv_3d(v.forward[1], grid, 2, so).evalf()
    vy_d3 = dv_3d(v.forward[1], grid, 3, so).evalf()
    vy_d4 = dv_3d(v.forward[1], grid, 4, so).evalf()

    vy_dx = 1 / (4 * x.spacing) * (vy_d1 + vy_d2 + vy_d3 + vy_d4)
    vy_dy = 1 / (4 * y.spacing) * (vy_d1 + vy_d2 - vy_d3 - vy_d4)
    vy_dz = 1 / (4 * z.spacing) * (vy_d1 - vy_d2 + vy_d3 - vy_d4)
    
    vz_d1 = dv_3d(v.forward[2], grid, 1, so).evalf()
    vz_d2 = dv_3d(v.forward[2], grid, 2, so).evalf()
    vz_d3 = dv_3d(v.forward[2], grid, 3, so).evalf()
    vz_d4 = dv_3d(v.forward[2], grid, 4, so).evalf()

    vz_dx = 1 / (4 * x.spacing) * (vz_d1 + vz_d2 + vz_d3 + vz_d4)
    vz_dy = 1 / (4 * y.spacing) * (vz_d1 + vz_d2 - vz_d3 - vz_d4)
    vz_dz = 1 / (4 * z.spacing) * (vz_d1 - vz_d2 + vz_d3 - vz_d4)
    strain_matrix = v.new_from_mat([vx_dx, vy_dy, vz_dz, vy_dz+vz_dy, vx_dz+vz_dx, vx_dy+vy_dx])
    return strain_matrix


def getdivtau_3d(tau, grid, so):
    from devito.finite_differences.operators import div
    [x,y,z] = grid.dimensions
    
    tau_dx =  (1 / (4 * x.spacing) * (dtau_3d(tau[0,0], grid, 1, so) +
                                      dtau_3d(tau[0,0], grid, 2, so) + 
                                      dtau_3d(tau[0,0], grid, 3, so) +
                                      dtau_3d(tau[0,0], grid, 4, so)) +
               1 / (4 * y.spacing) * (dtau_3d(tau[0,1], grid, 1, so) +
                                      dtau_3d(tau[0,1], grid, 2, so) - 
                                      dtau_3d(tau[0,1], grid, 3, so) -
                                      dtau_3d(tau[0,1], grid, 4, so)) + 
               1 / (4 * z.spacing) * (dtau_3d(tau[0,2], grid, 1, so) -
                                      dtau_3d(tau[0,2], grid, 2, so) + 
                                      dtau_3d(tau[0,2], grid, 3, so) -
                                      dtau_3d(tau[0,2], grid, 4, so)))

    tau_dy  = ( 1 / (4 * x.spacing) * (dtau_3d(tau[1,0], grid, 1, so) +
                                       dtau_3d(tau[1,0], grid, 2, so) +
                                       dtau_3d(tau[1,0], grid, 3, so) +
                                       dtau_3d(tau[1,0], grid, 4, so)) +
                1 / (4 * y.spacing) * (dtau_3d(tau[1,1], grid, 1, so) + 
                                       dtau_3d(tau[1,1], grid, 2, so) -
                                       dtau_3d(tau[1,1], grid, 3, so) -
                                       dtau_3d(tau[1,1], grid, 4, so)) +
                1 / (4 * z.spacing) * (dtau_3d(tau[1,2], grid, 1, so) - 
                                       dtau_3d(tau[1,2], grid, 2, so) + 
                                       dtau_3d(tau[1,2], grid, 3, so) -
                                       dtau_3d(tau[1,2], grid, 4, so)))

    tau_dz  = ( 1 / (4 * x.spacing) * (dtau_3d(tau[2,0], grid, 1, so) +
                                       dtau_3d(tau[2,0], grid, 2, so) +
                                       dtau_3d(tau[2,0], grid, 3, so) +
                                       dtau_3d(tau[2,0], grid, 4, so)) +
                1 / (4 * y.spacing) * (dtau_3d(tau[2,1], grid, 1, so) + 
                                       dtau_3d(tau[2,1], grid, 2, so) -
                                       dtau_3d(tau[2,1], grid, 3, so) -
                                       dtau_3d(tau[2,1], grid, 4, so)) +
                1 / (4 * z.spacing) * (dtau_3d(tau[2,2], grid, 1, so) - 
                                       dtau_3d(tau[2,2], grid, 2, so) + 
                                       dtau_3d(tau[2,2], grid, 3, so) -
                                       dtau_3d(tau[2,2], grid, 4, so)))

    divtau = div(tau).replace(div(tau)[0], tau_dx).replace(div(tau)[1], tau_dy).replace(div(tau)[2], tau_dz)
    return divtau



def dv_3d(v_comp, grid, dd, so):
    [x,y,z] = grid.dimensions
    dv_term = 0
    fdcoeff = fdcoeff_1st(so)
    for i in range(so):
        toshift = i - so // 2
        if dd == 1:
            dv_term += fdcoeff[i] * v_comp.shift(x, toshift*x.spacing).shift(y, toshift*y.spacing).shift(z, toshift*z.spacing)
        elif dd == 2:
            dv_term += fdcoeff[i] * v_comp.shift(z, -z.spacing).shift(x, toshift*x.spacing).shift(y, toshift*y.spacing).shift(z, (-1)*toshift*z.spacing)
        elif dd == 3:
            dv_term += fdcoeff[i] * v_comp.shift(y, -y.spacing).shift(x, toshift*x.spacing).shift(y, (-1)*toshift*y.spacing).shift(z, toshift*z.spacing)
        elif dd == 4:
            dv_term += fdcoeff[i] * v_comp.shift(y, -y.spacing).shift(z, -z.spacing).shift(x, toshift*x.spacing).shift(y, (-1)*toshift*y.spacing).shift(z, (-1)*toshift*z.spacing)
    return dv_term


def dtau_3d(tau_comp, grid, dd, so):
    # dd - m
    [x,y,z] = grid.dimensions
    dtau_term = 0
    fdcoeff = fdcoeff_1st(so)
    for i in range(so):
        toshift = i - so // 2
        if dd == 1:
            dtau_term += fdcoeff[i] * tau_comp.shift(x, x.spacing).shift(y, y.spacing).shift(z, z.spacing).shift(x, toshift*x.spacing).shift(y, toshift*y.spacing).shift(z, toshift*z.spacing)
        elif dd == 2:
            dtau_term += fdcoeff[i] * tau_comp.shift(x, x.spacing).shift(y, y.spacing).shift(x, toshift*x.spacing).shift(y, toshift*y.spacing).shift(z, (-1)*toshift*z.spacing)
        elif dd == 3:
            dtau_term += fdcoeff[i] * tau_comp.shift(x, x.spacing).shift(z, z.spacing).shift(x, toshift*x.spacing).shift(y, (-1)*toshift*y.spacing).shift(z, toshift*z.spacing)
        elif dd == 4:
            dtau_term += fdcoeff[i] * tau_comp.shift(x, x.spacing).shift(x, toshift*x.spacing).shift(y, (-1)*toshift*y.spacing).shift(z, (-1)*toshift*z.spacing)
    return dtau_term




def dv_2d(v_comp, grid, dd, so):
    [x,y] = grid.dimensions
    dv_term = 0                                                                                                                     
    fdcoeff = fdcoeff_1st(so)                                                                                                                     
    for i in range(so):
        toshift = i - so // 2
        if dd == 1:
            dv_term += fdcoeff[i] * v_comp.shift(x, toshift*x.spacing).shift(y, toshift*y.spacing)
        elif dd == 2:
            dv_term += fdcoeff[i] * v_comp.shift(y, -y.spacing).shift(x, toshift*x.spacing).shift(y, (-1)*toshift*y.spacing)
    return dv_term                                                                                                                                     


def dtau_2d(tau_comp, grid, dd, so):
    [x,y] = grid.dimensions
    dtau_term = 0                                                                                                                     
    fdcoeff = fdcoeff_1st(so)                                                                                                                     
    for i in range(so):
        toshift = i - so // 2
        if dd == 1:
            dtau_term += fdcoeff[i] * tau_comp.shift(x, x.spacing).shift(y, y.spacing).shift(x, toshift*x.spacing).shift(y, toshift*y.spacing)
        elif dd == 2:
            dtau_term += fdcoeff[i] * tau_comp.shift(x, x.spacing).shift(x, toshift*x.spacing).shift(y, (-1)*toshift*y.spacing)
    return dtau_term                                                                                                                               
def getdivtau_2d(tau, grid, so):
    # tau at integer index, v at half index
    [x, y] = grid.dimensions
    tau_dx = (1 / (2 * x.spacing) * (dtau_2d(tau[0,0], grid, 1, so)
                                + dtau_2d(tau[0,0], grid, 2, so)) + 
             1 / (2 * y.spacing) * (dtau_2d(tau[0,1], grid, 1, so)
                                - dtau_2d(tau[0,1], grid, 2, so)))
    tau_dz = (1 / (2 * x.spacing) * (dtau_2d(tau[1,0], grid, 1, so)
                                + dtau_2d(tau[1,0], grid, 2, so)) + 
             1 / (2 * y.spacing) * (dtau_2d(tau[1,1], grid, 1, so)
                                - dtau_2d(tau[1,1], grid, 2, so)))

    divtau = div(tau).replace(div(tau)[0], tau_dx).replace(div(tau)[1], tau_dz)
    return divtau


def getstrainmatrix_2d(v, grid, so):
    [x, y] = grid.dimensions
    # ////////////////////////////////////////////////////////////////////////////////
    vx_dx = 1 / (2 * x.spacing) * (dv_2d(v.forward[0], grid, 1, so)
                                  + dv_2d(v.forward[0], grid, 2, so))

    vz_dx = 1 / (2 * x.spacing) * (dv_2d(v.forward[1], grid, 1, so)
                                  + dv_2d(v.forward[1], grid, 2, so))

    vx_dz = 1 / (2 * y.spacing) * (dv_2d(v.forward[0], grid, 1, so)
                                  - dv_2d(v.forward[0], grid, 2, so))

    vz_dz = 1 / (2 * y.spacing) * (dv_2d(v.forward[1], grid, 1, so)
                                  - dv_2d(v.forward[1], grid, 2, so))

    grad_v_forward = v.new_from_mat([vx_dx, vz_dz, vx_dz + vz_dx])
    return grad_v_forward

def fdcoeff_1st(order):
    '''
    Return a list of coefficient at corresponding location in the given spatial order.
    '''
    s = np.arange(order) - order//2 + 0.5
    s_mat = np.empty((order,order))
    for i in range(order):
        s_row = s ** i
        s_mat[i,:] = s_row

    vector_order = np.zeros(order)
    vector_order[1] = 1 # because on 1st order derivative is calculated

    return np.linalg.inv(s_mat).dot(vector_order.transpose())


def getCmulstrain_3d(sec_contract):
    return Matrix(3,3, [sec_contract[0,0], sec_contract[5,0], sec_contract[4,0], 
                        sec_contract[5,0], sec_contract[1,0], sec_contract[3,0],
                        sec_contract[4,0], sec_contract[3,0], sec_contract[2,0]])

def getCmulstrain_2d(sec_contract):
    return Matrix(2,2, [sec_contract[0,0], sec_contract[2,0], sec_contract[2,0], sec_contract[1,0]])