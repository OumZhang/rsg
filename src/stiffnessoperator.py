import numpy as np
import math
from devito import info
from numpy.linalg import inv


class StiffnessMatrix:
    """
    Stiffness object supports ISO, VTI and 3d rotation

    Parameters
    ----------
    Cij, (i,j from 1 to 6) : float
        elastic constants in GPa
    mat : array_like (optional)
        array in shape 6*6 with elastic constants at corresponding indices.
    """
    def __init__(self, C11 = 20.37, C12 = 12.3, C13 = 12.3, C14 = 0, C15 = 0, C16 = 0,
                                   C22 = 20.37, C23 = 12.3, C24 = 0, C25 = 0, C26 = 0,
                                                C33 = 20.37, C34 = 0, C35 = 0, C36 = 0,
                                                            C44 = 4.035, C45 = 0, C46 = 0,
                                                                    C55 = 4.035, C56 = 0,
                                                                          C66 = 4.035, mat = None):
        if (mat != None):
            assert mat.shape == (6,6), "mat should be 6 by 6 matrix"
            C11 = mat[0][0]
            C12 = mat[0][1]
            C13 = mat[0][2]
            C14 = mat[0][3]
            C15 = mat[0][4]
            C16 = mat[0][5]
            
            C22 = mat[1][1]
            C23 = mat[1][2]
            C24 = mat[1][3]
            C25 = mat[1][4]
            C26 = mat[1][5]
            
            C33 = mat[2][2]
            C34 = mat[2][3]
            C35 = mat[2][4]
            C36 = mat[2][5]
            
            C44 = mat[3][3]
            C45 = mat[3][4]
            C46 = mat[3][5]
            
            C55 = mat[4][4]
            C56 = mat[4][5]
            
            C66 = mat[5][5]
            
        self.C11 = C11
        self.C12 = C12
        self.C13 = C13
        self.C14 = C14
        self.C15 = C15
        self.C16 = C16
        
        self.C22 = C22
        self.C23 = C23
        self.C24 = C24
        self.C25 = C25
        self.C26 = C26
        
        self.C33 = C33
        self.C34 = C34
        self.C35 = C35
        self.C36 = C36
        
        self.C44 = C44
        self.C45 = C45
        self.C46 = C46
        
        self.C55 = C55
        self.C56 = C56
        
        self.C66 = C66
        
    def getMatrix(self):
        '''
        Return a 6 by 6 array
        '''
        return np.array([[self.C11, self.C12, self.C13, self.C14, self.C15, self.C16],
                         [self.C12, self.C22, self.C23, self.C24, self.C25, self.C26],
                         [self.C13, self.C23, self.C33, self.C34, self.C35, self.C36],
                         [self.C14, self.C24, self.C34, self.C44, self.C45, self.C46],
                         [self.C15, self.C25, self.C35, self.C45, self.C55, self.C56],
                         [self.C16, self.C26, self.C36, self.C46, self.C56, self.C66]])
    def getInverse(self):
        '''
        Return a 6 by 6 compliance matrix
        '''
        return inv(self.getMatrix())
    
    def get4DTensor(self, compl=False, sym = True):
        """
        Return a 3 * 3 * 3 * 3 4-th order stiffness tensor.
        
        Parameters
        ----------
        compl : Boolean
            if True, return the 4th order compliance tensor.
            if False, return the 4th order stiffness tensor.
        sym : Boolean
            if True, return the symbolized array
            if False, return the numerical (float) array
        
        source: https://github.com/andreww/theia_tools/blob/master/tex2elas.py
        
        Convert from Voigt to full tensor notation 
        Convert from the 6*6 elastic constants matrix to 
        the 3*3*3*3 tensor representation. Recoded from 
        the Fortran implementation in DRex. Use the optional 
        argument "compl" for the elastic compliance (not 
        stiffness) tensor to deal with the multiplication 
        of elements needed to keep the Voigt and full 
        notation consistant.
        """
        from sympy import Array
        cij_mat = self.getMatrix()
        cij_tens = np.zeros((3,3,3,3))
        m2t = np.array([[0,5,4],[5,1,3],[4,3,2]])
        if compl:
            cij_mat = cij_mat / np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                          [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                          [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                          [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                          [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                          [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        cij_tens[i,j,k,l] = cij_mat[m2t[i,j],m2t[k,l]]
        if sym:
            return Array(cij_tens)
        else:
            return cij_tens

    def getArray(self, shape):
        """
        Return each elastic constants in array with the given shape. e.g.(201, 201)
        
        Write the stiffnessMatrix into array. The shape should be correspond to the model shape.
        The returned array will be used to generate AnisoSeismicModel, e.g (c11 = c11), where c11 is the output array
        by this method.
        
        To create a realistic geology model with various media, a corresponding mask of this shape is needed.
        """
   
        c11 = np.ones(shape) * self.C11
        c12 = np.ones(shape) * self.C12
        c13 = np.ones(shape) * self.C13
        c14 = np.ones(shape) * self.C14
        c15 = np.ones(shape) * self.C15
        c16 = np.ones(shape) * self.C16
        
        c22 = np.ones(shape) * self.C22
        c23 = np.ones(shape) * self.C23
        c24 = np.ones(shape) * self.C24
        c25 = np.ones(shape) * self.C25
        c26 = np.ones(shape) * self.C26 
        
        c33 = np.ones(shape) * self.C33
        c34 = np.ones(shape) * self.C34
        c35 = np.ones(shape) * self.C35
        c36 = np.ones(shape) * self.C36
        
        c44 = np.ones(shape) * self.C44
        c45 = np.ones(shape) * self.C45
        c46 = np.ones(shape) * self.C46
        
        c55 = np.ones(shape) * self.C55
        c56 = np.ones(shape) * self.C56
        
        c66 = np.ones(shape) * self.C66

        return c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66
    
    def tilt(self, xi=0., eta=0., xiprime = 0., dip = None, strike = None):
        """
        Return a StiffnessMatrixTTI object after rotation.
        
        self - StiffnessMatrix at material coordinate of its own. It will be rotated with the following tilting operation.
        
        Reference:  Auld Page.77-80 
        http://home.agh.edu.pl/~lesniak/papers/mono.pdf
        
        clockwise rotation through an angle _xi_ (radians) about the Z axis.
        clockwise rotation through an angle _eta_ (radians) about the transformed Y axis.
        clockwise rotation through an angle _xiprime_ (radians) about the transformed Z axis.
        
        When applying upon geology, eta corresponds to dip angle (dipping North or x-reverse direction),
            and xi corresponds to the strike (from North or x-reverse direction).
            
        Note: applying a subsequent transformation after another, is equivalent to rotating on the new coordinate
                e.g. rotate_xi along z-axis first, then apply rotate_eta, is equivalent to rotating along transformed y-axis
        
        """
        eta = dip if dip else eta
        xi = strike if strike else xi
        
        M_xi = np.array([[math.cos(xi)**2,   math.sin(xi)**2,      0,      0,       0,          math.sin(2*xi)],
                         [math.sin(xi)**2,   math.cos(xi)**2,      0,      0,       0,          -math.sin(2*xi)],
                         [0,                        0,             1,      0,       0,              0    ],
                         [0,                        0,             0,   math.cos(xi),  -math.sin(xi),         0],
                         [0,                        0,             0,   math.sin(xi),    math.cos(xi),         0],
                         [-(math.sin(2*xi))/2, (math.sin(2*xi))/2, 0,     0,      0,         math.cos(2*xi)]])
        
        
        
        M_eta = np.array([[math.cos(eta)**2,      0,     math.sin(eta)**2,     0,     -math.sin(2*eta),     0],
                          [0,                     1,         0,                0,           0,              0],
                          [math.sin(eta)**2,      0,     math.cos(eta)**2,     0,      math.sin(2*eta),     0],
                          [0,                     0,         0,            math.cos(eta),       0,     math.sin(eta)],
                          [math.sin(2*eta)/2,     0,     -math.sin(2*eta)/2,   0,      math.cos(2*eta),      0],
                          [0,                     0,         0,           -math.sin(eta),      0,     math.cos(eta)]])
        
        M_xiprime = np.array([[math.cos(xiprime)**2,   math.sin(xiprime)**2,     0,      0,                    0,           math.sin(2*xiprime)],
                             [math.sin(xiprime)**2,    math.cos(xiprime)**2,     0,      0,                    0,          -math.sin(2*xiprime)],
                             [0,                                  0,             1,      0,                    0,                         0    ],
                             [0,                                  0,             0,   math.cos(xiprime),  -math.sin(xiprime),             0    ],
                             [0,                                  0,             0,   math.sin(xiprime),    math.cos(xiprime),            0    ],
                             [-(math.sin(2*xiprime))/2, (math.sin(2*xiprime))/2, 0,      0,                    0,           math.cos(2*xiprime)]])

        C_1 = np.matmul(np.matmul(M_xi, self.getMatrix()), M_xi.transpose())
        C_2 = np.matmul(np.matmul(M_eta, C_1), M_eta.transpose())
        C_3 = np.matmul(np.matmul(M_xiprime, C_2), M_xiprime.transpose())
        
        return StiffnessMatrix(C11 = C_3[0,0], C12 = C_3[0,1], C13 = C_3[0,2], C14 = C_3[0,3], C15 = C_3[0,4], C16 = C_3[0,5],
                                                  C22 = C_3[1,1], C23 = C_3[1,2], C24 = C_3[1,3], C25 = C_3[1,4], C26 = C_3[1,5],
                                                                  C33 = C_3[2,2], C34 = C_3[2,3], C35 = C_3[2,4], C36 = C_3[2,5],
                                                                                  C44 = C_3[3,3], C45 = C_3[3,4], C46 = C_3[3,5],
                                                                                                  C55 = C_3[4,4], C56 = C_3[4,5],
                                                                                                                  C66 = C_3[5,5])
    
    
    

class StiffnessMatrixVTI(StiffnessMatrix):
    def __init__(self, vp = 2.0, vs = 1.0, rho = 1.0, epsilon = 0., gamma = 0., delta = 0.):
        C33 = float(vp**2 * rho)
        C44 = float(vs**2 * rho)
        C11 = float(C33 * (2 * epsilon + 1))
        C66 = float(C44 * (2 * gamma + 1))
        
        C22 = float(C11)
        C55 = float(C44)
        
        C12 = float(C11 - 2 * C66)
        C13 = np.sqrt((C33 - C44)*(2.0*C33*delta + C33 - C44)) - C44
        C13 = float(C13)
        C14 = 0.
        C15 = 0.
        C16 = 0.
        
        C23 = float(C13)
        C24 = 0.
        C25 = 0.
        C26 = 0.
        
        C34 = 0.
        C35 = 0.
        C36 = 0.
        
        C45 = 0.
        C46 = 0.
        
        C56 = 0.
        super().__init__(C11 = C11, C12 = C12, C13 = C13, C14 = C14, C15 = C15, C16 = C16,
                                    C22 = C22, C23 = C23, C24 = C24, C25 = C25, C26 = C26,
                                               C33 = C33, C34 = C34, C35 = C35, C36 = C36,
                                                          C44 = C44, C45 = C45, C46 = C46,
                                                                     C55 = C55, C56 = C56,
                                                                                C66 = C66)
            
    def getEpsilon(self):
        return (self.C11 - self.C33) / (2 * self.C33)

    def getGamma(self):
        return (self.C66 - self.C44) / (2 * self.C44)

    def getDelta(self):
        return (((self.C13 + self.C44)**2 - (self.C33 - self.C44)**2)
                /(2 * self.C33 * (self.C33 - self.C44)))
    
class StiffnessMatrixISO(StiffnessMatrix):
    def __init__(self, vp = 4., vs = 2., rho = 1.5):
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.mu = rho * vs**2
        self.l = vp**2 * rho - 2 * self.mu
        C11 = self.l + 2 * self.mu
        C22 = C33 = C11
     
        C12= self.l
        C13 = C23 = C12
        
        C44 = self.mu
        C55 = C66 = C44
        
        C14 = C15 = C16 = C24 = C25 = C26 = C34 = C35 = C36 = C45 = C46 = C56 = 0.
        
        super().__init__(C11 = C11, C12 = C12, C13 = C12, C14 = C14, C15 = C15, C16 = C16,
                                    C22 = C22, C23 = C23, C24 = C24, C25 = C25, C26 = C26,
                                               C33 = C33, C34 = C34, C35 = C35, C36 = C36,
                                                          C44 = C44, C45 = C45, C46 = C46,
                                                                     C55 = C55, C56 = C56,
                                                                                C66 = C66)
###------------------------------------------------------------------------------------------------------------------
class StiffnessMatrixFractured(StiffnessMatrix):
    '''
    To add arbitrary fracture set with stiffness matrix, the rountine is:
            1) create a fracture compliance Sfrac using StiffnessMatrixFractured and method addFracture()
            2) rotate it into the arbitrary direction, using tilt()
            3) have the intact rock stiffness matrix C, and convert it to S by getInverse()
            4) calculate the sum of these two compliance, and create a new StiffnessMatrix with mat=the sum S calculated
            5) convert it back to stiffness C by using getInverse() once more.
    '''
    
    def __init__(self, C11 = 0, C12 = 0, C13 = 0, C14 = 0, C15 = 0, C16 = 0,
                                C22 = 0, C23 = 0, C24 = 0, C25 = 0, C26 = 0,
                                         C33 = 0, C34 = 0, C35 = 0, C36 = 0,
                                                  C44 = 0, C45 = 0, C46 = 0,
                                                           C55 = 0, C56 = 0,
                                                                    C66 = 0):

        super().__init__(C11, C12, C13, C14, C15, C16,
                              C22, C23, C24, C25, C26,
                                   C33, C34, C35, C36,
                                        C44, C45, C46,
                                             C55, C56,
                                                  C66)
 
    def addLSDFracture(self, Z_N = None, Z_T = None):
        # LSD - linear slip deformation. where closure or opening of the fracture will not cause tangential slip (decoupled)
#         create a new set rotationally invariant of fracture. this can be tiled using the "tilt" method.
            return StiffnessMatrixTTI(C11 = Z_N, C12 = 0, C13 = 0, C14 = 0, C15 = 0,   C16 = 0,
                                                 C22 = 0, C23 = 0, C24 = 0, C25 = 0,   C26 = 0,
                                                          C33 = 0, C34 = 0, C35 = 0,   C36 = 0,
                                                                   C44 = 0, C45 = 0,   C46 = 0,
                                                                            C55 = Z_T, C56 = 0,
                                                                                       C66 = Z_T)
    def addGeneralFracture(self, Z_N = None, Z_2 = None, Z_1 = None, Z_N2 = None, Z_N1 = None, Z_12 = None):
        '''
        add a most general fracture system (triclinic anisotropy) to the stiffness matrix.
        The fracture set is horizontal, no matter how the stiffness was rotated previously.
        That all three components of the fracture system slip-strain are coupled to all three components of the traction
        across the fractures.
        Definition from Schoenberg and Douma (1988)
             [Z_N , Z_N2, Z_N1]
        Z =  [Z_N2, Z_2 , Z_12]
             [Z_N1, Z_12, Z_1 ]
        N_b - background sub-stiffness matrix
        '''
        Z = np.array([[Z_N , Z_N2, Z_N1],[Z_N2, Z_2 , Z_12],[Z_N1, Z_12, Z_1]])
        N_b = np.array([[self.C33, self.C34, self.C35], 
                        [self.C34, self.C44, self.C45], 
                        [self.C35, self.C45, self.C55]])
        N_e = np.matmul(N_b, inv(np.identity(3) + np.matmul(Z, N_b))) # Eq (16.) in Schoenberg and Douma (1988)
        
        self.C33 = N_e[0, 0]
        self.C44 = N_e[1, 1]
        self.C55 = N_e[2, 2]
        
        self.C34 = N_e[0, 1]
        self.C35 = N_e[0, 2]
        self.C45 = N_e[1, 2]






