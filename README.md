# RSG: Rotated Staggered Grid based on Devito

Built upon the open-source Python finite-difference framework, Devito, this RSG elastic wave solver is designed to model seismic wave propagation in intricate anisotropic media.

#### Key Components:

* `model.py`: A modified file derived from Devito's original model.py. It introduces a new class, `AnisoSeismicModel`, tailored to support the input of 21 distinct elastic parameters for each grid.

* `wavesolver.py`: This object furnishes the operator essential for forward modeling computations. It encapsulates both temporal and spatial discretizations pertinent to a specified problem setup.

* `stiffnessoperator.py`: Serving as an operator for the stiffness tensor, this object accommodates isotropy, Vertical Transverse Isotropy (VTI), and 3D rotations. Users can either opt for manual input of the 21 elastic parameters or utilize the automated setup. When paired with a mask, it aids in crafting the `AnisoSeismicModel`.

#### Getting Started:

For a practical introduction and hands-on experience, refer to the tutorial Jupyter notebook located at rsg/example/Tutorial.ipynb. This guide walks users through the application of the package, showcasing wave propagation modeling within a two-layered structure (comprising an upper water layer and a lower TTI media) in 2D.

#### Update:
Nov 9. Support lateast Devito 4.8.3
