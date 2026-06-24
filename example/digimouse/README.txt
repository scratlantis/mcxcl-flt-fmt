== Digimouse Atlas Photon Simulation ==

In this example, we demonstrate light transport simulation in a mouse
atlas template (Digimouse). There are 21 tissue types in the atlas. The volume
is made of 190 x 496 x 104 0.8 mm^3 isotropic voxels. See [Fang2012].

To run this example, please call

 ./run_atlas.sh

or

 ./run_atlas.sh -n 1e6 

to specify a different photon number


The JSON files (.json, .jnii) utlizes the JData specifiation (https://github.com/NeuroJSON/jdata) 
to include binary data with compression support. Please download JSONLab from

https://github.com/fangq/jsonlab

to open these files in MATLAB and GNU Octave, or PyJData from 

https://github.com/fangq/pyjdata

to open such in Python.


=== Reference ===

[Fang2012] Fang Q and Kaeli D, "Accelerating mesh-based Monte Carlo method 
 on modern CPU architectures," Biomed. Opt. Express, 3(12), 3223-3230, 2012

Fluence contribution moments
----------------------------

After rebuilding MCXCL, run a matched fluence and squared-contribution pair:

  python3 run_fluence_moments.py --photons 10000000

Both simulations use the same explicit RNG seed. The second pass uses output
type K and accumulates the square of every track segment's fluence contribution
before adding it to the voxel. Derived NumPy volumes are written under
fluence_moments/, including:

  fluence_contribution_ratio.npy       sum(c_i^2) / sum(c_i)^2
  fluence_effective_contributions.npy  sum(c_i)^2 / sum(c_i^2)

View the ratio with:

  python3 view_jacobian.py fluence_moments/fluence_contribution_ratio.npy
