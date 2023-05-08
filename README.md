# Ray-tracing-and-GA-for-crack-imaging

Lambs waves are widely adopted for defect detection and characterization in plate-like structures based on the scattering effect of defects on the Lamb waves. Here provides the demo for an efficient Lamb wave-based full-waveform inversion algorithm for crack imaging in plates, which adopts a ray-tracing algorithm as the forward model and the GA as the inverse method. Here, the algorithm is demonstrated by simulation. 'Rough_crack.mph' is the COMSOL FEM model for generating demonstration cases.
Please run the demo by the following steps:

0. An available GA package (https://github.com/rmsolgi/geneticalgorithm) is used in this code. Please import it before running.
1. Run the 'Dictionary_sim.py' to generate a dictionary of scattering fields for all possible point scatterers.
2. Open 'GA_demo.py' and modify the 'crack_name' to select a crack in FEM simulation to be characterized.
3. Run 'GA_demo.py' to obtain the imaging results.

