# Ray-tracing-and-GA-for-crack-imaging

Here provides the demo for an efficient full-waveform inversion algorithm for crack imaging in plates. Please run the demo by the following steps:

0. An available GA package (https://github.com/rmsolgi/geneticalgorithm) is used in this code. Please import it before running.
1. Run the 'Dictionary_sim.py' to generate a dictionary of scattering fields for all possible point scatterers.
2. Open 'GA_demo.py' and modify the 'crack_name' to select a crack in FEM simulation to be characterized.
3. Run 'GA_demo.py' to obtain the imaging results.

'Rough_crack.mph' is the COMSOL FEM model for generating demonstration cases.
