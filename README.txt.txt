1. gsd_file folder
This folder contains the example snapshots from the simulations, including d_B=90.24&r_0=7, d_B=101.52&r_0=9, d_B=112.8&r_0=13, d_B=124.08&r_0=15.

2. analysis_code folder
This folder contains the code used for the analysis.
	a. curvature.py: The python script to calculate the Menger curvature.
	b. pitch_length&_pitch_angle.py: The python script to calculate the pitch angle and pitch length.
	
3. sim_scripts folder
This folder contains an example HOOMD v3.11 script file along with the initial configuration files to run d_BP=112.8&r_0=25 system.
	chiral.py: script to start a new run.
	continue.py: script to continue an old run.
	5_gon.txt: pentagonal prism information.
	angle_cfg.txt: angle configuration.
	bond_cfg.txt: bond configuration.
	polymer_cfg_diff_types.txt: initial particle configuration.
	rigid_ghost_up.txt: rigid body information (red particled which is attched to the pentagonal prism).
	system_information.txt: system information, including bakcboen particle diameter, red particle diameter and box dimensions.
