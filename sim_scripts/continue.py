###############
### Modules ###
###############

import hoomd
from hoomd import md
import datetime
import coxeter as cx
import os
import numpy as np
import random
import gsd.hoomd

# script to continue an old run with existing gsd file.

###############
###############

#####################################
### Read in initial configuration ###
#####################################

# define bond length, simulation steps, random number seed, old file name, output file name.
path = os.path.dirname(os.path.abspath(__file__))
path = path + "/"
bondlength = 25.0
continue_filename = "_chiral_a_" + str(bondlength)        #################################
write_filename = continue_filename + "_one"     #################################
chain_length = 200
random_number_seed = 5065
start_seg_index = 192
each_step_length = 2E5
start_type_5gon = 4

# Read bond configuration 
file_name = path + 'bond_cfg.txt'
f = open(file_name,'r')
message = f.readlines()
bond_config = []
bond_type_config = []
for line in message:

  # Read file
  tmp_line = line
  bond_type,bond_i,bond_j = tmp_line.split()
  bond_tmp = [int(bond_i),int(bond_j)]  

  # Add
  bond_config.append(bond_tmp)
  bond_type_config.append(bond_type)

# Close file
f.close()

# read angle configuration file
file_name = path + 'angle_cfg.txt'
f = open(file_name, 'r')
message = f.readlines()
angle_config = []
angle_type_config = []
for line in message:

  # Read file
  tmp_line = line
  angle_type,angle_i,angle_j,angle_k, Theta_value = tmp_line.split()
  angle_tmp = [int(angle_i),int(angle_j),int(angle_k)]  

  # Add
  angle_config.append(angle_tmp)
  angle_type_config.append(angle_type)

f.close()
Theta_value = float(Theta_value)

# Read pentagonal prism information
file_name = path + '5_gon.txt'
f = open(file_name,'r')
message = f.readlines()
vertices_Aup = []
for line in message:

  # Read file
  tmp_line = line
  x_tmp,y_tmp,z_tmp, insphere_A,sigma_prism_A,v_tmp_A,Ixx_A,Iyy_A,Izz_A,Rratio_A = tmp_line.split()
  pts_tmp = [float(x_tmp),float(y_tmp),float(z_tmp)]  

  # Add
  vertices_Aup.append(pts_tmp)  

# Close file
f.close()
insphere_A_d = float(insphere_A) * 2.0
# Volume
v_A = float(v_tmp_A)
# bonding site monomer radius calculation
#bonding_particle_diameter = 2.0*np.cbrt(3.0*v_A/4.0/np.pi)

# Circumsphere / Insphere ratios
insphere_A = float(insphere_A)
Rratio_A = float(Rratio_A)

# Moment of inertia
Ixx_A = float(Ixx_A)
Iyy_A = float(Iyy_A)
Izz_A = float(Izz_A)

##Read rigid body information
file_name = path + 'rigid_ghost_up.txt'
f = open(file_name,'r')
message = f.readlines()
rigid_up = []
rigid_up_type = []
for line in message:

  # Read file
  tmp_line = line
  x_tmp,y_tmp,z_tmp = tmp_line.split()
  pts_tmp = [float(x_tmp),float(y_tmp),float(z_tmp)]  

  # Add
  rigid_up.append(pts_tmp)  
  rigid_up_type.append('1')
  
# Close file
f.close()

# Read system information
file_name = path + 'system_information.txt'
f = open(file_name, 'r')
message = f.readlines()
for line in message:
    tmp_line = line
    backbone_particle_diameter,bonding_particle_diameter,Lx,Ly,Lz = tmp_line.split()
f.close()
# tube_diameter = float(tube_diameter)
# tube_height = float(tube_height)
# tube_partcile_diameter = float(tube_partcile_diameter)
backbone_particle_diameter = float(backbone_particle_diameter)
bonding_particle_diameter = float(bonding_particle_diameter)
v_bonding_particle = 4.0/3.0*np.pi*(bonding_particle_diameter/2.0)**3
v_backbone_particle = 4.0/3.0*np.pi*(backbone_particle_diameter/2.0)**3
Lx = float(Lx)
Ly = float(Ly)
Lz = float(Lz)
#print(backbone_particle_diameter, bonding_particle_diameter)
pts_config = []
q_config = []
type_config = []
body_config = []
moment_config = []
mass_config = []
diameter_config = []
tag_ID = []

# Read initial configuration
file_name = path + 'polymer_cfg_diff_types.txt'
f = open(file_name,'r')
message = f.readlines()
for line in message:

  # Read file
  tmp_line = line
  type_tmp,body_tmp,x_tmp,y_tmp,z_tmp,qs,qx,qy,qz = tmp_line.split()
  pts_tmp = [float(x_tmp),float(y_tmp),float(z_tmp)]    
  q_tmp = [float(qs),float(qx),float(qy),float(qz)]

  # Add
  pts_config.append(pts_tmp)
  type_config.append(int(type_tmp))
  body_config.append(int(body_tmp))
  q_config.append(q_tmp)
  moment_factor = 1E-6
  if int(type_tmp) >= start_type_5gon:  
    monomer_add = [moment_factor*Ixx_A,moment_factor*Iyy_A,moment_factor*Izz_A]
    moment_config.append(monomer_add)
    mass_config.append(1.0)  
    diameter_config.append(0.0)
    tag_ID.append(str(body_tmp))
  elif int(type_tmp) == 0:
    moment_add = [1.0,1.0,1.0]
    moment_config.append(moment_add)
    mass_config.append(1.0)      
    diameter_config.append(float(backbone_particle_diameter))
    tag_ID.append(str(body_tmp))    
  else:
    moment_add = [0.0,0.0,0.0]
    moment_config.append(moment_add)
    mass_config.append(0.0)      
    diameter_config.append(float(bonding_particle_diameter))
# Close file
f.close()

# Monomer total volume
#v_monomer = float(v_monomer)

# Box size
Lx = float(Lx)*2.0
Ly = float(Ly)*2.0
Lz = float(Lz)*2.0
 
#####################################
#####################################

###################
### Set up sims ###
###################

# Box factor
box_factor = 1

# Define simulation device
cpu = hoomd.device.CPU()

# Create simulation object
sim = hoomd.Simulation(device=cpu, seed=random_number_seed)

######### make snapshot ################
snapshot = gsd.hoomd.Snapshot()

# continue from existing gsd file
name_temp = continue_filename + '.gsd'
f = gsd.hoomd.open(name = path + name_temp, mode = 'rb')
frame = f[len(f)-1]
f.close()
snapshot = frame
# create pentagon tags
type_set = set(type_config)
types_array = []
type_5gon_array = []
for i in range(len(type_set)):
    types_array.append(str(i))
    if i >= start_type_5gon:
        type_5gon_array.append(str(i))

# define rigid body
rigid = md.constrain.Rigid()

for i in range(start_type_5gon,len(type_set),1):
    rigid.body[types_array[i]] = {
    "positions": rigid_up,
    "constituent_types": rigid_up_type,   
    "orientations": [(1.0, 0.0, 0.0, 0.0)],
    "charges": [0.0],
    "diameters": [bonding_particle_diameter]
    }

rigid.body['0'] = {
"positions": [(0.0,0.0, -backbone_particle_diameter/2.0),(0.0,0.0,backbone_particle_diameter/2.0), (backbone_particle_diameter/2.0,0.0,0.0)], #########################################################################
"constituent_types": ['2', '2', '3'],   
"orientations": [(1.0, 0.0, 0.0, 0.0),(1.0, 0.0, 0.0, 0.0),(1.0, 0.0, 0.0, 0.0)],
"charges": [0.0, 0.0, 0.0],
"diameters": [0.0, 0.0, 0.0]           ###########################################################################################################
}   

# create simulation state from simulation
communicator = hoomd.communicator.Communicator()
snapshot_1 = hoomd.Snapshot.from_gsd_snapshot(snapshot, communicator)
#snapshot_1 = snapshot_1.replicate(nx=nrep_xz,ny=nrep_y,nz=nrep_xz)
sim.create_state_from_snapshot(snapshot_1, domain_decomposition=(2, 2, 2))  #(2,2,1)
#sim.state.replicate(nx=nrep_xz,ny=nrep_y,nz=nrep_xz) , domain_decomposition=(2,2,2)

# create rigid bodies
#rigid.create_bodies(sim.state)
# Define current z direction
Lz_current = sim.state.box.Lz
Lx_current = sim.state.box.Lx
Ly_current = sim.state.box.Ly


###################
###################

###############################
### Define potential params ###
###############################

# Neighbor list
nl = md.nlist.Cell(buffer=0.0, exclusions=['body'])

# ALJ parameters
kbt = 1.0
eps_rep = 1.0
eps_att = 1.0

# Define cutoffs
cutoff_rep = 2.0**(1.0/6.0)
cutoff_att = 2.5

# Cutoff buffer
buffer_cutoff_5gon = 1.25 * 3.0
buffer_cutoff_sphere = 1.50
Rratio_A = 2.5
# Cutoffs
coef = 1.0
rcut_5gon = buffer_cutoff_5gon*cutoff_rep*insphere_A*Rratio_A
#rcut_tube_particle = buffer_cutoff_sphere*cutoff_rep*tube_partcile_diameter
rcut_backbone_particle = buffer_cutoff_sphere*cutoff_rep*backbone_particle_diameter
rcut_bonding_particle = buffer_cutoff_sphere*cutoff_rep*bonding_particle_diameter
#rcut_5gon_tube_particle = coef * (rcut_5gon + rcut_tube_particle)
rcut_5gon_backbone_particle = coef*(rcut_5gon + rcut_backbone_particle) 
rcut_5gon_bonding_particle = coef*(rcut_5gon + rcut_bonding_particle) 
#rcut_tube_backbone = coef*(rcut_tube_particle + rcut_backbone_particle) 
#rcut_tube_bonding = coef *(rcut_tube_particle + rcut_bonding_particle) 
rcut_backbone_bonding = coef* (rcut_backbone_particle + rcut_bonding_particle) 
#rcut_sphere = buffer_cutoff*cutoff_rep*r_sphere
#rcut_monomer = buffer_cutoff*cutoff_rep*sigma_monomer

# Mixing diameter (for bond potential)
#sigma_ghost_monomer = 0.5*(sigma_ghost+sigma_monomer)

# Shape Potential
shapeA = cx.shapes.ConvexPolyhedron(np.array(vertices_Aup))
#shapeB = cx.shapes.ConvexPolyhedron(np.array(vertices_Bdown))
# exit()

alj = md.pair.aniso.ALJ(nl, default_r_cut=rcut_5gon, mode='shift')
#alj.params[('A','A')] = dict(epsilon=0,sigma_i=tube_partcile_diameter,sigma_j=tube_partcile_diameter, alpha=0)
#alj.params[('A','B')] = dict(epsilon=eps_rep,sigma_i=tube_partcile_diameter,sigma_j=backbone_particle_diameter, alpha=0)
#alj.params[('A','C')] = dict(epsilon=eps_rep,sigma_i=tube_partcile_diameter,sigma_j=insphere_A_d, alpha=0)
#alj.params[('A','D')] = dict(epsilon=eps_rep,sigma_i=tube_partcile_diameter,sigma_j=bonding_particle_diameter, alpha=0)

alj.params[('0','0')] = dict(epsilon=eps_rep,sigma_i=backbone_particle_diameter,sigma_j=backbone_particle_diameter, alpha=0)
alj.params[('0',type_5gon_array)] = dict(epsilon=eps_rep,sigma_i=backbone_particle_diameter,sigma_j=insphere_A_d, alpha=0)
alj.params[('0','1')] = dict(epsilon=eps_rep,sigma_i=backbone_particle_diameter,sigma_j=bonding_particle_diameter, alpha=0)
alj.params[(type_5gon_array,type_5gon_array)] = dict(epsilon=eps_rep,sigma_i=insphere_A_d,sigma_j=insphere_A_d, alpha=0)
alj.params[(type_5gon_array,'1')] = dict(epsilon=eps_rep,sigma_i=insphere_A_d,sigma_j=bonding_particle_diameter, alpha=0)
alj.params[('1','1')] = dict(epsilon=eps_rep,sigma_i=bonding_particle_diameter,sigma_j=bonding_particle_diameter, alpha=0)
alj.params[(types_array, ['2', '3'])] = dict(epsilon=0,sigma_i=bonding_particle_diameter,sigma_j=bonding_particle_diameter, alpha=0)


#alj.shape["A"] = dict(vertices = [], rounding_radii = tube_partcile_diameter/2.0, faces = [])
alj.shape['0'] = dict(vertices = [], rounding_radii = backbone_particle_diameter/2.0, faces = [])  
alj.shape[type_5gon_array] = dict(vertices = shapeA.vertices, rounding_radii=(0.0,0.0,0.0), faces = shapeA.faces)
alj.shape['1'] = dict(vertices = [], rounding_radii=bonding_particle_diameter/2.0, faces = [])
alj.shape[('2', '3')] = dict(vertices = [], rounding_radii=0.0, faces = [])

# alj.r_cut[('A','A')] = 0.0
# alj.r_cut[('A','B')] = rcut_tube_backbone
# alj.r_cut[('A','C')] = rcut_5gon_tube_particle
# alj.r_cut[('A','D')] = rcut_tube_bonding
alj.r_cut[('0','0')] = rcut_backbone_particle
alj.r_cut[('0',type_5gon_array)] = rcut_5gon_backbone_particle
alj.r_cut[('0','1')] = rcut_backbone_bonding
alj.r_cut[(type_5gon_array,type_5gon_array)] = rcut_5gon
alj.r_cut[(type_5gon_array,'1')] = rcut_5gon_bonding_particle
alj.r_cut[('1','1')] = rcut_bonding_particle
alj.r_cut[(types_array, '2')] = 0.0

if (cpu.communicator.rank == 0):
    print(rcut_5gon, rcut_backbone_particle, rcut_5gon_backbone_particle)
# dipole dipole setup parameter
#type5gon = hoomd.filter.Type('B')

# define dipole-dipole potential
cut_coefficient = 10.05      # define all r_cut explicitly for all pair potential
dipole = md.pair.aniso.Dipole(nl, default_r_cut=Lx/cut_coefficient)
dipole.params[(types_array, types_array)] = dict(A=0.0, kappa=0.0)
#dipole.params[('A', 'C')] = dict(A=20000.0, kappa=0.0)
dipole.params[(type_5gon_array, type_5gon_array)] = dict(A=80000.0, kappa=0.0)
dipole.mu[(types_array)] = (0.0, 0.0, 0.0)
dipole.mu[(type_5gon_array)] = (0.0, 0.0, 20.0)
dipole.r_cut[(types_array, types_array)] = 0.0
# print('a', dipole.mu[(types_array)])
# print('b', dipole.r_cut[(types_array, types_array)])
# print('c', dipole.params[(types_array, types_array)])
#dipole.r_cut[('B', 'B')] = dd_cut_distance       ################ dipole-dipole interaction range

# Define bond potential
kspring_pendent = 60.0
kspring_backbone = 30.0
# kspring_nn = 30.0
# kspring_mg = 30.0
ro = 1.5
ro_gg = 0.8*ro
bond_backback = 1.15*(backbone_particle_diameter/2.0 + backbone_particle_diameter/2.0)
bond_gb = 1.15*(bonding_particle_diameter/2.0 + backbone_particle_diameter/2.0)
fenewca = md.bond.FENEWCA()
harmonic = md.bond.Harmonic()
#coeff = 1.0
ro_init_BB = backbone_particle_diameter
ro_init_GB = bonding_particle_diameter/2
harmonic.params['back-back'] = dict(k=kspring_backbone, r0=ro_init_BB)
harmonic.params['ghost-back'] = dict(k=kspring_pendent, r0=ro_init_GB)
#fenewca.params['ghost-back'] = dict(k=kspring_pendent, r0=ro_init_GB*bond_gb, sigma=bond_gb, epsilon= eps_rep, delta = 0.0)

# define yukawa potential
yukawa = md.pair.Yukawa(default_r_cut=0.0, nlist=nl)
yukawa.params[(types_array, types_array)] = dict(epsilon=-5000.0, kappa=0.0)
yukawa.r_cut[(types_array, types_array)] = 0.0

# angle potential setup
angleharmonic = md.angle.Harmonic()
angleharmonic.params['bbb'] = dict(k=5000, t0=Theta_value)

# dihedralOPLS = md.dihedral.OPLS()
# dihedralOPLS.params['abcd'] = dict(k1=10.0, k2=0.0, k3=0.0, k4=0.0)

# if (cpu.communicator.rank == 0):
    # print(V_total, Lfinal, rcut_5gon, Lfinal/cut_coefficient, sim.state.box.Lz)                           

integrator = hoomd.md.Integrator(dt=2.0E-3, integrate_rotational_dof=True)
sim.operations.integrator = integrator
integrator.rigid = rigid


# Group
all = hoomd.filter.All()
rigid_centers = hoomd.filter.Rigid(flags=('center',))
#anchor_point = hoomd.filter.Tags(['0'])
#moving_set = hoomd.filter.SetDifference(rigid_centers, anchor_point)
#backbone_P = hoomd.filter.Type('A')
#union = hoomd.filter.Union(rigid_centers, backbone_P)
null = hoomd.filter.Null()

# set up the system
kBT_bond = 1.0
nvt = hoomd.md.methods.NVT(filter=rigid_centers, kT=kBT_bond, tau=0.1)
integrator.methods.append(nvt)
#integrator.forces.append(fenewca)       ##########################################
integrator.forces.append(harmonic)
integrator.forces.append(alj) 
integrator.forces.append(angleharmonic) 
# integrator.forces.append(dihedralOPLS)  
#integrator.forces.append(yukawa)                        
# Assign to simulation
sim.operations.integrator = integrator

#print(rigid.body['A'])
# Thermalize - assigns random values of momentum, velocity, and angular momentum consistent with ensemble properties
sim.state.thermalize_particle_momenta(filter=rigid_centers, kT=kBT_bond)

ndump = 20000
ndump_gsd = ndump

# Define the writer to GSD output
logger_shape = hoomd.logging.Logger()
logger_shape.add(alj,quantities=['type_shapes'])
gsd_writer = hoomd.write.GSD(filename=path + write_filename  + '.gsd',
                             trigger=hoomd.trigger.Periodic(int(ndump_gsd)),
                            mode='xb',filter = all,
                            log=logger_shape)
sim.operations.writers.append(gsd_writer)

#output initial cfg
hoomd.write.GSD.write(state=sim.state, mode='wb', filename=path + 'initial.gsd')

###################  test part  #######################
# N_mono = len(tag_ID)/2
# for i in range(5):
    # tag_array = tag_ID[0:(i+1)*10*2]
    # moving_set = hoomd.filter.Tags(tag_array)
    # print('aaaa', tag_array)
##########################################         
         
# ro_space_array_BB = np.linspace(ro_init_BB,bondlength,int(100))
# ro_space_array_GB = np.linspace(ro_init_GB,10,int(100))
# for i in range(0,len(ro_space_array_BB)):
    # if (cpu.communicator.rank == 0):
        # print('Run: ',i, ro_space_array_BB[i], ro_space_array_GB[i])

    # Update bonds
harmonic.params['back-back'] = dict(k=kspring_backbone, r0=bondlength)
harmonic.params['ghost-back'] = dict(k=kspring_pendent, r0=ro_init_GB)
    #fenewca.params['ghost-back'] = dict(k=kspring_pendent, r0=ro_space_array_GB[i]*bond_gb, sigma=bond_gb, epsilon= eps_rep, delta = 0.0)

## Define logger for tps printing ###
########################################################
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=rigid_centers)
sim.operations.computes.append(thermodynamic_properties)
logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps'])
logger.add(thermodynamic_properties, quantities = ['kinetic_temperature'])
class Status():
    def __init__(self, sim):
        self.sim = sim
    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0
    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))
status = Status(sim)
logger[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=int(ndump)),
                          logger=logger)
sim.operations.writers.append(table)

integrator.forces.append(dipole)
integrator.forces.append(yukawa)

# attraction is activated sequentially 
for i in range(start_seg_index,chain_length,1):              # start,stop # 5 5gons: (0,10,1) ; 1 5gon: (3,50,1); 2 5gons: (3,25,1) #
    # zero momentum, and renormalize momentum
    zero_trigger = hoomd.trigger.On(sim.timestep)
    hoomd.md.update.ZeroMomentum(zero_trigger)
    dipole.r_cut[(types_array, types_array)] = 0.0
    dipole.r_cut[(type_5gon_array[0:i+1], type_5gon_array[0:i+1])] = backbone_particle_diameter*5      #slicing does not count for the stop value  
    yukawa.r_cut[(types_array, types_array)] = 0.0
    yukawa.r_cut[(type_5gon_array[i-1:i+1], type_5gon_array[i-1:i+1])] = backbone_particle_diameter*5
    if (cpu.communicator.rank == 0):
        print('bbbb', type_5gon_array[i-1:i+1])    
    integrator.methods.remove(nvt)
    nvt = hoomd.md.methods.NVT(filter=rigid_centers, kT=kbt, tau=0.1)
    integrator.methods.append(nvt)
    sim.state.thermalize_particle_momenta(filter=rigid_centers, kT=kbt)    
    sim.run(each_step_length)    
# equilibrate the system after the run    
zero_trigger = hoomd.trigger.On(sim.timestep)
hoomd.md.update.ZeroMomentum(zero_trigger)
# temp_index = tag_backbone[chain_length-1]            # 5 5gons: *5 ; 1 5gon: *1; 2 5gons: *2 #
# tag_array = tag_ID[0:temp_index+1]              ###########################################  
dipole.r_cut[(type_5gon_array[0:chain_length], type_5gon_array[0:chain_length])] = backbone_particle_diameter*5      #slicing does not count for the stop value
# moving_set = hoomd.filter.Tags(tag_array)
# print('aaaa', tag_array)
integrator.methods.remove(nvt)
nvt = hoomd.md.methods.NVT(filter=rigid_centers, kT=kbt, tau=0.1)
integrator.methods.append(nvt)
sim.state.thermalize_particle_momenta(filter=rigid_centers, kT=kbt)
# if (i == 5) :
    # sim.run(1E6)      # change according to 5gon # each time
# else:
sim.run(1E6)    
    

