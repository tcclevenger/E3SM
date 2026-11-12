&ctl_nl
vthreads          = 1
NThreads          = 1
partmethod        = 4
topology          = 'cube'
geometry          = 'sphere'
test_case         = 'held_suarez0'
u_perturb         = 1
rotate_grid       = 0
ne                = 30
qsize             = 1
ndays             = 1
statefreq         = 9999
restartfreq       = 0
runtype           = 0
mesh_file         = 'none'
tstep             = 300
dt_remap_factor   = 1
dt_tracer_factor  = 6
transport_alg     = 12
integration       = 'explicit'
smooth            = 0
cubed_sphere_map  = 2
disable_diagnostics = false
hv_ref_profiles   = 6
nu                = 3.4e-08
nu_div            = 3.4e-08
nu_p              = -1
nu_q              = -1
nu_s              = -1
nu_top            = 250000.0
tom_sponge_start  = 0.0
laplace_scaling   = 0.0
se_ftype          = 2
limiter_option    = 9
vert_remap_q_alg  = 10
hypervis_scaling  = 3.0
hypervis_order    = 2
hypervis_subcycle = 1
hypervis_subcycle_q = 6
hypervis_subcycle_tom = 1
horiz_turb_subcycle = 1
horiz_turb_subcycle_q = 1
pgrad_correction  = 1
theta_hydrostatic_mode = false
theta_advect_form = 1
tstep_type        = 9
vtheta_thresh     = 100.0
internal_diagnostics_level = 0
semi_lagrange_trajectory_nsubstep = 0
semi_lagrange_trajectory_nvelocity = -1
semi_lagrange_halo = -1
semi_lagrange_diagnostics = 0
do_3d_turbulence  = false
moisture          = 'notdry'
/
&vert_nl
vfile_mid = './vcoord/scream-128m.ascii'
vfile_int = './vcoord/scream-128i.ascii'
/
&analysis_nl
output_frequency = -1
/