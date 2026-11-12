SET(TEST_NAME theta-ne30pg2-nlev128-eamxx-kokkos)
SET(EXEC_NAME theta-nlev128-kokkos)
SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/theta-ne30pg2-nlev128-eamxx.nl)
SET(VCOORD_FILES
  ${HOMME_ROOT}/test/vcoord/scream-128i.ascii
  ${HOMME_ROOT}/test/vcoord/scream-128m.ascii)
SET(NC_OUTPUT_FILES)