#ifndef ISING_HEADER
#define ISING_HEADER

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "ising-param.h"

#define PROGRAM_FILE "ising.cl"
#define ISING_FUNC "ising_calc"
#define RAND_FUNC "ising_rand"
#define MEAS_FUNC "ising_mag"
#define COUNTER_INCR "counter_incr"

// Local size might need adjustment for different platforms
#define global_2D_size (size_t[]){sizeX,sizeY}
#define local_2D_size (size_t[]){16,16}
#define local_1D_size (size_t[]){256}
#define local_length 256

#define num_plat 4 // total number of OpenCL platforms in system
#define plat_id 3 // manually chosen platform id

// Main simulation structure
#define NUM_KERNEL 6
#define NUM_COUNT 3
typedef struct ising_ocl
{
  cl_kernel kernel[NUM_KERNEL];
  cl_mem state;
  cl_mem rand_buff;
  cl_mem counter[NUM_COUNT];
  cl_mem prob;
  cl_mem output;
  uint prob_num;
} system_t;

// Public function prototypes:
int ising_init(void);
system_t ising_new(void);
int ising_free(system_t *);
int ising_enqueue(system_t *);
int ising_configure(system_t *, state_t *, float);
int ising_configure_betas(system_t *, uint, float*);
int ising_get_states(system_t *, state_t *);
int ising_get_data(system_t *, int *);
void ising_profile(void);

#endif
