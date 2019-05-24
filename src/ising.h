#ifndef ISING_HEADER
#define ISING_HEADER

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "ising-param.h"

// Local size might need adjustment for different platforms
#define global_2D_size (size_t[]){sizeX,sizeY}
#define local_2D_size (size_t[]){16,16}
#define local_1D_size (size_t[]){256}
#define local_length 256

#define num_plat 4 // total number of OpenCL platforms in system
#define plat_id 1 // manually chosen platform id

// Main simulation structure
#define NUM_KERNEL 9
#define NUM_BUFFER 9
typedef struct ising_ocl
{
	cl_kernel kernel[NUM_KERNEL];
	cl_mem buffer[NUM_BUFFER];
} system_t;

enum buffers_enum {state_b,flipE_b,seeds_b,input_b,outpt_b,iseed_b,neigt_b,
  betas_b,probb_b,count_b};
enum kernels_enum {gen_sys_k,gen_rand_k,sum_neigth_k,get_prob_k,compare_k,
  arb_neigth_k,flip_k,save_state_k,measure1_k,measure2_k,measure3_k,next_iter_k};
typedef struct _kernel_list {enum kernels_enum i; char* s} kernel_list;
typedef struct _buffer_list {enum buffers_enum i; size_t l} buffer_list;

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
