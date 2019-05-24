#include "ising.h"
#include "ising-param.h"

// -------------- OpenCL "helper"
// This part was copied from some examples found over the Internet, with
// minor modifications

/* Find a GPU or CPU associated with the choosen platform

The `platform` structure identifies the first platform identified by the
OpenCL runtime. A platform identifies a vendor's installation, so a system
may have an NVIDIA platform and an AMD platform.

The `device` structure corresponds to the first accessible device
associated with the platform. Because the second parameter is
`CL_DEVICE_TYPE_GPU`, this device must be a GPU.
*/
cl_device_id
clh_create_device ()
{
	cl_platform_id platform[num_plat];
	cl_device_id dev;
	int err;

	// Identify a platform
	err = clGetPlatformIDs(num_plat, platform, NULL);
	if(err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	// Print platform name for debugging
	char infostr[100];
	clGetPlatformInfo(platform[plat_id],CL_PLATFORM_NAME,100,infostr,NULL);
	fprintf(stderr,"Platform name: %s\n",infostr);

	// Access a device
	// GPU
	err = clGetDeviceIDs(platform[plat_id], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		// CPU
		fprintf(stderr,"USING CPU!\n");
		err = clGetDeviceIDs(platform[plat_id], CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if(err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	return dev;
}

// Create program from a file and compile it
cl_program
clh_build_program (cl_context ctx, cl_device_id dev, const char* filename)
{
	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	// Read program file and place content into buffer
	program_handle = fopen(filename, "r");
	if(program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file

	Creates a program from the source code in the add_numbers.cl file.
	Specifically, the code reads the file's content into a char array
	called program_buffer, and then calls clCreateProgramWithSource.
	*/
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if(err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program

	The fourth parameter accepts options that configure the compilation.
	These are similar to the flags used by gcc. For example, you can
	define a macro with the option -DMACRO=VALUE and turn off optimization
	with -cl-opt-disable.
	*/
	err = clBuildProgram(program, 0, NULL, "-I.", NULL, NULL);
	if(err < 0) {
		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		program_log = (char*) malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
				program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

cl_int clh_setkarg(system_t *sys, kernels_enum kerneli, size_t n, buffers_enum args[n])
{
  cl_uint err = 0;
  for (size_t i = 0; i < n; i++) {
  	err |= clSetKernelArg(sys->kernel[kerneli], i, sizeof(cl_mem), &sys->buffer[args[i]]);
  }
  return err;
}
#define SETKARG_M(sys,k,...) clh_setkarg((sys),(k),sizeof((buffers_enum [])(__VA_ARGS__))/sizeof(buffers_enum),(buffers_enum [])(__VA_ARGS__))

// -------------- Main ising code

#define NUM_QUEUE 2

// OpenCL global variables (can't be directly accessed outside this file)
cl_device_id device;
cl_context context;
cl_program program;
cl_command_queue queue[NUM_QUEUE];

int sys_count = 0;
int sys_init = 0;

// Initialize basic ising global structures, build program and random buffer
int
ising_init()
{
	cl_int err=0;
	// Create device and context
	device = clh_create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		return(1);
	}

	// Create command queues
	for (int i = 0; i < NUM_QUEUE; ++i)
	{
		queue[i] = clCreateCommandQueue(context, device,
			CL_QUEUE_PROFILING_ENABLE, &err);
	}
	if(err < 0) {
		perror("Couldn't create a command queue");
		return(1);
	}

	// Build program
	program = clh_build_program(context, device, PROGRAM_FILE);

	sys_init == 1;
	return 0;
}

system_t
ising_new(kernel_list *kernel_opt, buffer_list *buffer_opt)
{
	system_t *newsys = malloc(sizeof *newsys);
	cl_int err = 0;

	// Create buffers
  buffer_list blist[] = {
    {state_b, svec_length*sizeof(state_t)},
    {flipE_b, svec_length*sizeof(cl_float)},
    {seeds_b, svec_length*sizeof(cl_uint)},
    {input_b, svec_length*sizeof(state_t)},
    {outpt_b, iter*svec_length*sizeof(state_t)},
    {iseed_b, sizeof(cl_uint)},
    {neigt_b, neight_count*3*sizeof(cl_int)},
    {betas_b, iter*sizeof(cl_float)},
    {probb_b, 2*pbuff_size*sizeof(cl_float)},
  };

  buffers_enum cblist = {count_b, count1_b, count2_b, count3_b};

  for(int i = 0; buffer_opt[i] != NULL; i++)
  {
    for(int k = 0; k < (sizeof(klist)/sizeof(klist[0])); k++)
    {
      if(klist[k].i == buffer_opt[i].i)
      {
        blist[k].l = buffer_opt[i].l;
        break;
      }
    }
  }

  for(int in = 0; in < (sizeof(blist)/sizeof(blist[0])); in++)
  {
    newsys->buffer[blist[in].i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
      blist[in].l, NULL, &err);
  }

  for (size_t i = 0; i < sizeof(cblist)/sizeof(cblist[0]); i++) {
    newsys->buffer[cblist[i]] = clCreateBuffer(context, CL_MEM_READ_WRITE,
      CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &(cl_uint){0}, &err);
  }

	if(err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	}

	// Create kernels
  kernel_list klist[] = {
    {gen_sys_k,   "gen_sys"},
    {gen_rand_k,  "gen_rand"},
    {sum_neigth_k,"sum_neigth"},
    {get_prob_k,  "get_prob"},
    {compare_k,   "compare_rand"},
    {arb_neigth_k,"arb_neigth"},
    {flip_k,      "flip_state"},
    {save_state_k,"save_state"},
    {measure1_k,  "measure1"},
    {measure2_k,  "measure2"},
    {measure3_k,  "measure3"},
    {next_iter_k, "next_iter"},
  };

  for(int i = 0; kernel_opt[i] != NULL; i++)
  {
    for(int k = 0; k < (sizeof(klist)/sizeof(klist[0])); k++)
    {
      if(klist[k].i == kernel_opt[i].i)
      {
        klist[k].s = kernel_opt[i].s;
        break;
      }
    }
  }

  for(int in = 0; in < (sizeof(klist)/sizeof(klist[0])); in++)
  {
    newsys->kernel[klist[in].i] = clCreateKernel(program, klist[in].s, &err);
  }

	if(err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	}

  err  = SETKARG_M(newsys,gen_sys_k   ,input_b,state_b);
  err |= SETKARG_M(newsys,gen_rand_k  ,iseed_b,seeds_b);
  err |= SETKARG_M(newsys,sum_neigth_k,state_b,flipE_b,neigt_b);
  err |= SETKARG_M(newsys,get_prob_k  ,flipE_b,betas_b,flipE_b,probb_b);
  err |= SETKARG_M(newsys,compare_k   ,flipE_b,seeds_b,flipE_b);
  err |= SETKARG_M(newsys,arb_neigth_k,flipE_b,seeds_b,flipE_b,neigt_b);
  err |= SETKARG_M(newsys,flip_k      ,state_b,flipE_b,state_b);
  err |= SETKARG_M(newsys,save_state_k,state_b,outpt_b,count_b);
  err |= SETKARG_M(newsys,measure1_k  ,state_b,outpt_b,count1_b);
  err |= SETKARG_M(newsys,measure2_k  ,state_b,outpt_b,count2_b);
  err |= SETKARG_M(newsys,measure3_k  ,state_b,outpt_b,count3_b);
  err |= SETKARG_M(newsys,next_iter_k ,seeds_b);

	if(err < 0) {
		perror("Couldn't set a kernel argument");
		exit(1);
	}

	sys_count++;
	return *newsys;
}

int
ising_configure(system_t *cursys, state_t *initial, float beta)
{
	cl_int err = 0;

	// fill the output buffer with 0
	clEnqueueFillBuffer(queue[1], cursys->buffer[output_b], (uint[]){0}, sizeof(uint), 0, iter*svec_length*sizeof(state_t), 0, NULL, NULL);

	if(initial != NULL)
	{
		err |= clEnqueueWriteBuffer(queue[1], cursys->buffer[input_b], CL_FALSE, 0, svec_length*sizeof(state_t), initial, 0, NULL, NULL);
	}

	if(beta != 0.0)
	{
		err |= clEnqueueFillBuffer(queue[1], cursys->buffer[betas_b],beta, sizeof(cl_float), 0, iter*sizeof(cl_float), 0, NULL, NULL);
	}

	if(err < 0) {
		perror("Couldn't write buffers");
		exit(1);
	}
}

int
ising_configure_betas(system_t *cursys, uint count, float *betas)
{
	cl_int err;
	cl_float betas_i[iter];

	for (int k = 0; k < iter; ++k)
	{
			betas_i[k] = betas[count*k/iter];
	}

  err |= clEnqueueWriteBuffer(queue[1], cursys->buffer[betas_b], CL_FALSE, 0, sizeof(betas_i), betas_i, 0, NULL, NULL);
	if(err < 0) {
		perror("Couldn't write buffers");
		exit(1);
	}
}

int
ising_enqueue(system_t *cursys)
{
	cl_int err;// = clEnqueueMarker(queue[1],&calc_done[0]);

	// // Enqueue kernels
	// for(int i = 1; i < iter; i++)
	// {
	// 	fprintf(stderr,
  //
	// 	// Calculate next iteration:
	// 	err |= clEnqueueNDRangeKernel(queue[1], cursys->kernel[0], 2, NULL,
	// 		global_2D_size, local_2D_size, 2, &calc_done[4*i-4], &calc_done[4*i]);
  //
	// 	// Increment counter
	// 	err |= clEnqueueTask(queue[0], cursys->kernel[4], 1, &calc_done[4*i],
	// 		&calc_done[4*i+1]);
  //
	// 	// Measure:
	// 	err |= clEnqueueNDRangeKernel(queue[1], cursys->kernel[2], 1, NULL,
	// 		(size_t[]){svec_length/2}, local_1D_size, 2, &calc_done[4*i-1],
	// 		&calc_done[4*i+2]);
  //
	// 	// Increment counter
	// 	err |= clEnqueueTask(queue[0], cursys->kernel[5], 1, &calc_done[4*i+2],
	// 		&calc_done[4*i+3]);
  //
	// 	if((i % (iter/cursys->prob_num)) == 0)
	// 	{
	// 		err |= clEnqueueTask(queue[1], cursys->kernel[3], 1,
	// 			&calc_done[4*i], NULL);
	// 	}
	// }
	if(err < 0){
		perror("Couldn't enqueue the kernel");
		exit(1);
	}

	clFlush(queue[1]);
	clFlush(queue[0]);
}

int
ising_get_data(system_t *cursys, int *data)
{
	clFinish(queue[0]);
	clFinish(queue[1]);

	cl_int err = clEnqueueReadBuffer(queue[1], cursys->buffer[output_b], CL_TRUE,
    0, iter*svec_length*sizeof(state_t), data, 0, NULL, NULL);
	if(err < 0) {
		perror("Couldn't read the buffer");
		exit(1);
	}
}

int
ising_free(system_t *cursys)
{
	for (int i = 0; i < NUM_KERNEL; ++i)
	{
		clReleaseKernel(cursys->kernel[i]);
	}

	for (int i = 0; i < NUM_BUFFER; ++i)
	{
		clReleaseMemObject(cursys->buffer[i]);
	}

	sys_count--;
	if(sys_count<=0)
	{
		for (int i = 0; i < NUM_QUEUE; ++i)
		{
			clReleaseCommandQueue(queue[i]);
		}

		clReleaseProgram(program);
		clReleaseContext(context);
	}
}
