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

  // cl_uint dev_num;
  //
  // err|=clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &dev_num, NULL);
  // if(err < 0) {
  //   perror("Couldn't get binary length");
  //   exit(1);
  // }
  //
  // size_t bin_length[dev_num], tsize=0;
  //
  // err|=clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, dev_num*sizeof(size_t), bin_length, NULL);
  // if(err < 0) {
  //   perror("Couldn't get binary length");
  //   exit(1);
  // }
  //
  // unsigned char *bin_out[dev_num];
  //
  // for(uint8_t i = 0; i < dev_num; i++)
  // {
  //   printf("dev_num: %d, bin size: %d\n", i, bin_length[i]);
  //   bin_out[i] = malloc(bin_length[i]);
  // }
  //
  // err=clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*dev_num, bin_out, NULL);
  // if(err < 0) {
  //   perror("Couldn't get binary");
  //   exit(1);
  // }
  //
  // for(int i = 0; i < bin_length[0]; i++) printf("0x%02X ",bin_out[0][i]);
  //
  // FILE *fp = fopen("./clprog.bin", "wb");
  // fwrite(bin_out[0], 1, bin_length[0], fp);
  // fclose(fp);

	return program;
}

// -------------- Main ising code

#define NUM_QUEUE 1

// Ising OpenCL local variables
cl_device_id device;
cl_context context;
cl_program program;
cl_command_queue queue[NUM_QUEUE];
cl_event *calc_done;

cl_mem rand_buff_g;
cl_kernel kernel_rand;
cl_event last_event;

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
		queue[i] = clCreateCommandQueueWithProperties(context, device,
      (const cl_queue_properties[]){CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,0},
        &err);
	}
	if(err < 0) {
		perror("Couldn't create a command queue");
		return(1);
	}

	// Build program
	program = clh_build_program(context, device, PROGRAM_FILE);

	// Event list
	calc_done = malloc(2*iter*sizeof(cl_event)+2);

	// Random seeds (1 per iteration)
	cl_uint rand_seed[iter];
	for(int i = 0; i < iter; i++)
	{
		rand_seed[i] = rand();
	}

	// Create and write random buffer
	rand_buff_g = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
		iter*sizeof(cl_uint), rand_seed, &err);

	// Create random generator kernel
	kernel_rand = clCreateKernel(program, RAND_FUNC, &err);

	// Set random generator kernel arguments
	err |= clSetKernelArg(kernel_rand, 0, sizeof(cl_mem), &rand_buff_g);
	if(err < 0) {
		perror("Couldn't create a kernel argument");
		exit(1);
	}

	sys_init == 1;
	return 0;
}

system_t
ising_new()
{
	system_t *newsys = malloc(sizeof *newsys);
	cl_int err = 0;

	// Create buffers
	newsys->rand_buff = rand_buff_g; // Buffer already allocated on init

	newsys->state = clCreateBuffer(context, CL_MEM_READ_WRITE,
		(iter+2)*svec_length*sizeof(state_t), NULL, &err);

	newsys->output = clCreateBuffer(context, CL_MEM_READ_WRITE,
		(iter+2)*sizeof(cl_int), NULL, &err);

	newsys->prob = clCreateBuffer(context, CL_MEM_READ_WRITE,
		prob_buff*prob_length*sizeof(cl_uint), NULL, &err);

	newsys->counter[0] = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &(cl_uint){1}, &err);

	for (int i = 1; i < NUM_COUNT; ++i)
	{
		newsys->counter[i] = clCreateBuffer(context, CL_MEM_READ_WRITE |
			CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &(cl_uint){0}, &err);
	}

	if(err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	}

	// Create kernels
	newsys->kernel[0] = clCreateKernel(program, ISING_FUNC, &err);
	newsys->kernel[1] = kernel_rand;
	newsys->kernel[2] = clCreateKernel(program, MEAS_FUNC, &err);
	newsys->kernel[3] = clCreateKernel(program, COUNTER_INCR, &err);
	newsys->kernel[4] = clCreateKernel(program, COUNTER_INCR, &err);
	newsys->kernel[5] = clCreateKernel(program, COUNTER_INCR, &err);
	if(err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	}

	// Set base kernel arguments
	err  = clSetKernelArg(newsys->kernel[0], 0, sizeof(cl_mem), &newsys->state);
	err |= clSetKernelArg(newsys->kernel[0], 1, sizeof(cl_mem), &newsys->rand_buff);
	err |= clSetKernelArg(newsys->kernel[0], 2, sizeof(cl_mem), &newsys->counter[0]);
	err |= clSetKernelArg(newsys->kernel[0], 3, sizeof(cl_mem), &newsys->prob);
	err |= clSetKernelArg(newsys->kernel[0], 4, sizeof(cl_mem), &newsys->counter[1]);

	// Set measurement kernel arguments
	err |= clSetKernelArg(newsys->kernel[2], 0, sizeof(cl_mem), &newsys->state);
	err |= clSetKernelArg(newsys->kernel[2], 1, sizeof(cl_mem), &newsys->counter[2]);
	err |= clSetKernelArg(newsys->kernel[2], 2, sizeof(cl_mem), &newsys->output);
	err |= clSetKernelArg(newsys->kernel[2], 3, local_length*sizeof(cl_int), NULL);

	// Set next_prob kernel arguments
	err |= clSetKernelArg(newsys->kernel[3], 0, sizeof(cl_mem), &newsys->counter[1]);

	// Set counter0 kernel arguments
	err |= clSetKernelArg(newsys->kernel[4], 0, sizeof(cl_mem), &newsys->counter[0]);

	// Set counter2 kernel arguments
	err |= clSetKernelArg(newsys->kernel[5], 0, sizeof(cl_mem), &newsys->counter[2]);

	if(err < 0) {
		perror("Couldn't create a kernel argument");
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
	clEnqueueFillBuffer(queue[0], cursys->output, (uint[]){0}, sizeof(uint),
		0, (iter+2)*sizeof(uint), 0, NULL, NULL);

	if(initial != NULL)
	{
		err |= clEnqueueWriteBuffer(queue[0], cursys->state, CL_FALSE, 0,
			svec_length*sizeof(state_t), initial, 0, NULL, NULL);
	}

	if(beta != 0.0)
	{
		// Probability vector (index = number of aligned spins in neighborhood)
		cl_uint prob[prob_length];
		for (int i = 0; i < prob_length; i++)
		{
			prob[i] = (cl_ulong)CL_UINT_MAX * max_prob *
			((i <= prob_zero)? 1 : exp(-beta*(i-prob_zero)));
		}

		err |= clEnqueueWriteBuffer(queue[0], cursys->prob, CL_FALSE, 0,
			prob_length*sizeof(cl_uint), prob, 0, NULL, NULL);

		cursys->prob_num = 1;
	}


	if(err < 0) {
		perror("Couldn't write buffers");
		exit(1);
	}

	clFlush(queue[0]);
	clFinish(queue[0]);
}

int
ising_configure_betas(system_t *cursys, uint count, float *betas)
{
	cl_int err;
	cl_uint prob[count*prob_length];

	for (int k = 0; k < count; ++k)
	{
		for (int i = 0; i < prob_length; i++)
		{
			prob[prob_length * k + i] = (cl_ulong)CL_UINT_MAX * max_prob *
				((i <= prob_zero)? 1 : exp(-betas[k]*(i-prob_zero)));
		}
	}

	cursys->prob_num = count;

	err |= clEnqueueWriteBuffer(queue[0], cursys->prob, CL_FALSE, 0,
		count*prob_length*sizeof(cl_uint), prob, 0, NULL, NULL);

	if(err < 0) {
		perror("Couldn't write buffers");
		exit(1);
	}

	clFlush(queue[0]);
	clFinish(queue[0]);
}

int
ising_enqueue(system_t *cursys)
{
	// Enqueue kernels
	cl_int err = clEnqueueNDRangeKernel(queue[0], cursys->kernel[2], 1, NULL,
		(size_t[]){svec_length}, local_1D_size, 0, NULL, NULL);

	for(int i = 0; i < iter; i++)
	{
		fprintf(stderr,"\rEnqueue loop %d", i);

		// Calculate next iteration:
		err |= clEnqueueNDRangeKernel(queue[0], cursys->kernel[0], 2, NULL,
			global_2D_size, local_2D_size, 0, NULL, &calc_done[2*i]);

		// Measure:
		err |= clEnqueueNDRangeKernel(queue[0], cursys->kernel[2], 1, NULL,
			(size_t[]){svec_length}, local_1D_size, 0, NULL, &calc_done[2*i+1]);

    if(err < 0){
      perror("Couldn't enqueue the kernels");
      exit(1);
    }
	}

	clFlush(queue[0]);
}

int
ising_get_states(system_t *cursys, state_t *states)
{
	clFinish(queue[0]);

	cl_int err = clEnqueueReadBuffer(queue[0], cursys->state, CL_TRUE, 0,
		iter*svec_length*sizeof(state_t), states, 0, NULL, NULL);
	if(err < 0) {
		perror("Couldn't read the buffer");
		exit(1);
	}
}

int
ising_get_data(system_t *cursys, int *data)
{
	clFinish(queue[0]);

	cl_int err = clEnqueueReadBuffer(queue[0], cursys->output, CL_TRUE, 0,
		iter*sizeof(cl_int), data, 0, NULL, NULL);
	if(err < 0) {
		perror("Couldn't read the buffer");
		exit(1);
	}
}

int
ising_next_prob(system_t *cursys)
{
	clFinish(queue[0]);

	cl_int err = clEnqueueNDRangeKernel(queue[0], cursys->kernel[3], 1, NULL,
               (size_t[]){1}, (size_t[]){1}, 0, NULL, NULL);
	if(err < 0) {
		perror("Couldn't enqueue task");
		exit(1);
	}
}

int
ising_free(system_t *cursys)
{
	sys_count--;

	clReleaseMemObject(cursys->state);
	clReleaseMemObject(cursys->rand_buff);
	clReleaseMemObject(cursys->prob);

	for (int i = 0; i < NUM_KERNEL; ++i)
	{
		clReleaseKernel(cursys->kernel[i]);
	}

	for (int i = 0; i < NUM_COUNT; ++i)
	{
		clReleaseMemObject(cursys->counter[i]);
	}

	if(sys_count<=0)
	{
		for (int i = 0; i < NUM_QUEUE; ++i)
		{
			clReleaseCommandQueue(queue[0]);
		}
		clReleaseProgram(program);
		clReleaseContext(context);
	}
}

void
ising_profile()
{
	double runtimes[2], initial=0, final;

	for (int i = 0; i < 2*iter; ++i)
	{
		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(calc_done[i], CL_PROFILING_COMMAND_START,
			sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(calc_done[i], CL_PROFILING_COMMAND_END,
			sizeof(time_end), &time_end, NULL);

    if(!initial) initial=time_start;
    final = time_end;

		runtimes[i%2] += time_end-time_start;
	}


	printf("Mean runtime for each kernel per iteration:\nMain: %5.3f µs\nMeasure: %5.3f µs\nTotal: %5.2f ms\n",
		runtimes[0]/1e3/iter, runtimes[1]/1e3/iter, (final-initial)/1e6);
}
