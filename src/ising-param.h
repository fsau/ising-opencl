#define PROGRAM_FILE "ising.cl"
#define ISING_FUNC "ising_calc"
#define CINCR_FUNC "counter_incr"
#define SAVE_FUNC "ising_save"

#define sizeX 64
#define sizeY 64
#define iter 512
#define prob_length 5 // num of neighborhood+1

#define svec_length sizeX*sizeY

#ifdef __OPENCL_VERSION__
typedef char state_t;
typedef char4 state_v;
#else
typedef cl_char state_t;
typedef cl_char4 state_v;
#endif

#define max(x,y) ((x)>(y)?(x):(y))

// Index macro for 2D -> 1D. f: fast, c: iteration counter
#define ind(x,y) ( ((x)%sizeX)*sizeX + ((y)%sizeY) )
#define find(x,y) ( (x)*sizeX + (y) )
#define cind(c,x,y) ( ((c)%iter)*svec_length + ((x)%sizeX)*sizeX + ((y)%sizeY) )
#define cfind(c,x,y) ( (c)*svec_length + (x)*sizeX + (y) )
