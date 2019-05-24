#define sizeY 256
#define sizeX 256
#define iter 256 // number of iteration to calculate each cycle
#define neight_count 4
#define pbuff_size 5

#define svec_length (sizeX*sizeY)

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
#define cind(c,x,y) ( (c)*svec_length + ((x)%sizeX)*sizeX + ((y)%sizeY) )
#define cfind(c,x,y) ( (c)*svec_length + (x)*sizeX + (y) )
