#define sizeY 64
#define sizeX 64
#define iter 1024 // number of iteration to calculate each cycle
#define prob_length 5 // num of neighborhood+1
#define prob_buff 16 // prob buffer size when using multiple probs

#define prob_zero ((prob_length-1)/2)
#define max_prob 1.0

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
