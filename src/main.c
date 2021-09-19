#include "ising.h"

#include <stdio.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>

int kbhit();

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

int
main(void)
{
	int *mag_data = malloc(iter*sizeof(uint));
	state_t *states_data = malloc(iter*svec_length*sizeof(state_t));
	state_t *initial = malloc(svec_length*sizeof(state_t));

	// Hot start
  uint rseed = (uint)time(NULL);
	srand(rseed);
  srand(rand());
  printf("seed: %lu\n", rseed);
	for (int i = 0; i < svec_length; ++i)
	{
		initial[i] = (rand() < (RAND_MAX/2))? 1 : -1;
	}

	// Annealing
	float betas[prob_buff];
	for (int i = 0; i < prob_buff; ++i)
	{
		betas[i] = 0.8/(0.6 - 0.5*MIN(i,prob_buff)/(float)prob_buff);
    // betas[i] = 0.5/(0.6 - 0.5*(prob_buff-i/5.0-1)/(float)prob_buff);
	}

	// Load system and calculate
	ising_init();
	system_t mysys = ising_new();
	ising_configure(&mysys, initial, 0.0);
	ising_configure_betas(&mysys, prob_buff, betas);
	ising_enqueue(&mysys);
  printf("\nDone! Reading output");

	ising_get_data(&mysys, mag_data);
	ising_get_states(&mysys, states_data);
	ising_free(&mysys);

	// Print states/data
	printf("\e[1;1H\e[2J"); // clear screen
	for (int k = 0; k < iter; k+=1)
	{
		printf("\033[0;0H"); // Move cursor to (0,0)

		// Print states
		printf("┌");
		for(int i = 0; i < 2*sizeX/OVERSAMPLE+2; i++) printf("─");
		printf("┐\n");

		for (int i = 0; i < sizeY; i+=OVERSAMPLE)
		{
		printf("│ ");
			for (int j = 0; j < sizeX; j+=OVERSAMPLE)
			{
				int sum = 0;
        for(int iov = 0; iov < OVERSAMPLE; iov++)
        {
          for(int jov = 0; jov < OVERSAMPLE; jov++)
          {
            sum+=states_data[svec_length*k + (i+iov)*sizeX + (j+jov)];
          }
        }
				printf("\033[48;5;%3dm  \e[0m",242 + 8*sum/(OVERSAMPLE*OVERSAMPLE));
			}
			printf("\e[0m │\n");
		}

		printf("└");
		for(int i = 0; i < 2*sizeX/OVERSAMPLE+2; i++) printf("─");
		printf("┘\n");

		// Calculate mag on CPU (sum of every cell)
		int sum = 0;
		for (int i = 0; i < svec_length; ++i)
		{
			sum += states_data[svec_length*k+i];
		}

		// Print data
		printf("Iter: %d/%d %05f %05f\n", k+1, iter, states_data[svec_length*k],states_data[svec_length*(k+1)-1]);
		printf("Mag (GPU): %+5.4f\n", (float)mag_data[k]/(sizeX*sizeY));
		printf("Mag (CPU): %+5.4f %s\n", (float)sum/(sizeX*sizeY),(mag_data[k]==sum)?"OK ":"ERR");
		printf("[");
		for (int i = 0; i < 100; ++i)
		{
			printf("%c",(((float)i/100) < (0.5+(float)mag_data[k]/(2*sizeX*sizeY)))?'#':'-');
		}
		printf("]\n");
		printf("                                                  |\n");
		printf("                                                  0\n");

		if(kbhit()) // Break if key is pressed
		{
			getc(stdin);
			break;
		}

		usleep(1000000/10); // ~1e6/fps
	}

	ising_profile();

	free(initial);
	free(states_data);
	free(mag_data);
	return 0;
}

/**
 Linux (POSIX) implementation of _kbhit().
 Morgan McGuire, morgan@cs.brown.edu
 */
int kbhit() {
    static const int STDIN = 0;
    static int initialized = 0;

    if (! initialized) {
        // Use termios to turn off line buffering
        struct termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        tcsetattr(STDIN, TCSANOW, &term);
        setbuf(stdin, NULL);
        initialized = 1;
    }

    int bytesWaiting;
    ioctl(STDIN, FIONREAD, &bytesWaiting);
    return bytesWaiting;
}
