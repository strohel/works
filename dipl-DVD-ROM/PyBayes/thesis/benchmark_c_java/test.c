// by Erik Wrenholt
#include <stdio.h>
#include <sys/time.h>

#define BAILOUT 16
#define MAX_ITERATIONS 1000

int mandelbrot(float x, float y) {
	float cr = y - 0.5;
	float ci = x;
	float zi = 0.0;
	float zr = 0.0;
	int i = 0;
	while(1) {
		i ++;
		float temp = zr * zi;
		float zr2 = zr * zr;
		float zi2 = zi * zi;
		zr = zr2 - zi2 + cr;
		zi = temp + temp + ci;
		if (zi2 + zr2 > BAILOUT)
			return i;
		if (i > MAX_ITERATIONS)
			return 0;
	}
}

void run2() {
	int x,y;
	for (y = -39; y < 39; y++) {
		fputs("\n", stderr);
		for (x = -39; x < 39; x++) {
			int i = mandelbrot(x/40.0, y/40.0);
			if (i==0)
				fputs("*", stderr);
			else
				fputs(" ", stderr);
		}
	}
	fputs("\n", stderr);
}

void run() {
	struct timeval aTv;
	gettimeofday(&aTv, NULL);
	long init_time = aTv.tv_sec;
	long init_usec = aTv.tv_usec;
	int i;
	for (i = 0; i < 100; i++)
		run2();
	gettimeofday(&aTv,NULL);
	double query_time = (aTv.tv_sec - init_time) + (double)(aTv.tv_usec - init_usec)/1000000.0;
	printf ("C Elapsed %0.2f\n", query_time);
}

int main (int argc, const char * argv[]) {
	run();
	run();
	run();
}
