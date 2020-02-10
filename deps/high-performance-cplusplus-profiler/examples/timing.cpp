#if defined(_MSC_VER)
	#undef inline
	#define inline __forceinline

	#pragma comment( lib, "Winmm.lib" )

	#include <windows.h>
#else
	#include <sys/time.h>
#endif


#include <math.h>
#include <stdio.h>
#include <time.h>


#include "Profiler.h"

struct rdtsc { inline void getTicks() { Profiler::Timer::getticks(); } };
struct rdtsc_serial { inline void getTicks() { Profiler::Timer::getticks_serial(); } };
struct clck { inline void getTicks() { clock(); } };

#if defined(_MSC_VER)
struct qpc { inline void getTicks() { LARGE_INTEGER li; QueryPerformanceCounter( &li ); } };
struct tgt { inline void getTicks() { timeGetTime(); } };
#else
struct gtod { inline void getTicks() { timeval t; gettimeofday( &t, NULL); } };
#endif

template< class Timer > double time( const int trials ) {
	Timer t;
	Profiler::u64 min = 1000000000000000ULL;
	for ( int run = 0; run < 5000; run++ ) {
		Profiler::Timer measure;
		{
			ScopedTimer scoped( measure );
			for ( int i = 0; i < trials; i++ )
				t.getTicks();
		}
		if ( measure.ticks < min )
			min = measure.ticks;
	}

	return Profiler::average( min, trials );
}

int main( int argc, const char *argv[] ) {
	const int trials = 1000;

	printf( "rdtsc, %.0f cycles\n", time<rdtsc>( trials ) );
	printf( "rdtsc_serial, %.0f cycles\n", time<rdtsc_serial>( trials ) );
	printf( "clock, %.0f cycles\n", time<clck>( trials ) );
#if defined(_MSC_VER)
	printf( "QueryPerformanceCounter, %.0f cycles\n", time<qpc>( trials ) );
	printf( "timeGetTime, %.0f cycles\n", time<tgt>( trials ) );
#else
	printf( "gettimeofday, %.0f cycles\n", time<gtod>( trials ) );
#endif

	return 0;
}
