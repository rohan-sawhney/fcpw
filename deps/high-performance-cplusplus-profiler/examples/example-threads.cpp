#include <string.h>
#include <stdio.h>

#include "Profiler.h"

#if defined(_MSC_VER)
	#include <windows.h>
	typedef HANDLE thread_handle;
	#define THREAD_FUNC() DWORD WINAPI
	typedef DWORD (WINAPI *thread_function)( void * );
	thread_handle thread_create( thread_function thread, void *parm ) { DWORD id; return CreateThread( NULL, 0, thread, parm, 0, &id ); }
	void thread_wait( thread_handle h ) { PROFILE_PAUSE_SCOPED() WaitForSingleObject( h, INFINITE ); }
	void thread_sleep( unsigned int ms ) { PROFILE_PAUSE_SCOPED() Sleep( ms ); }
#else
	#include <pthread.h>
	#include <sys/time.h>
	#include <sys/types.h>
	#include <unistd.h>
	#include <stdlib.h>

	typedef pthread_t thread_handle;
	#define THREAD_FUNC() static void *
	typedef void *(*thread_function)( void * );
	thread_handle thread_create( thread_function thread, void *parm ) { thread_handle h; pthread_create( &h, NULL, thread, parm ); return h; }
	void thread_wait( thread_handle h ) { PROFILE_PAUSE_SCOPED() pthread_join( h, NULL ); }
	void thread_sleep( unsigned int ms ) { PROFILE_PAUSE_SCOPED() timeval tv; tv.tv_sec = 0; tv.tv_usec = ms * 1000; select(0, NULL, NULL, NULL, &tv); }
#endif

const int threadCount = 64, dumpCount = 8;
volatile bool done = false;

size_t wasteTime( const char *hashMe ) {
	size_t hash = strlen( hashMe );
	for ( ; *hashMe; hashMe++ )
		hash = ( hash * hash ) ^ *hashMe;
	return hash;
}

THREAD_FUNC() TestThread( void *in ) {
	PROFILE_THREAD_SCOPED()

	size_t sum = 0;
	for ( int i = 0; i < 6000; i++ ) {
		PROFILE_SCOPED_DESC( "inner" )
		thread_sleep( rand() % 2 );
		sum += wasteTime( "This is a decently long string to hash ok? This is a decently long string to hash ok? This is a decently long string to hash ok? This is a decently long string to hash ok? This is a decently long string to hash ok?" );
	}

	return 0;
}

THREAD_FUNC() DumpThread( void *in ) {
	PROFILE_THREAD_SCOPED()

	while ( !done ) {
		if ( ( rand() % 100 ) < 25 ) {
			PROFILE_SCOPED_DESC( "/dump" );
			Profiler::dump();
		} else {
			if ( rand() % 100 < 35 ) {
				PROFILE_SCOPED_DESC( "/reset" );
				Profiler::reset();
			}
		}
		thread_sleep( 20 );
	}

	return 0;
}


int main( int argc, const char **argv ) {
	thread_handle threads[threadCount], dumpers[dumpCount];

	srand(0);
	for ( int i = 0; i < dumpCount; i++ ) 
		dumpers[i] = thread_create( DumpThread, NULL );

	thread_sleep( 100 );

	printf( "make threads\n" );

	for ( int i = 0; i < threadCount; i++ ) {
		thread_sleep( 10 );
		threads[i] = thread_create( TestThread, NULL );
	}

	printf( "wait threads\n" );

	for ( int i = 0; i < threadCount; i++ ) 
		thread_wait( threads[i] );

	done = true;

	printf( "wait dumpers\n" );

	for ( int i = 0; i < dumpCount; i++ ) 
		thread_wait( dumpers[i] );
}
