#include <stdio.h>
#include <stdlib.h>
#include "Profiler.h"

void test1() { PROFILE_SCOPED() printf( "1" ); }
void test2() { PROFILE_SCOPED() printf( "2" ); }
void test3() { PROFILE_SCOPED() printf( "3" ); }
void test4() { PROFILE_SCOPED() printf( "4" ); }
void test5() { PROFILE_SCOPED() printf( "5" ); }

void test6() { PROFILE_SCOPED() if ( rand() & 3 ) test1(); else test6(); printf( "6" ); }
void test7() { PROFILE_SCOPED() if ( rand() & 3 ) test5(); else test7(); printf( "7" ); }
void test8() { PROFILE_SCOPED() if ( rand() & 3 ) test2(); else test8(); printf( "8" ); }
void test9() { PROFILE_SCOPED() if ( rand() & 3 ) test4(); else test9(); printf( "9" ); }
void testa() { PROFILE_SCOPED() if ( rand() & 3 ) test3(); else testa(); printf( "a" ); }

int main( int argc, char *argv[] ) {
	srand( 0 );

 	for ( int i = 0; i < 2000; ++i ) {
		switch ( rand() % 10 ) {
			case 0: test1(); break;
			case 1: test2(); break;
			case 2: test3(); break;
			case 3: test4(); break;
			case 4: test5(); break;
			case 5: test6(); break;
			case 6: test7(); break;
			case 7: test8(); break;
			case 8: test9(); break;
			case 9: testa(); break;
		}
	}

	printf( "\n" );
	Profiler::dump();


	return ( 0 );
}
