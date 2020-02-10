GXX     := g++
CFLAGS  := -Wall -Werror -lpthread -O2 -fomit-frame-pointer
INCLUDE := -I.

COMMON      = Profiler.cpp
SPHEREFLAKE = $(COMMON) examples/example-sphereflake.cpp
THREADS     = $(COMMON) examples/example-threads.cpp
SIMPLE      = $(COMMON) examples/example-simplerecursive.cpp
TIMING      = $(COMMON) examples/timing.cpp

sphereflake: $(SPHEREFLAKE)
	$(GXX) $(INCLUDE) $(CFLAGS) $(SPHEREFLAKE) -o $@

all: sphereflake threads simple timing

threads: $(THREADS)
	$(GXX) $(INCLUDE) $(CFLAGS) $(THREADS) -o $@ 
simple: $(SIMPLE)
	$(GXX) $(INCLUDE) $(CFLAGS) $(SIMPLE) -o $@
timing: $(TIMING)
	$(GXX) $(INCLUDE) $(CFLAGS) $(TIMING) -o $@

clean:
	rm -f examples/*.o
	rm -f sphereflake
	rm -f simple
	rm -f threads
	rm -f timing
