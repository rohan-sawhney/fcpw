
@echo off
mkdir output
echo [BENCHING] Spiral
..\build\tests\RelWithDebInfo\aggregate_tests -s input\spiral.obj --auto -n 100000 > output\spiral.txt
echo [BENCHING] Bunny
..\build\tests\RelWithDebInfo\aggregate_tests -s input\bunny.obj --auto -n 100000 > output\bunny.txt
echo [BENCHING] Armadillo
..\build\tests\RelWithDebInfo\aggregate_tests -s input\armadillo.obj --auto -n 100000 > output\armadillo.txt
echo [BENCHING] Kitten
..\build\tests\RelWithDebInfo\aggregate_tests -s input\kitten.obj --auto -n 100000 > output\kitten.txt
echo [BENCHING] Sponza
..\build\tests\RelWithDebInfo\aggregate_tests -s large_input\sponza.obj --auto -n 100000 > output\sponza.txt
echo [BENCHING] Hairball
..\build\tests\RelWithDebInfo\aggregate_tests -s large_input\hairball.obj --auto -n 100000 > output\hairball.txt
echo [BENCHING] Buddha
..\build\tests\RelWithDebInfo\aggregate_tests -s large_input\buddha.obj --auto -n 100000 > output\buddha.txt
