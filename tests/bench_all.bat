
set QUERIES=100000

mkdir output
echo "[BENCHING] Spiral"
..\build\tests\RelWithDebInfo\aggregate_tests -s input\spiral.obj --auto -n %QUERIES% > output\spiral.txt
echo "[BENCHING] Bunny"
..\build\tests\RelWithDebInfo\aggregate_tests -s input\bunny.obj --auto -n %QUERIES% > output\bunny.txt
echo "[BENCHING] Armadillo"
..\build\tests\RelWithDebInfo\aggregate_tests -s input\armadillo.obj --auto -n %QUERIES% > output\armadillo.txt
echo "[BENCHING] Kitten"
..\build\tests\RelWithDebInfo\aggregate_tests -s input\kitten.obj --auto -n %QUERIES% > output\kitten.txt
@REM echo "[BENCHING] Sponza"
@REM ../build/tests/RelWithDebInfo/aggregate_tests -s large_input/sponza.obj --auto -n %QUERIES% > output/sponza.txt
@REM echo "[BENCHING] Hairball"
@REM ../build/tests/RelWithDebInfo/aggregate_tests -s large_input/hairball.obj --auto -n %QUERIES% > output/hairball.txt
@REM echo "[BENCHING] Buddha"
@REM ../build/tests/RelWithDebInfo/aggregate_tests -s large_input/buddha.obj --auto -n %QUERIES% > output/buddha.txt
@REM echo "[BENCHING] Powerplant"
@REM ../build/tests/RelWithDebInfo/aggregate_tests -s large_input/powerplant.obj --auto -n $((QUERIES/10)) > output/powerplant.txt
@REM ..\build\tests\RelWithDebInfo\aggregate_tests -s input\bunny.obj --auto -n 1000000 > output\bunny.txt