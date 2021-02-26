
QUERIES=1000000

mkdir -p output
echo "[BENCHING] Spiral"
../build/tests/aggregate_tests -s input/spiral.obj --auto -n $QUERIES > output/spiral.txt 2> /dev/null
echo "[BENCHING] Bunny"
../build/tests/aggregate_tests -s input/bunny.obj --auto -n $QUERIES > output/bunny.txt
echo "[BENCHING] Armadillo"
../build/tests/aggregate_tests -s input/armadillo.obj --auto -n $QUERIES > output/armadillo.txt
echo "[BENCHING] Kitten"
../build/tests/aggregate_tests -s input/kitten.obj --auto -n $QUERIES > output/kitten.txt
echo "[BENCHING] Sponza"
../build/tests/aggregate_tests -s large_input/sponza.obj --auto -n $QUERIES > output/sponza.txt
echo "[BENCHING] Hairball"
../build/tests/aggregate_tests -s large_input/hairball.obj --auto -n $QUERIES > output/hairball.txt
echo "[BENCHING] Buddha"
../build/tests/aggregate_tests -s large_input/buddha.obj --auto -n $QUERIES > output/buddha.txt
echo "[BENCHING] Powerplant"
../build/tests/aggregate_tests -s large_input/powerplant.obj --auto -n $((QUERIES/10)) > output/powerplant.txt
