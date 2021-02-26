

if [[ $# -eq 0 ]] ; then
    echo 'Pass path to data file'
    exit 1
fi

mkdir -p output/graphs/$(basename $1)

# Vectorized, non coherent, CPQs
python3 graph.py $1 CPQ yes no avg avg auto auto output/graphs/$(basename $1)

# Scalar, non coherent, CPQs
python3 graph.py $1 CPQ no no avg avg auto auto output/graphs/$(basename $1)

# Vectorized, coherent, CPQs
python3 graph.py $1 CPQ yes yes avg avg auto auto output/graphs/$(basename $1)

# Scalar, coherent, CPQs
python3 graph.py $1 CPQ no yes avg avg auto auto output/graphs/$(basename $1)

# Vectorized, non coherent, Rays
python3 graph.py $1 RAY yes no avg avg auto auto output/graphs/$(basename $1)

# Scalar, non coherent, Rays
python3 graph.py $1 RAY no no avg avg auto auto output/graphs/$(basename $1)

# Vectorized, coherent, Rays
python3 graph.py $1 RAY yes yes avg avg auto auto output/graphs/$(basename $1)

# Scalar, coherent, Rays
python3 graph.py $1 RAY no yes avg avg auto auto output/graphs/$(basename $1)
