
mkdir -p output
mkdir -p output/graphs

if [[ $# -eq 0 ]] ; then
    echo 'Pass path to data file'
    exit 1
fi

# Vectorized, non coherent, CPQs
python3 graph.py $1 CPQ yes no avg avg auto auto output/graphs

# Scalar, non coherent, CPQs
python3 graph.py $1 CPQ no no avg avg auto auto output/graphs

# Vectorized, coherent, CPQs
python3 graph.py $1 CPQ yes yes avg avg auto auto output/graphs

# Scalar, coherent, CPQs
python3 graph.py $1 CPQ no yes avg avg auto auto output/graphs

# Vectorized, non coherent, Rays
python3 graph.py $1 RAY yes no avg avg auto auto output/graphs

# Scalar, non coherent, Rays
python3 graph.py $1 RAY no no avg avg auto auto output/graphs

# Vectorized, coherent, Rays
python3 graph.py $1 RAY yes yes avg avg auto auto output/graphs

# Scalar, coherent, Rays
python3 graph.py $1 RAY no yes avg avg auto auto output/graphs
