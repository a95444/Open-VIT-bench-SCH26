
module load openmpi/4.1.6--gcc--12.2.0

if [ ! -d "obj" ]; then
    mkdir "obj"
fi
if [ ! -d "bin" ]; then
    mkdir "bin"
fi
if [ ! -d "omp_obj" ]; then
    mkdir "omp_obj"
fi
if [ ! -d "omp_bin" ]; then
    mkdir "omp_bin"
fi


if [ ! -d "include" ]; then
    echo Error: missing include folder!
    exit 1
fi

if [ ! -d "src" ]; then
    echo Error: missing src folder!
    exit 1
fi
make bin/vit.exe && echo bin/vit.exe created

if [ ! -d "omp_src" ]; then
    echo Error: missing omp_src folder!
    exit 1
fi
make omp_bin/vit.exe && echo omp_bin/vit.exe created
