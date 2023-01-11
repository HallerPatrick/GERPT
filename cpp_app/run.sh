
# Check if venv is already activated
#
if [[ "$VIRTUAL_ENV" == "" ]]
then
    echo "Venv not activated"
    exit
fi

# Into build dir
cd build

# Build with pip/conda installed libtorch
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release

# Run
./cpp-app

cd ..

