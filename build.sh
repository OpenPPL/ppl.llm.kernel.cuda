#!/bin/bash

workdir=`pwd`

if [[ `uname` == "Linux" ]]; then
    processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
    processor_num=64
elif [[ `uname` == "Darwin" ]]; then
    processor_num=`sysctl machdep.cpu | grep machdep.cpu.core_count | cut -d " " -f 2`
else
    processor_num=1
fi

build_type='Release'
options="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_INSTALL_PREFIX=install $*"

ppl_build_dir="${workdir}/ppl-build"
mkdir ${ppl_build_dir}
cd ${ppl_build_dir}
cmd="cmake $options .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
echo "cmd -> $cmd"
eval "$cmd"
