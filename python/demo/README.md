```bash
export LIBTORCH_DIR=/path/to/libtorch
cd ppl.llm.kernel.cuda
```

build:

```bash
PPL_BUILD_THREAD_NUM=32 ./build.sh {options} -DPPLNN_ENABLE_TORCH_API=ON -DPPLNN_LIBTORCH_DIR=$LIBTORCH_DIR
```

run:

```bash
LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=./ppl-build/install/lib python3 python/demo/demo.py
```
