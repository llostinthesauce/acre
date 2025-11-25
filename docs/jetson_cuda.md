# Jetson CUDA hookup for ACRE

Use this to reuse your working CUDA build of llama.cpp inside the ACRE app without affecting other platforms.

## Copy your CUDA lib into the app (once per Jetson)
```bash
cd /home/acre/ACRE_Capstone/acre
mkdir -p vendor/jetson
# copy the lib you built in ~/llama.cpp
cp ~/llama.cpp/build/lib/libllama.so vendor/jetson/
```
The app auto-sets `LLAMA_CPP_LIB` to `vendor/jetson/libllama.so` on Jetson. If that file is missing, it falls back to your `~/llama.cpp/build/lib/libllama.so` if present; otherwise it will warn you.

## Make your models visible to the app
If you already downloaded GGUFs to `~/models`, reuse them via a symlink:
```bash
cd /home/acre/ACRE_Capstone/acre
ln -s ~/models models    # if models/ doesn’t already exist
```
Otherwise copy the files into `./models`.

## Install the Python binding with CUDA (inside the app venv)
```bash
cd /home/acre/ACRE_Capstone/acre
source venv/bin/activate
export LLAMA_CPP_LIB=/home/acre/ACRE_Capstone/acre/vendor/jetson/libllama.so  # optional; auto-set if present
pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python
```
Notes:
- `LLAMA_CPP_LIB` just points the wheel to your CUDA-enabled `libllama.so`; it doesn’t rebuild ggml.
- If you prefer the wheel to build its own CUDA lib: `unset LLAMA_CPP_LIB` and run `CMAKE_ARGS="-DLLAMA_CUDA=ON" pip install --force-reinstall --no-cache-dir llama-cpp-python`.

## Run the app
Launch the app normally. On Jetson, it will print which `LLAMA_CPP_LIB` it picked. Set device preference to `auto` or `cuda` in Settings and load your GGUF.
