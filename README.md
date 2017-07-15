# fftw-wrapper
RAII-based C++ wrapper of FFTW

## Requirements

Use -lfftw3 to build programs with this wrapper. It's more intuitive to use than fftw, and it cleans up after itself.
This currently does not build for calculations with floats, though it could be modified to do so.
