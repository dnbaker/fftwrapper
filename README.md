# fftw-wrapper
RAII-based C++ wrapper of FFTW

## Requirements

Use -lfftw3 to build programs with this wrapper. It's more intuitive to use than fftw, and it cleans up after itself.

It now supports floats, doubles, and long doubles, though you'll need to link against the appropriate -lfftw*.
