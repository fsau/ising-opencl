# Metropolis algorithm for 2D Ising model with OpenCL

<sub><sup>Note: This code is based on [some examples](https://github.com/rsnemmen/OpenCL-examples/) I found online.</sup><sub>

To run, first build the program with make:

```
cd ./src/
make
```

That will create a file `ising.out` which is the main executable and can be run standalone.

## Configuration

The main configuration is choosing which platform/device to use. You can query the available OpenCL platforms with the program `clinfo`, then configure `ising.h` definitions properly.
