Installation
============

Quick Install (Python Interface)
---------------------------------

The easiest way to get started is to install via pip::

    pip install snapy

This will install the Python interface with pre-built binaries for Python 3.9-3.13 on Linux (x86_64) and macOS (ARM64).

**Requirements:**

* Python 3.9 or higher
* PyTorch 2.7.x
* NumPy
* kintera >= 1.1.5

Build from Source (Advanced)
-----------------------------

Building from source is recommended only for advanced users who need to:

* Modify the C++ core
* Use custom PyTorch versions
* Access the C++ interface directly
* Develop new features

Prerequisites
~~~~~~~~~~~~~

* CMake 3.20+
* C++17 compatible compiler
* PyTorch 2.7.x with C++ libraries
* NetCDF C library
* kintera >= 1.1.5

Build Steps
~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/chengcli/snapy.git
    cd snapy

2. Install dependencies::

    pip install numpy kintera torch==2.7.1

3. Install NetCDF:

   * **Linux (Ubuntu/Debian)**::

       sudo apt-get install libnetcdf-dev

   * **macOS**::

       brew install netcdf

4. Configure and build::

    # UCX builds require the commux Python package in this environment.
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON
    cmake --build build --parallel 3

5. Install the Python package::

    pip install .

Verification
------------

To verify the installation, you can run::

    python -c "import snapy; print(snapy.__version__)"

This should print the version number without any errors.
