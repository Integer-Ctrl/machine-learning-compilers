.. _getting_started_building_project:

.. role:: raw-html(raw)
    :format: html

Building the Project
====================


Requirements
------------

To use this project ``CMake`` is required.
You can check if CMake is installed by running

.. code-block:: bash

    cmake --version

.. note::
    ``CMake`` can be `downloaded <https://cmake.org/download/#latest>`_ or installed OS specific through other methods

    *Linux*
        A simple way is using ``apt-get`` by running the command

        .. code-block:: bash

            sudo apt-get install cmake

Building
--------

1. Download the `git repository <https://github.com/Integer-Ctrl/machine-learning-compilers>`_ with git

    HTTPS

    .. code-block:: bash

        git clone https://github.com/Integer-Ctrl/machine-learning-compilers

    SSH

    .. code-block:: bash

        git clone git@github.com:Integer-Ctrl/machine-learning-compilers

2. Go into the project folder. Your current path should look like this ``../machine-learning-compilers``.

3. Now create a new folder called ``build`` with

    .. code-block:: bash

        mkdir build

4. Go into this directory. Your current path should look like this ``../machine-learning-compilers/build``.

5. Now we can start with CMake. Run the following command

    .. code-block:: bash

        cmake .. -DCMAKE_BUILD_TYPE=<Type>

    Supported values for ``<Type>`` are **Release** and **Debug**.
    If only ``cmake ..`` is used the Release build is selected.

    The most desired command might be:

    .. code-block:: bash

        cmake .. -DCMAKE_BUILD_TYPE=Release

    .. note::

        With the Option ``-G`` a Generator can be defined used to create the make files and compile the Code.
        All available Generators can be found at the bottom of the :raw-html:`<br/>` ``cmake --help`` text.
        An Example could look like this

        .. code-block:: bash

            cmake .. -G "MinGW Makefiles"

        
    .. important::

        When using a multi-config Generator, i.e. Ninja Multi-Config, Visual Studio Generators, Xcode, 
        ``-DCMAKE_BUILD_TYPE=<Type>`` is not needed, and the build type is configured on compilation.
        
        Therefore, this cmake build command is used:

        .. code-block:: 

            cmake --build . --config Release --target benchmark

        Options for ``--config`` are **Release** and **Debug**. :raw-html:`</br>`
        Options for ``--target`` are **benchmark**, **microkernel**, **loops**, and **test**

6. Now we can build the project. The most desired command might be

    .. code-block:: bash

        cmake --build . --target simulation

    Options for ``--target`` are **benchmark**, **microkernel**, **loops**, and **test**


    +--------------------+--------------------------------------------------------------------------------------------------------------------+
    | Option             |  Description                                                                                                       |
    +====================+====================================================================================================================+
    | benchmark          | Build the benchmark executable to run throughput und latency benchmarks.                                           |
    +--------------------+--------------------------------------------------------------------------------------------------------------------+
    | microkernel        | Build the microkernel executable to run a simple 16x6 matrix kernels                                              |
    +--------------------+--------------------------------------------------------------------------------------------------------------------+
    | loops              | Build the loops executable to run matrix kernels with loops over K, M or N.                                        |
    +--------------------+--------------------------------------------------------------------------------------------------------------------+
    | test               | Builds the unit test executable                                                                                    |
    +--------------------+--------------------------------------------------------------------------------------------------------------------+

Running the Executables & Tests
------------------------------

The executables have been build in to the ``../machine-learning-compilers/build`` directory with their corresponding name.
E.g. If ``test`` is built than the executable name is ``test``, 
for ``microkernel`` the executable name would be ``microkernel``, etc.

All the executables can be found in ``../machine-learning-compilers/build``.
The available executables are ``benchmark``, ``microkernel``, ``loops``, and ``test``.

.. note::
    They are available when build with their respective ``--target``

E.g. the ``benchmark`` executable can be run with the following command:

.. code-block::

    ./benchmark
