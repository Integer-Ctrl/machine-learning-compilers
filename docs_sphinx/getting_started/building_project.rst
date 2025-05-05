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
        Options for ``--target`` are **benchmarks** and **tests**

6. Now we can build the project. The most desired command might be

    .. code-block:: bash
        
    Options for ``--target`` are **benchmark** and **tests**


    +--------------------+--------------------------------------------------------------------------------------------------------------------+
    | Option             |  Description                                                                                                       |
    +====================+====================================================================================================================+
    | benchmarks         | Build the benchmarks executable.                                                                                   |
    +--------------------+--------------------------------------------------------------------------------------------------------------------+
    | tests              | Builds the tests executable.                                                                                       |
    +--------------------+--------------------------------------------------------------------------------------------------------------------+

Running the Benchmarks & Tests
------------------------------

The executables have been build in to the ``../machine-learning-compilers/build`` directory with their corresponding name.
E.g. If ``tests`` is built than the executable name is ``tests``, 
for ``benchmarks`` the executable name would be ``benchmarks``, etc.

All the executables can be found in ``../machine-learning-compilers/build``.
The available executables are ``benchmarks`` and ``tests``.

.. note::
     
    They are available when build with their respective ``--target``

E.g. the ``benchmarks`` executable can be run with the following command:

.. code-block::

    ./benchmarks

The most desired command for the ``benchmarks`` might be:

.. code-block::

    ./benchmarks --benchmark_counters_tabular=true --benchmark_repetitions=10 --benchmark_report_aggregates_only=true