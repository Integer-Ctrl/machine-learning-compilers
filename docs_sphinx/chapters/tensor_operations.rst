Tensor Operations
=================

This section implements a backend for binary tensor contractions and unary tensor permutations. The backend performs the provided tensor
operation exactly as defined by the interface and does not optimize it. Contractions are executed as recursive loops over small GEMM or
Batch-Reduce GEMM (BRGEMM) kernels. Permutations are executed as recursive loops over small transposition kernels.

Backend
-------

User Interface
""""""""""""""

1. setup
^^^^^^^^

**Task**: Begin implementing the ``setup`` function of the class ``einsum::backend::TensorOperation`` for binary tensor contractions.
Parse the configuration parameters passed to the function and generate the corresponding (BR)GEMM kernel at runtime.

File: ``TensorOperation.cpp``

Before generating any kernel, we make sure that all necessary conditions are met. Therefore, we have created a number of checks to verify
that the input configuration for the tensor operation is correct and executable.

.. code-block:: cpp

    mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(dtype_t dtype, prim_t prim_first_touch, prim_t prim_main,
                                                                        prim_t prim_last_touch, std::span<const dim_t> dim_types,
                                                                        std::span<const exec_t> exec_types, std::span<const int64_t> dim_sizes,
                                                                        std::span<const int64_t> strides_in0,
                                                                        std::span<const int64_t> strides_in1,
                                                                        std::span<const int64_t> strides_out)
    {
    hasSetupError = false;
    indexPrimBatch = -1;
    indexPrimK = -1;
    indexPrimM = -1;
    indexPrimN = -1;

    // Validate dimensions
    if (dim_sizes.size() != dim_types.size() || dim_sizes.empty() || dim_types.empty()) {...}

    if (!(strides_in0.size() == dim_sizes.size() && strides_out.size() == dim_sizes.size() && (strides_in1.size() == dim_sizes.size()
        // strides_in1 can be empty for unary operations
        || ((isUnary(prim_first_touch)
        || prim_first_touch == prim_t::none) && (isUnary(prim_main) || prim_main == prim_t::none) && (isUnary(prim_last_touch)
        || prim_last_touch == prim_t::none) && strides_in1.empty())))) {...}

    for (exec_t exec : exec_types) { if (exec == exec_t::shared) {...} }

    // Validate dtype types - currently only fp32 is supported
    if (dtype != dtype_t::fp32) {...}

    if (!isSortedConfiguration(exec_types)) {...}

    if (!isValidPrimConfig(dim_types, exec_types, strides_in0, strides_out)) {...}

    if (!isValidKDim(dim_types, exec_types, strides_in1, prim_main)) {...}

    if (isUnary(prim_main)) { if (!isValidStride(dim_types, strides_in0, stride_t::out) || !isValidStride(dim_types, strides_out, stride_t::out)) {...} }
    else if (isBrgemm(prim_main)) { if (!isValidStride(dim_types, strides_in0, stride_t::in0) 
                                       || !isValidStride(dim_types, strides_in1, stride_t::in1) 
                                       || !isValidStride(dim_types, strides_out, stride_t::out)) {...} }
    else if (prim_main == prim_t::none) { /* Do nothing */ }
    else { release_assert(false, "Unexpected value for the main primitive"); }

    // Validated through isValidPrimConfig that these indices exists
    indexPrimM = findMatch(dim_types, exec_types, dim_t::m, exec_t::prim);
    indexPrimN = findMatch(dim_types, exec_types, dim_t::n, exec_t::prim);

    release_assert(indexPrimM != -1, "Expected a valid index for the M dimension but found none.");
    release_assert(indexPrimN != -1, "Expected a valid index for the N dimension but found none.");


Possible errors that can occur during the configuration verification step are:

.. code-block:: cpp

    err_wrong_dtype,
    err_wrong_dimension,
    err_wrong_primitive,
    err_wrong_first_touch_primitive,
    err_wrong_main_primitive,
    err_wrong_last_touch_primitive,
    err_execution_type_not_supported,
    err_invalid_primitive_configuration,
    err_invalid_first_touch_configuration,
    err_invalid_main_configuration0,
    err_invalid_last_touch_configuration,
    err_invalid_execution_order,
    err_invalid_strides,

If the verification step is successful, we check whether ``prim_first_touch``, ``prim_main``, and ``prim_last_touch`` are defined. If so, we create the corresponding kernel.
``prim_first_touch`` and ``prim_last_touch`` are restricted to unary operations, but ``prim_main`` can be either a unary or a GEMM or BRGEMM.

.. code-block:: cpp
    
    if (prim_first_touch != prim_t::none) {...}

    if (prim_main != prim_t::none)
    {
        if (isBrgemm(prim_main)) {...}
        else if (isUnary(prim_main)) {...}
    }

    if (prim_last_touch != prim_t::none) {...}

    return error_t::success;
    }


Recursive Loops Over Primitives
-------------------------------

1. execute
^^^^^^^^^^

**Task**: Implement the ``execute`` function of the ``einsum::backend::TensorOperation`` class using recursive loops over primitives.
Limit your implementation to single-threaded execution.

The ``execute`` function is used to perform the configured tensor operation on two or three input tensors. Since we also support tensor
operations consisting of only a unary, the second input tensor is not always necessary. We parse the input tensors and call the actual 
executer function, ``execute_dimension``.

.. code-block:: cpp

    void mini_jit::TensorOperation::execute(void const *tensor_in0, void const *tensor_in1, void *tensor_out)
    {
    release_assert(hasSetupError != true, "The setup resulted in a error, do not execute the setup");
    release_assert(tensor_in0 != nullptr, "The tensor_in0 parameter is a nullptr, but should be a valid pointer to memory.");
    release_assert(tensor_out != nullptr, "The tensor_out parameter is a nullptr, but should be a valid pointer to memory.");

    if (isBrgemm(prim_main))
    {
        release_assert(tensor_in1 != nullptr, "The tensor_in1 parameter is a nullptr, but should be a valid pointer to memory");
    }

    char const *ptr_in0 = static_cast<char const *>(tensor_in0);
    char const *ptr_in1 = static_cast<char const *>(tensor_in1);
    char *ptr_out = static_cast<char *>(tensor_out);

    execute_dimension(0, ptr_in0, ptr_in1, ptr_out, true, true);
    }

``execute_dimension`` has three main tasks. First, if defined, check whether the ``prim_first_touch`` or ``prim_last_touch`` primitive
should be executed on the output pointer. Second, if there are outer loops, meaning the tensors have a dimension greater than the dimension
of the used primitive kernel, run a loop over those dimensions until the primitive kernel inside that loop can be called. Third, if there
are no higher dimensions left for iteration, execute the primitive kernels in the correct order.

Compute the ``first_access`` and ``last_access`` and check if higher dimensions are present. If so, execute recursively:

.. code-block:: cpp

    void mini_jit::TensorOperation::execute_dimension(int64_t index_dim, char const *ptr_in0, char const *ptr_in1, char *ptr_out,
                                                  bool first_access, bool last_access)
    {
    uint32_t dtype_bytes = 4;
    int64_t dim_size = dim_sizes[index_dim];
    int64_t stride_in0 = strides_in0[index_dim];
    int64_t stride_in1 = isUnary(prim_main) ? 1 : strides_in1[index_dim];
    int64_t stride_out = strides_out[index_dim];

    // std::cout << "Execute check " << index_dim + 1 << " " << std::endl;
    if (exec_types[index_dim] == exec_t::seq)
    {
        release_assert(exec_types[index_dim] == exec_t::seq, "Expected a sequential loop");

        bool is_first = first_access;
        bool is_last = last_access;

        for (int64_t iDim = 0; iDim < dim_size; iDim++)
        {
        if (dim_types[index_dim] == dim_t::k)
        {
            is_first = first_access && (iDim == 0);
            is_last = last_access && (iDim == (dim_size - 1));
        }

        char const *rec_ptr_in0 = ptr_in0 + iDim * stride_in0 * dtype_bytes;
        char const *rec_ptr_in1 = ptr_in1 + iDim * stride_in1 * dtype_bytes;
        char *rec_ptr_out = ptr_out + iDim * stride_out * dtype_bytes;
        execute_dimension(index_dim + 1, rec_ptr_in0, rec_ptr_in1, rec_ptr_out, is_first, is_last);
        }
    }

If no higher dimension is left for iteration, call the primitive kernels:

.. code-block:: cpp

    else
    {
        release_assert(exec_types[index_dim] == exec_t::prim, "Expected a primitive loop");

        // call first touch kernel if necessary
        if (first_access && prim_first != prim_t::none) {...}

        // call main_kernel kernel
        if (prim_main != prim_t::none)
        {
            if (std::holds_alternative<Unary>(main_kernel)) {...}
            else if (std::holds_alternative<Brgemm>(main_kernel)) {...}
            else {...} // error case
        }

        // call last touch kernel if necessary
        if (last_access && prim_last != prim_t::none) {...}
    }

2. Verify
^^^^^^^^^

**Task**: Verify your implementation against a reference implementation.

We implemented the following tests to verify the functionality of our ``TensorOperation.cpp`` when performing the first, main, and last
primitives in combination with a naive version. The tests are located in the following file: ``TensorOperation.test.cpp``.

.. code-block:: cpp

    // // without outer dimensions
    TEST_CASE("Test tensor operation with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
    TEST_CASE("Test tensor operation with main kernel: gemm", "[tensor_operation][gemm][correctness]")
    TEST_CASE("Test tensor operation with main kernel: brgemm", "[tensor_operation][brgemm][correctness]")

    TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
    TEST_CASE("Test tensor operation with last touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")

    TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: gemm", "[tensor_operation][unary][gemm][correctness]")
    TEST_CASE("Test tensor operation with last touch: unary (zero, relu, copy) & main kernel: gemm", "[tensor_operation][unary][gemm][correctness]")
    TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: gemm & last touch: unary (zero, relu, copy)", "[tensor_operation][unary][gemm][correctness]")
    TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: brgemm", "[tensor_operation][unary][brgemm][correctness]")
    TEST_CASE("Test tensor operation with last touch: unary (zero, relu, copy) & main kernel: brgemm", "[tensor_operation][unary][brgemm][correctness]")
    TEST_CASE("Test tensor operation with first touch: unary (zero, relu, copy) & main kernel: brgemm & last touch: unary (zero, relu, copy)", "[tensor_operation][unary][brgemm][correctness]")
    TEST_CASE("Test tensor operation with outer loop with main kernel: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")

    // with outer dimensions
    TEST_CASE("Test tensor operation with outer loop with main kernel: gemm", "[tensor_operation][gemm][correctness]")
    TEST_CASE("Test tensor operation with outer loop with main kernel: brgemm", "[tensor_operation][brgemm][correctness]")

    TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
    TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy)", "[tensor_operation][unary][correctness]")
    TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: gemm", "[tensor_operation][unary][gemm][correctness]")
    TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy) & main kernel: gemm", "[tensor_operation][unary][gemm][correctness]")

    TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: gemm & last touch: unary (zero, relu, copy)", "[tensor_operation][unary][brgemm][correctness]")
    TEST_CASE("Test tensor operation with outer loop with last touch: unary (zero, relu, copy) & main kernel: brgemm", "[tensor_operation][unary][brgemm][correctness]")
    TEST_CASE("Test tensor operation with outer loop with first touch: unary (zero, relu, copy) & main kernel: brgemm & last touch: unary (zero, relu, copy)", "[tensor_operation][unary][brgemm][correctness]")

Performance Benchmarking
------------------------

1. Performance
^^^^^^^^^^^^^^

**Task**: Benchmark the performance of your implementation for the above examples. Report the measured performance in GFLOPS.

Tensor contraction using the GEMM primitive:

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:0/min_warmup_time:0.300_mean                  4359838 ns      4343934 ns           10 123.593G/s
    BM_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:0/min_warmup_time:0.300_median                4361667 ns      4344882 ns           10 123.564G/s
    BM_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:0/min_warmup_time:0.300_stddev                  17304 ns        17543 ns           10  500.82M/s
    BM_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:0/min_warmup_time:0.300_cv                       0.40 %          0.40 %            10      0.41%


Tensor contraction using the BRGEMM primitive:

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:1/min_warmup_time:0.300_mean                4365885 ns      4350242 ns           10 123.413G/s
    BM_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:1/min_warmup_time:0.300_median              4361928 ns      4346152 ns           10 123.528G/s
    BM_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:1/min_warmup_time:0.300_stddev                14186 ns        14016 ns           10  396.45M/s
    BM_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:1/min_warmup_time:0.300_cv                     0.32 %          0.32 %            10      0.32%


Tensor contraction using the Zero, BRGEMM and ReLU primitives:

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:2/min_warmup_time:0.300_mean      4464672 ns      4448666 ns           10 120.682G/s
    BM_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:2/min_warmup_time:0.300_median    4461153 ns      4444776 ns           10 120.787G/s
    BM_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:2/min_warmup_time:0.300_stddev      14498 ns        14307 ns           10   387.2M/s
    BM_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:2/min_warmup_time:0.300_cv           0.32 %          0.32 %            10      0.32%

2. Own Setups
^^^^^^^^^^^^^

**Task**: Design your own setups. Which setups achieve a high performance and which setups are slow?

- First: Zero & Main: BRGEMM
- A: 262144, B: 262144, C: 1048576

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:3/min_warmup_time:0.300_mean           4449301 ns      4433374 ns           10 121.098G/s
    BM_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:3/min_warmup_time:0.300_median         4448818 ns      4433182 ns           10 121.103G/s
    BM_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:3/min_warmup_time:0.300_stddev            8350 ns         7959 ns           10   217.4M/s
    BM_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:3/min_warmup_time:0.300_cv                0.19 %          0.18 %            10      0.18%


- Last: Relu
- A: 8388608, B: 8192, C: 8388608

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:4/min_warmup_time:0.300_mean                   1694290 ns      1685602 ns           10 9.95364G/s
    BM_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:4/min_warmup_time:0.300_median                 1693287 ns      1685075 ns           10 9.95636G/s
    BM_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:4/min_warmup_time:0.300_stddev                   11637 ns        11124 ns           10 65.7127M/s
    BM_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:4/min_warmup_time:0.300_cv                        0.69 %          0.66 %            10      0.66%


- Main: BRGEMM & Last: RELU
- A: 262144, B: 262144, C: 1048576
- Poor performance due to memory bound

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:5/min_warmup_time:0.300_mean           4474456 ns      4458350 ns           10  120.42G/s
    BM_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:5/min_warmup_time:0.300_median         4476878 ns      4460413 ns           10 120.364G/s
    BM_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:5/min_warmup_time:0.300_stddev            9309 ns         9001 ns           10 243.248M/s
    BM_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:5/min_warmup_time:0.300_cv                0.21 %          0.20 %            10      0.20%


- Main: BRGEMM & Last: RELU
- A: 524288, B: 524288, C: 1048576

.. code-block:: bash

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                            Time             CPU   Iterations      FLOPS
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:6/min_warmup_time:0.300_mean           8660603 ns      8629735 ns           10 124.424G/s
    BM_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:6/min_warmup_time:0.300_median         8651362 ns      8620884 ns           10 124.551G/s
    BM_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:6/min_warmup_time:0.300_stddev           15382 ns        15092 ns           10 217.397M/s
    BM_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:6/min_warmup_time:0.300_cv                0.18 %          0.17 %            10      0.17%


Shared Memory Parallelization
-----------------------------

In the shared memory domain, loops can be parallelized at any point within the nested loop structure. However, to simplify the
implementation, we only parallelize the outermost loops. In other words, we do not parallelize loops that are nested inside
sequential loops.

1. execute_iter_parallel
^^^^^^^^^^^^^^^^^^^^^^^^

**Task**: Implement the function ``execute_iter_parallel``, which parallelizes a binary tensor contraction in the shared memory domain.

File: ``TensorOperation.cpp``

To enable our tensor operations to be processed in parallel, we now accept ``shared`` as an execution type. In the setup, we check if
an execution type of ``shared`` exists. Additionally, we ensure that the k dimensions are not ``shared``.

.. code-block:: cpp

    // Check if shared exists and set parallel flag
    for (exec_t exec : exec_types)
    {
        if (exec == exec_t::shared)
        {
        isParallel = true;
        }
    }

    if (isParallel)
    {
        // K dimension must not be shared
        int32_t kDimExecType = findMatch(dim_types, exec_types, dim_t::k, exec_t::shared);
        if (kDimExecType != -1)
        {
        hasSetupError = true;
        return error_t::err_k_dimension_must_not_be_shared;
        }
    }

Lastly, we check if the execution types are sorted in the correct order:
first ``shared``, then ``sequential``, and finally ``primitive``.

.. code-block:: cpp

    bool mini_jit::TensorOperation::isSortedConfiguration(const std::span<const exec_t> &exec)
    {
    bool seenSequential = false;
    bool seenPrimitive = false;
    for (exec_t exec_type : exec)
    {
        if (exec_type == exec_t::shared && !seenSequential && !seenPrimitive)
        {
        // Nothing to do, shared must be first
        }
        else if (exec_type == exec_t::shared && (seenSequential || seenPrimitive))
        {
        return false;
        }
        else if (exec_type == exec_t::seq && !seenPrimitive)
        {
        seenSequential = true;
        }
        else if (exec_type == exec_t::seq && seenPrimitive)
        {
        return false;
        }
        else if (exec_type == exec_t::prim)
        {
        seenPrimitive = true;
        }
    }

    return true;
    }


The benchmark results of the serial tensor operations had a peak performance of around :math:`120` GFLOPS. Now, we benchmark using OpenMP
and 4 threads, resulting in performance measurements around :math:`420` GFLOPS.

.. code-block:: bash
    :emphasize-lines: 4, 8, 12, 16, 20, 24, 28

    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                                               Time             CPU   Iterations      FLOPS
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_parallel_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:7/min_warmup_time:0.300/threads:4_mean                  5201950 ns      1292865 ns           10 415.261G/s
    BM_parallel_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:7/min_warmup_time:0.300/threads:4_median                5193611 ns      1291863 ns           10 415.579G/s
    BM_parallel_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:7/min_warmup_time:0.300/threads:4_stddev                  32185 ns         4344 ns           10 1.39347G/s
    BM_parallel_tensor_GEMM/size_a:262144/size_b:262144/size_c:1048576/config:7/min_warmup_time:0.300/threads:4_cv                       0.62 %          0.34 %            10      0.34%
    BM_parallel_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:8/min_warmup_time:0.300/threads:4_mean                5195357 ns      1287333 ns           10 417.045G/s
    BM_parallel_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:8/min_warmup_time:0.300/threads:4_median              5167859 ns      1287433 ns           10 417.009G/s
    BM_parallel_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:8/min_warmup_time:0.300/threads:4_stddev                80687 ns         4319 ns           10 1.39959G/s
    BM_parallel_tensor_BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:8/min_warmup_time:0.300/threads:4_cv                     1.55 %          0.34 %            10      0.34%
    BM_parallel_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:9/min_warmup_time:0.300/threads:4_mean      5577549 ns      1313489 ns           10 408.757G/s
    BM_parallel_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:9/min_warmup_time:0.300/threads:4_median    5491313 ns      1310114 ns           10 409.789G/s
    BM_parallel_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:9/min_warmup_time:0.300/threads:4_stddev     353091 ns         9804 ns           10 3.03171G/s
    BM_parallel_tensor_Zero+BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:9/min_warmup_time:0.300/threads:4_cv           6.33 %          0.75 %            10      0.74%
    BM_parallel_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:10/min_warmup_time:0.300/threads:4_mean          5336436 ns      1295288 ns           10 414.481G/s
    BM_parallel_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:10/min_warmup_time:0.300/threads:4_median        5306927 ns      1295453 ns           10 414.427G/s
    BM_parallel_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:10/min_warmup_time:0.300/threads:4_stddev          95431 ns         1975 ns           10  632.06M/s
    BM_parallel_tensor_Zero+BRGEMM/size_a:262144/size_b:262144/size_c:1048576/config:10/min_warmup_time:0.300/threads:4_cv               1.79 %          0.15 %            10      0.15%
    BM_parallel_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:11/min_warmup_time:0.300/threads:4_mean                  2954501 ns       735408 ns           10 22.8172G/s
    BM_parallel_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:11/min_warmup_time:0.300/threads:4_median                2947921 ns       735807 ns           10 22.8011G/s
    BM_parallel_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:11/min_warmup_time:0.300/threads:4_stddev                  55255 ns         9959 ns           10 307.823M/s
    BM_parallel_tensor_Relu/size_a:8388608/size_b:8192/size_c:8388608/config:11/min_warmup_time:0.300/threads:4_cv                       1.87 %          1.35 %            10      1.35%
    BM_parallel_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:12/min_warmup_time:0.300/threads:4_mean          5243909 ns      1301545 ns           10 412.507G/s
    BM_parallel_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:12/min_warmup_time:0.300/threads:4_median        5239656 ns      1299425 ns           10 413.161G/s
    BM_parallel_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:12/min_warmup_time:0.300/threads:4_stddev          35856 ns         9430 ns           10 2.98182G/s
    BM_parallel_tensor_BRGEMM+RELU/size_a:262144/size_b:262144/size_c:1048576/config:12/min_warmup_time:0.300/threads:4_cv               0.68 %          0.72 %            10      0.72%
    BM_parallel_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:13/min_warmup_time:0.300/threads:4_mean         10136019 ns      2524142 ns           10 425.392G/s
    BM_parallel_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:13/min_warmup_time:0.300/threads:4_median       10143290 ns      2523724 ns           10 425.459G/s
    BM_parallel_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:13/min_warmup_time:0.300/threads:4_stddev          59898 ns         7583 ns           10 1.27538G/s
    BM_parallel_tensor_BRGEMM+RELU/size_a:524288/size_b:524288/size_c:1048576/config:13/min_warmup_time:0.300/threads:4_cv               0.59 %          0.30 %            10      0.30%


Optimization Passes
-------------------

1. Intermediate Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Task**: Develop an IR that supports transformations such as dimension reordering, dimension splitting and fusing dimensions.

We created a struct ``TensorConfig`` in ``TensorConfig.h`` to support transformations and optimization passes on our tensor operation.
This configuration contains all the input data for our tensor operation. Before handing this configuration over to our tensor operation
setup, we run our optimization passes over it. We also added a ``equal(const TensorConfig &config1, const TensorConfig config2)`` and
``to_string()`` method for testing purposes.

2. Optimization Passes
^^^^^^^^^^^^^^^^^^^^^^

**Task**: Implement optimization passes. At a minimum, support primitive identification and shared memory parallelization.

**Dimension Reordering Fusing**

We added dimension reordering to our optimization passes to improve dimension fusion.
The idea is to move any dimension X next to dimension Y if they are the same type and the ``Stride(X) = |Y| * Stride(Y)`` condition is met.

.. code-block:: cpp

    void mini_jit::TensorOptimization::_dimension_reordering_fusing(TensorConfig &config)

**Dimension Splitting**

We added dimension splitting to our optimization passes. The idea is to check if any dimension is greater than or equal to 256. If so, we
split the dimension into two, starting at the floor of the square root of the dimension size, and check if it is a dominator. Otherwise,
we decrement the possible dominator and test until it is 2. If a dominator is found, the dimension is split.

.. code-block:: cpp

    void mini_jit::TensorOptimization::_dimension_splitting(TensorConfig &config)
    
**Dimension Fusing**

We added dimension fusion to our optimization passes. The idea is to check if two neighboring dimensions have the same dimension type and
if the product of both dimension sizes is less than or equal to 256. We also check if the condition ``Stride(X) = |Y| * Stride(Y)`` is true.
If so, we fuse the two dimensions.

.. code-block:: cpp

    void mini_jit::TensorOptimization::_dimension_fusing(TensorConfig &config)

**Dimension Reordering Shared**

We added dimension reordering to our optimization passes for better shared identification. We reorder sequential loops with other sequential
loops and shared loops with other shared loops. We sort by strides but discourage any k-dimensional or repeating dimensions. We sum the
strides and divide by eight if it is a k-dimensional stride and divide by two if it is a repeating dimension, excluding the c-dimension.

.. code-block:: cpp

    void mini_jit::TensorOptimization::_dimension_reordering_shared(TensorConfig &config)
    {
    ...
        uint64_t value = (*jStrideIn0 * *jStrideIn0) + (*jStrideIn1 * *jStrideIn1) + (*jStrideOut * *jStrideOut);

        // value/8 if we have a k-dimension
        value >>= (*jDim == TensorConfig::dim_t::k) * 3;

        // value/2 if we have the same dimension type as the last dimension, but not for c dimension
        value >>= (*jDim == previous_dim && *jDim != TensorConfig::dim_t::c) * 1;
    ...
    }


**Primitive Identification**

We added primitive identification support to our optimization pass.
The following rules are applied based on the dimension type:
- m-dimension: search m-dimension with a unit-stride in the first input 
- n-dimension: search in the second input and in the output for the smallest stride
- k-dimension: only applies to GEMM or BRGEMM, search for unit--stride in the second input
- second-k-dimension: only applies to BRGEMM, search for the smallest stride in first input or second input, but not select the already found k-dimension

Additionally, we do not modify any existing chosen primitives by the user.

.. code-block:: cpp

    void mini_jit::TensorOptimization::_primitive_identification(TensorConfig &config)


**Shared Identification**

We added shared identification support to our optimization pass. At most, we can convert to shared until the first primitive arises or the
first k-dimensional primitive. We only tag as many dimensions as are shared, i.e., if the first dimension is perfectly divisible by the
number of OpenMP threads in use, we do not convert any further dimensions as shared. Additionally, we only convert to shared if the
unbalanced ratio of the dimensions is greater than 1%.
:code:`(shared_dimensions_size % thread_count) / shared_dimensions_size < 1%`.

.. code-block::

    void mini_jit::TensorOptimization::_shared_identification(TensorConfig &config)


3. Lowering
^^^^^^^^^^^

**Task**: Lower the optimized IR code to your tensor operation backend. Verify the correctness of the optimizations.

Since our IR is the struct ``TensorConfig``, we only need to provide the configuration to our optimization, and then to our tensor operation
setup. This order ensures that the optimizer creates a valid configuration for the tensor operation.

.. code-block:: cpp

    mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(const TensorConfig &config)
    {
    mini_jit::TensorOptimization optimization;
    TensorOperation::config = optimization.optimize(config);

    return setup_no_optimization(TensorOperation::config.dtype, TensorOperation::config.first_touch, TensorOperation::config.main,
                                 TensorOperation::config.last_touch, TensorOperation::config.dim_types, TensorOperation::config.exec_types,
                                 TensorOperation::config.dim_sizes, TensorOperation::config.strides_in0, TensorOperation::config.strides_in1,
                                 TensorOperation::config.strides_out);
    }

Our ``TensorOptimization`` 's ``optimize`` method executes individual optimization passes on the config struct.

.. code-block:: cpp

    mini_jit::TensorConfig mini_jit::TensorOptimization::optimize(TensorConfig config)
    {
    _dimension_reordering_fusing(config);

    _dimension_splitting(config);

    _dimension_fusing(config);

    _primitive_identification(config);

    _dimension_reordering_shared(config);

    // Only call shared after reordering it only parallelize the first loops until the first seq k-loops at maximum
    _shared_identification(config);
    return config;
    }


4. Performance
^^^^^^^^^^^^^^

**Task**: Benchmark the performance of your implementation for the above matrix multiplication and tensor contraction examples. Report the measured performance in GFLOPS.

File: ``TensorOptimization.bench.cpp``

**Matrix multiplication example**

.. code-block:: bash

    -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                              Time             CPU   Iterations      FLOPS
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_optimized_tensor_GEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:0/min_warmup_time:0.300_mean        1316172 ns      1303763 ns           10 411.786G/s
    BM_optimized_tensor_GEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:0/min_warmup_time:0.300_median      1313935 ns      1303515 ns           10 411.864G/s
    BM_optimized_tensor_GEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:0/min_warmup_time:0.300_stddev         7770 ns         1120 ns           10   353.7M/s
    BM_optimized_tensor_GEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:0/min_warmup_time:0.300_cv             0.59 %          0.09 %            10      0.09%

**Tensor contraction example**

.. code-block:: bash

    -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                              Time             CPU   Iterations      FLOPS
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_optimized_tensor_BRGEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:1/min_warmup_time:0.300_mean      1310327 ns      1295379 ns           10 414.451G/s
    BM_optimized_tensor_BRGEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:1/min_warmup_time:0.300_median    1307359 ns      1295362 ns           10 414.456G/s
    BM_optimized_tensor_BRGEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:1/min_warmup_time:0.300_stddev       8579 ns         1229 ns           10 393.184M/s
    BM_optimized_tensor_BRGEMM/size_a:2560000/size_b:2560000/size_c:2560000/config:1/min_warmup_time:0.300_cv           0.65 %          0.09 %            10      0.09%

5. Own Examples
^^^^^^^^^^^^^^^

**Task**: Demonstrate the capabilities of your optimization passes using your own examples.

We tested our optimization passes in ``TensorOptimization.test.cpp``. One exhaustive test case is shown below. This optimization involves
primitive ``reordering``, ``fusing``, ``primitive identification``, and ``shared identification``. In addition to testing the correctness of the tensor
configuration after the optimization passes, we also test the correctness of the tensor operation.

.. code-block:: cpp
    :emphasize-lines: 5-18, 20-33, 35-36

    TEST_CASE("Test tensor operation with optimization dimension test reordering and fusing", "[tensor_optimization][gemm][correctness]")
    {
    using namespace mini_jit;

    mini_jit::TensorConfig config{
        mini_jit::TensorConfig::prim_t::none,  // first_touch
        mini_jit::TensorConfig::prim_t::gemm,  // main
        mini_jit::TensorConfig::prim_t::none,  // last touch
        {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
        mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k},  // dim_types
        {mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq,
        mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::seq},  // exec_types
        {32, 8, 32, 5, 32, 32},                                                                                           // dim_sizes
        {0, 1024, 1, 0, 0, 32},                                                                                           // strides_in0
        {8192, 1024, 0, 8192 * 32, 32, 1},                                                                                // strides_in1
        {1024, 0, 1, 32768, 32, 0},                                                                                       // strides_out
        mini_jit::TensorConfig::dtype_t::fp32,                                                                            // dtype_t
    };

    mini_jit::TensorConfig expected{
        mini_jit::TensorConfig::prim_t::none,  // first_touch
        mini_jit::TensorConfig::prim_t::gemm,  // main
        mini_jit::TensorConfig::prim_t::none,  // last touch
        {mini_jit::TensorConfig::dim_t::n, mini_jit::TensorConfig::dim_t::k, mini_jit::TensorConfig::dim_t::m, mini_jit::TensorConfig::dim_t::n,
        mini_jit::TensorConfig::dim_t::k},  // dim_types
        {mini_jit::TensorConfig::exec_t::shared, mini_jit::TensorConfig::exec_t::seq, mini_jit::TensorConfig::exec_t::prim,
        mini_jit::TensorConfig::exec_t::prim, mini_jit::TensorConfig::exec_t::prim},  // exec_types
        {5 * 32, 8, 32, 32, 32},                                                       // dim_sizes
        {0, 1024, 1, 0, 32},                                                           // strides_in0
        {8192, 1024, 0, 32, 1},                                                        // strides_in1
        {1024, 0, 1, 32, 0},                                                           // strides_out
        mini_jit::TensorConfig::dtype_t::fp32,                                         // dtype_t
    };

    mini_jit::TensorOperation tensor_op;
    TensorOperation::error_t err = tensor_op.setup(config);

    INFO(tensor_op.get_config().to_string());

    REQUIRE(err == TensorOperation::error_t::success);
    REQUIRE_FALSE(mini_jit::TensorConfig::equals(config, tensor_op.get_config()));
    REQUIRE(mini_jit::TensorConfig::equals(expected, tensor_op.get_config()));

    GenerationTest test(32, 32, 32, 32 * 1 * 32 * 8 * 1 * 1, 32 * 32 * 1 * 8 * 32 * 5, 1 * 32 * 32 * 1 * 32 * 5);
    test.SetUp(TestInfill::Random);

    tensor_op.execute(test.matrix_a.data(), test.matrix_b.data(), test.matrix_c.data());

    for (int64_t i0 = 0; i0 < expected.dim_sizes[0]; i0++)
    {
        for (int64_t i1 = 0; i1 < expected.dim_sizes[1]; i1++)
        {
        uint64_t offset_a = i0 * expected.strides_in0[0] + i1 * expected.strides_in0[1];
        uint64_t offset_b = i0 * expected.strides_in1[0] + i1 * expected.strides_in1[1];
        uint64_t offset_c = i0 * expected.strides_out[0] + i1 * expected.strides_out[1];
        test.naive_matmul_M_N_K_Batch(test.matrix_a.data() + offset_a, test.matrix_b.data() + offset_b,
                                        test.matrix_c_verify.data() + offset_c, 32, 32, 32, 32 * 32, 32 * 32);
        }
    }

    test.verify_matmul(test.matrix_c_verify.data(), test.matrix_c.data(), test.matrix_c.size());
    }

Unary Operations
----------------

The support for none transposed unary operations was already added in the chapter :ref:`unary_primitives`.
Therefore, we only needed to include the transpose operation additionally.

We added transpose support to parse our ``TensorConfig`` in the ``TensorOperation.cpp``.
And validated with some additional tests: File: ``TensorOperation.test.cpp``.

.. code-block:: cpp

    bool mini_jit::TensorOperation::isValidPrimStrides(const std::span<const TensorConfig::dim_t> &dim,
                                                    const std::span<const TensorConfig::exec_t> &exec,
                                                    const std::span<const int64_t> &strides_in0, const std::span<const int64_t> &strides_out,
                                                    const TensorConfig::prim_t main_prim)
    {
    // ...

    // no transpose
    if (isExpectedStride(1, indexM, strides_in0) && isExpectedStride(1, indexM, strides_out))
    {
        return true;
    }

    // Check transpose in unary op
    if (isUnary(main_prim) && isExpectedStride(1, indexM, strides_in0) && isExpectedStride(1, indexN, strides_out))
    {
        isTranspose = true;
        return true;
    }
    
    // ...
    }


.. attention::
  DOCUMENTATION IS NOT COMPLETE YET.

  https://github.com/scalable-analyses/pbtc/tree/main/lab/tensor_op#unary-operations