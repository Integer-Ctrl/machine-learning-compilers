Einsum Trees
============

This chapter expands the capabilities of our tensor compiler by adding support for einsum trees. Specifically, we execute einsum trees
by mapping them to a tree of unary and binary tensor operations. These operations can then be executed by our tensor operation backend.

Lowering
--------

An einsum tree represents multiple dependent tensor operations. It may contain nodes with two children (contractions), nodes with one
child (transpositions), or leaf nodes (input tensors). The output tensor of the root node represents the result of the entire tree.

1. Parsing
^^^^^^^^^^

**Task**: Implement a function that parses the string representation of a tree and the numerically sorted dimension sizes.

To complete the task we have to parse a string of the format, where a tensor is represented by a list of numerical dimension e.g. ``[3,2,4,1]``.
Further a operation is written by a ``->`` symbol e.g. a contraction ``[<input0>][<input1>]->[<output>]`` or a unary ``[<input>]->[<output>]``.
Also note that we can produce dependencies on other operation by nesting a operation inside a Tensor e.g. ``[[<in0>][<in1>]->[<op0>]]->[<output>]``,
where the output tensor of the nested operations defines the shape of the immediate input tensor.

A complete example of a einsum tree input is the string: 

.. code-block:: 

  [[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]

And an additional list of dimensions sizes sorted by the numerical id of the dimension:

.. code-block::

  60,60,20,20,8,8,8,8,8,8


To achieve this goal, we implemented a struct called ``EinsumNode`` to parse the string representation of a tree and the numerically sorted dimension sizes.
This structure holds one node of the tree, its possible children, dimension sizes, and a tensor representing an intermediate or final
(root node) result.

.. code-block:: cpp

    struct EinsumNode
    {
      NodeType type;
      int32_t input_tensor_index = -1;
      float *tensor = nullptr;
      mini_jit::TensorOperation tensor_op;

      // Always filled — dims of the output tensor
      std::vector<int64_t> output_dim_ids;

      // Pointers to children
      EinsumNode *left = nullptr;
      EinsumNode *right = nullptr;

      /**
        * Gets a string representation of the einsum tree.
        */
      std::string to_string() const;

      /**
        * Get the size of the tensor represented by this node.
        *
        * @param dim_sizes A vector of dimension sizes corresponding to the output dimensions.
        */
      int64_t get_size(const std::vector<int64_t> dim_sizes) const;

    private:
      /**
        * This method recursively formats the node and its children into a string.
        *
        * @param depth The current depth in the tree, used for indentation.
        * @param connection A string representing the connection type.
        * @param depthString A string representation of the current depth.
        * @return A formatted string representing the einsum tree.
        */
      std::string _to_string(uint depth, std::string connection, std::string depthString) const;
    };

Then, we implemented the logic to parse the string into a set of nodes in the ``parse_tree_no_optimization(bool)`` method. This method also indicates whether
the parsing was successful, ``ErrorParse``.

.. code-block:: cpp

    ErrorParse parse_tree_no_optimization(bool build_operators);

    // AND

    EinsumNode *parse_node(size_t &pos, const std::string &str);


2. Lowering
^^^^^^^^^^^

**Task**: Implement a function that lowers the contraction and permutation nodes of the tree to your tensor operation backend.

To lower our tree to the tensor operation backend, each ``EinsumNode`` is lowered to one ``TensorConfig``. This configuration can then be
passed to the ``TensorOperation``. The main method for doing so is ``lower_node``.

.. code-block::

    TensorConfig lower_node(const EinsumNode *node);
    

3. Optimization
^^^^^^^^^^^^^^^

**Task**: Run your optimization passes on the lowered tensor operations.

Our ``EinsumTree`` has an ``execute()`` method. This method recursively executes one tensor operation per ``EinsumNode``. Therefore, the
``TensorConfig`` of the current node is used as input for the ``TensorOperation``. Since our ``TensorOperation`` receives a ``TensorConfig``
as input, it runs all optimization passes on the config before executing the operation. Therefore, no additional step is needed to run
optimization passes on the lowered tensor operations.

To ensure the success of all tensor operations, the methods return an ``ErrorExecute``.

.. code-block:: cpp

    ErrorExecute execute(std::vector<void *> tensors);

    // AND

    ErrorExecute execute_node(EinsumNode *node);


4. Performance
^^^^^^^^^^^^^^

**Task**: Benchmark the performance of your implementation for the above examples. Report the measured performance in GFLOPS.

**First Example**

Einsum tree:

.. code-block:: vim

   0,1,2,3,4
   ├─ 7,3,4
   |  ├─ 8,4
   |  └─ 7,3,8
   └─ 0,1,2,7
      ├─ 1,2,5,7
      |  ├─ 2,6,7
      |  └─ 1,5,6
      └─ 0,5

String representation:

.. code-block::

   [[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]

Dimension sizes (sorted by numerical ID):

.. code-block::

   100,72,128,128,3,71,305,32,3

**Second Example**

Einsum tree:

.. code-block:: vim

   0,1,2,3
   ├─ 0,4,7,8,2,3
   |  ├─ 7,8,5,6,2,3
   |  |  ├─ 8,6,9,3
   |  |  |  └─ 3,6,8,9
   |  |  └─ 7,5,2,9
   |  |     └─ 2,5,7,9
   |  └─ 0,4,5,6
   └─ 1,4,7,8

String representation:

.. code-block::

   [[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]

Dimension sizes (sorted by numerical ID):

.. code-block::

   60,60,20,20,8,8,8,8,8,8


Performing a benchmark on both Einsum Trees, we get the following performance:

.. code-block:: bash
    :emphasize-lines: 4, 8
  
    ------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                         Time             CPU   Iterations      FLOPS
    ------------------------------------------------------------------------------------------------------------------------------
    BM_einsum_tree_first_example/einsum_tree:0/min_warmup_time:0.300_mean     280607767 ns    279227060 ns           10  142.03G/s
    BM_einsum_tree_first_example/einsum_tree:0/min_warmup_time:0.300_median   277448741 ns    276113901 ns           10 143.454G/s
    BM_einsum_tree_first_example/einsum_tree:0/min_warmup_time:0.300_stddev    10891315 ns     10817141 ns           10 5.02424G/s
    BM_einsum_tree_first_example/einsum_tree:0/min_warmup_time:0.300_cv            3.88 %          3.87 %            10      3.54%
    BM_einsum_tree_second_example/einsum_tree:1/min_warmup_time:0.300_mean     12415368 ns     12304609 ns           10 249.808G/s
    BM_einsum_tree_second_example/einsum_tree:1/min_warmup_time:0.300_median   12389493 ns     12296296 ns           10 249.965G/s
    BM_einsum_tree_second_example/einsum_tree:1/min_warmup_time:0.300_stddev      98826 ns        90496 ns           10 1.83123G/s
    BM_einsum_tree_second_example/einsum_tree:1/min_warmup_time:0.300_cv           0.80 %          0.74 %            10      0.73%

- **First Example**: :math:`143.4` GiB/s
- **Second Example**: :math:`249.9` GiB/s


Optimization
------------

1. Optimization Pass
^^^^^^^^^^^^^^^^^^^^

**Task**: Develop an optimization pass for einsum trees that applies the three transformations.

Three transformation that can be performed on the einsum tree are reorder, swap and permutation insert.

- **Reorder**: Operates on individual tensor to reorder its dimensions such that next involved tensor operation has a better performance.
- **Swap**: Swap the two children of a contraction to mitigate the usage of permutation inserts.
- **Permutation Insert**: Inserts an additional node in the tree to perform a reordering for the next tensor operation.

Reorder Node
""""""""""""

For the reorder node we divided the optimization into an different pass for the left and the right node.

For the reorder pass, we divided the transformation into two methods. The first is ``reorder_left_node``, which reorders the left child node
of a node. The second method is ``reorder_right_node``, which is designed to reorder the right child node of a node.
This division is due to the fact that the left node requires the M dimension as the unit-stride, while the right node requires the K1 dimension
as unit-stride.

*Left Node:*

The method ``reorder_left_node`` checks if the last dimensions of the left child node are ``KM`` dimensions. If not, it permutes the dimensions to
move ``KM`` to the rightmost location. First, we determine the index of the first occurrence of the ``M`` and ``K`` dimension in the left
child node of the node from right to left. If they are already in order, we return. Otherwise, we place them at the desired index location.

.. code-block:: cpp

    void mini_jit::EinsumTree::reorder_left_node(EinsumNode *node)
    {
    int32_t indexLeftMDim = findMDim(node);
    int32_t indexLeftKDim = findKDim(node, true);

    if (indexLeftKDim == static_cast<int32_t>(node->left->output_dim_ids.size()) - 2 &&
        indexLeftMDim == static_cast<int32_t>(node->left->output_dim_ids.size()) - 1)
    {
        // Already ordered
        return;
    }

    std::vector<int64_t> reorderDimIds = node->left->output_dim_ids;  // copy
    // iter_swap -> swap values between two indices
    std::iter_swap(reorderDimIds.begin() + indexLeftMDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 1);
    if (indexLeftKDim != static_cast<int32_t>(node->left->output_dim_ids.size()) - 1)
    {
        std::iter_swap(reorderDimIds.begin() + indexLeftKDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 2);
    }
    else  // Swapped mDim with kDim -> kDim was placed at indexLeftMDim
    {
        std::iter_swap(reorderDimIds.begin() + indexLeftMDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 2);
    }
  
    void mini_jit::EinsumTree::reorder_left_node(EinsumNode *node)
    {
      release_assert(node->left != nullptr, "Expected a valid pointer.");

      int32_t indexLeftMDim = findMDim(node);
      int32_t indexLeftKDim = findKDim(node, true);

      release_assert(indexLeftKDim != -1, "Did not find a 'k' dimension in left child.");
      release_assert(indexLeftMDim != -1, "Did not find a 'm' dimension in left child.");

      if (indexLeftKDim == static_cast<int32_t>(node->left->output_dim_ids.size()) - 2 &&
          indexLeftMDim == static_cast<int32_t>(node->left->output_dim_ids.size()) - 1)
      {
        // Already ordered
        return;
      }

      std::vector<int64_t> reorderDimIds = node->left->output_dim_ids;  // copy
      // iter_swap -> swap values between two indices
      std::iter_swap(reorderDimIds.begin() + indexLeftMDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 1);
      if (indexLeftKDim != static_cast<int32_t>(node->left->output_dim_ids.size()) - 1)
      {
        std::iter_swap(reorderDimIds.begin() + indexLeftKDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 2);
      }
      else  // Swapped mDim with kDim -> kDim was placed at indexLeftMDim
      {
        std::iter_swap(reorderDimIds.begin() + indexLeftMDim, reorderDimIds.begin() + node->left->output_dim_ids.size() - 2);
      }

      if (node->left->type == NodeType::Leaf)
      {
        // Add additional Permutation Node
        EinsumNode *reorderNode = new EinsumNode();
        reorderNode->type = NodeType::Transposition;
        reorderNode->output_dim_ids = std::move(reorderDimIds);

        reorderNode->left = node->left;
        node->left = reorderNode;
      }
      else
      {
        // Only reorder the output of the left operation
        node->left->output_dim_ids = std::move(reorderDimIds);
      }
    }

*Right Node:*

The method ``reorder_right_node`` checks if the last dimensions of the right child node are ``NK`` dimensions. If not, it permutes the dimensions to
move ``NK`` to the rightmost location. First, we determine the index of the first occurrence of the ``N`` and ``K`` dimension in the right
child node of the node from right to left. If they are already in order, we return. Otherwise, we place them at the desired index location.

.. code-block:: cpp

    void mini_jit::EinsumTree::reorder_right_node(EinsumNode *node)
    {
    int32_t indexRightNDim = findNDim(node);
    int32_t indexRightKDim = findKDim(node, false);

    if (indexRightNDim == static_cast<int32_t>(node->right->output_dim_ids.size()) - 2 &&
        indexRightKDim == static_cast<int32_t>(node->right->output_dim_ids.size()) - 1)
    {
        // Already ordered
        return;
    }

    std::vector<int64_t> reorderDimIds = node->right->output_dim_ids;  // copy
    // iter_swap -> swap values between two indices
    std::iter_swap(reorderDimIds.begin() + indexRightKDim, reorderDimIds.begin() + node->right->output_dim_ids.size() - 1);
    if (indexRightNDim != static_cast<int32_t>(node->right->output_dim_ids.size()) - 1)
    {
        std::iter_swap(reorderDimIds.begin() + indexRightNDim, reorderDimIds.begin() + node->right->output_dim_ids.size() - 2);
    }
    else  // Swapped kDim with nDim -> nDim was placed at indexRightKDim
    {
        std::iter_swap(reorderDimIds.begin() + indexRightKDim, reorderDimIds.begin() + node->right->output_dim_ids.size() - 2);
    }ode:*

The right node reordering is very similar to the left node reordering, but it orders K at the last index and N at the second-last index.

Insert Permutation Node
"""""""""""""""""""""""

The permutation node is only added if the ``reorder_left_node`` or ``reorder_right_node`` method reorders a leaf node i.e. a node that is provided by the user.

The code fragment of a permutation node in the ``reorder_left_node`` method:

.. code-block:: cpp

    void mini_jit::EinsumTree::reorder_left_node(EinsumNode *node)
    {
        ...
        if (node->left->type == NodeType::Leaf)
        {
            // Add additional Permutation Node
            EinsumNode *reorderNode = new EinsumNode();
            reorderNode->type = NodeType::Transposition;
            reorderNode->output_dim_ids = std::move(reorderDimIds);

            reorderNode->left = node->left;
            node->left = reorderNode;
        }
        ...
    }

And for the ``reorder_right_node`` method:

.. code-block:: cpp

    void mini_jit::EinsumTree::reorder_right_node(EinsumNode *node)
    {
        ...
        if (node->right->type == NodeType::Leaf)
        {
            // Add additional Permutation Node
            EinsumNode *reorderNode = new EinsumNode();
            reorderNode->type = NodeType::Transposition;
            reorderNode->output_dim_ids = std::move(reorderDimIds);

            reorderNode->left = node->right;
            node->right = reorderNode;
        }
        ...
    }


Swap Contraction Nodes
""""""""""""""""""""""

The swap method allows optimization so that the order of the input tensor does not affect the performance of the contraction.
Therefore, the idea behind the swap method is to check if a node's unit-stride dimension is of type ``N``.
If this is the case, we swap its children to obtain a unit-stride dimension in the first input tensor (left child node). 
We use the C++ ``swap`` method to swap the child nodes of a node, swapping the left child node pointer with the right child node pointer.

.. code-block:: cpp

    void mini_jit::EinsumTree::conditional_swap(mini_jit::EinsumTree::EinsumNode *node)
    {
        // Ensure that 'm' dimension has unit stride
        if (is_unit_stride_n(node))
        {
            std::swap(node->left, node->right);
        }
    }

Heuristic
"""""""""

To apply the optimization passes to three, we used a heuristic to decided when and how the optimization are applied.
We do the following steps:

1. First, we check whether the node is a contraction node, and if it is, we proceed to the next check. Otherwise we return from the optimization.
2. Next, we check if the unit stride dimension type of the node is ``N``. If so, we swap the child nodes of the node to get a unit stride
   in the ``M`` dimension of the first input tensor (the left child node).
3. We call the ``reorder_left_node`` method on the node. The method then checks if the last dimensions of the left child node are
   ``KM``. If not, it permutes the dimensions to move ``KM`` to the rightmost location.
4. We call the ``reorder_right_node`` method on the node. The method then checks if the last dimensions of the right child node are
   ``NK``. If not, it permutes the dimensions to move ``NK`` to the rightmost location.
5. We call on both child nodes recursively the optimization pass.

Implementation of the heuristic:

.. code-block:: cpp

    void mini_jit::EinsumTree::optimize(EinsumNode *node)
    {
      if (node->type != NodeType::Contraction)
      {
          return;
      }

      conditional_swap(node);

      reorder_left_node(node);
      reorder_right_node(node);

      optimize(node->left);
      optimize(node->right);
    }


2. Performance
^^^^^^^^^^^^^^

**Task**: Benchmark the performance of your implementation on the provided examples. Report the measured performance in GFLOPS.

**First Example**

Einsum tree:

.. code-block:: vim

   0,1,2,3,4
   ├─ 7,3,4
   |  ├─ 7,3,8
   |  └─ 8,4
   └─ 0,1,2,7
      ├─ 0,5
      └─ 5,1,2,7
         ├─ 5,1,6
         └─ 6,2,7

String representation:

.. code-block::

   [[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]

Dimension sizes (by numerical ID):

.. code-block::

   100,72,128,128,3,71,305,32,3

**Second Example**

Einsum tree:

.. code-block:: vim

   0,1,2,3
   ├─ 1,4,7,8
   └─ 0,4,2,7,3,8
      ├─ 0,4,5,6
      └─ 2,5,7,3,6,8
         ├─ 2,5,7,9
         └─ 3,6,8,9

String representation:

.. code-block::

  [1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]

Dimension sizes (by numerical ID):

.. code-block::

   60,60,20,20,8,8,8,8,8,8

**Third Example**

.. code-block:: vim

   5,6,7,8,9
   ├─ 2,7,8,4
   |  ├─ 2,7,3
   |  └─ 3,8,4
   └─ 4,9,5,6,2
      ├─ 4,9,0
      └─ 0,5,6,2
         ├─ 0,5,1
         └─ 1,6,2

String representation:

.. code-block::

  [[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]

Dimension sizes (by numerical ID):

.. code-block::

   40,40,40,40,40,25,25,25,25,25

On the three example we get the following performance:

.. code-block:: bash
    :emphasize-lines: 4, 8, 12
  
    ---------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                        Time             CPU   Iterations      FLOPS
    ---------------------------------------------------------------------------------------------------------------------------------------------
    BM_einsum_tree_optimize_first_example/config:2/optimize:1/min_warmup_time:0.300_mean     280864567 ns    277445492 ns           10 142.788G/s
    BM_einsum_tree_optimize_first_example/config:2/optimize:1/min_warmup_time:0.300_median   279656272 ns    277621435 ns           10 142.675G/s
    BM_einsum_tree_optimize_first_example/config:2/optimize:1/min_warmup_time:0.300_stddev     5541524 ns      3668945 ns           10 1.86476G/s
    BM_einsum_tree_optimize_first_example/config:2/optimize:1/min_warmup_time:0.300_cv            1.97 %          1.32 %            10      1.31%
    BM_einsum_tree_optimize_second_example/config:3/optimize:1/min_warmup_time:0.300_mean     11268668 ns     11099948 ns           10 276.956G/s
    BM_einsum_tree_optimize_second_example/config:3/optimize:1/min_warmup_time:0.300_median   11249846 ns     11018021 ns           10 278.965G/s
    BM_einsum_tree_optimize_second_example/config:3/optimize:1/min_warmup_time:0.300_stddev     160890 ns       159649 ns           10 3.89922G/s
    BM_einsum_tree_optimize_second_example/config:3/optimize:1/min_warmup_time:0.300_cv           1.43 %          1.44 %            10      1.41%
    BM_einsum_tree_optimize_third_example/config:4/optimize:1/min_warmup_time:0.300_mean     121200659 ns    120226859 ns           10 277.896G/s
    BM_einsum_tree_optimize_third_example/config:4/optimize:1/min_warmup_time:0.300_median   121008763 ns    120129765 ns           10 278.117G/s
    BM_einsum_tree_optimize_third_example/config:4/optimize:1/min_warmup_time:0.300_stddev      853382 ns       535716 ns           10 1.23652G/s
    BM_einsum_tree_optimize_third_example/config:4/optimize:1/min_warmup_time:0.300_cv            0.70 %          0.45 %            10      0.44%

- **First Example:** 142.7 GFLOPS
- **Second Example:** 276.9 GFLOPS
- **Third Example:** 277.8 GFLOPS