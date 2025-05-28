#include "TensorOperation.h"
#include "release_assert.h"
#include <iostream>
#include <ranges>
#include <tuple>

/**
                                             .:=+######*=:.
                                         .=*##%%%%%%%%%%%%%%*-....
                                     ..=###%%%%%%%%%%%%%%%%%%%%%+:...           ..........    ..
                                   ..*######%%%%%%%%%%%%%%%%%%%%@@@@%%%%%%%%%%%%%%%%%%%%%%%%%#*=:..
                                 ..:-=*######%%%%%%%%%%%%%%%%%%%%%@@@%%%%%@@@@@%%####%%##****##**##*
                                 .:::--#######%%%%%%###%%%%%%@@@@%%##***++++++===+**+=--:....
                                .-::---*########%%%%%%%%%%%#*+++++++++======--:............
                               .+::::--*########%%%%%%%%%%#++++===--==-.........
                               --:::---*#########%%%%%%%%%%%%%%%*......
                              .*:::::--+##########%%%%%%%%%%%@%=..
                              .#=:::::-=###########%%%%%%%%%%%-.
                              :##::::::-*###########%%%%%%%%%#..
                              -###=:::::-*###########%%%%%%%%#..
                              =###%#+::::-+*##########%%%%%%%%=..
                             .+######%%+-::-=**########%%%%%%%%*..
                             .+#########%%%%+-:--*######%%%#*====....
                             .############%%@@@@@@@@@@*===========:..
                             -############%%%@@@@@@#+=====---=--==-..
                            .#%###########%@@@@@@#====-=------------.
                           .############%%%@@@@%+=------------------.
                          :############%%%%%%%%=--------------------:.
                         -############%%%%%%%%+=::::::::::::::::----:..
                        .#############%%%%%%@#=-:::::::::::::::::::::..
                       .#############%%%%%%%@--::::::::::::::::::::::..
                       -###########%%%%%%%%@#-:::::::::::::::::::::::..
                       =########%%%##%%%%%@@=:..:::::::::::::::::::::..
                      .*######%@####*%%%%%@%:.....................:::..
                      :######%@#####*%%%%%@*........................:..
                     .*#####%@#####*#%%%%@@-.....................:::::.
                     :#####%@%#####*%@%%@@#:.......::::::::::::::::::*...
                     =#####@%%%####*%@%@@%-..........................*+..
                    .*####@%%%%####*%@@@@#...........................:#*.
                   .=####%@%%%%####*%@@@@:............................=#=..
                   :####%@%%%%%####*%@@@=.............................:-#=.
                  .=###%@@%%%%%####+%@@*..............................::-#-...
                  .*###%@%%%%%%####+*@%:..............................::.:#:..
                  =###%@@#%%%%%%###**%-.............................:::...=#..
                 .*##%%@@#%%%%%%###*+=.............................::::....++...
                .-###%%@@#%%%%%%%###+=...........................:::::::....*-..
                .=###%%@@#%%%%%%%###*=..........................::::::::....=%..
                .+###%@@@%%%%%%%%###*+:.......................:::::::::::..::#+.
                .*##%%@@@@*%%%%%%####+:......................::::::::::::::::-#.
                .*%%%%%%@@%#%%%%%####*+......................:::::::::::::::::*=..
                .#%%%%%%@@@#%%%%%%####+....................:::::::::::::::::::#%..
                :#%%%%%%%@@%%%%%%%####*-.................:::::::::::::::::::::#%-.
                :#%%%%%%%%@@%%%%%%#####=.................:::::::::::::::::::::=%+.
                :#%%%%%%%%@@@%%%%%#####+:.............:::::::::::::::::-:::::::##.
                -#%%%%%%%%%@@%%%%%%####*:............::::::::::::::::::-:::::::-%:
                -#%%%%%%%%%@@@%%%%%#####-............:::::::::::::::::::::::::::%=
                -#%%%%%%%%%%@@%%%%%#####+..........::::::::::::::::::::.::::::::#*
                :#%%%%%%%%%%#@@%%%%#####*.........:::::::::::::::::::::..:::::::##.
                -%%%%%%%%%%%-@@%%%%%####*.......::::::::::::::::::::::...:::::::##.
                -%%%%%%%%%%*-%@%%%%%#####......:::::::::::::::::::::::...:::::::##.
                -%%%%%%%%%%+-*@@%%%%#####.....::::::::::::::::::::::::. ..-:::::%+.
                =%%%%%%%%%%--*@@%%%%####+....:::::::::::::::::::::::::. ..-::::*%-.
               .#%%%%%%%%%+--+@@%%%%%###-..:::::::::::::::::::::::::::. ..+:::*%%:.
               .%%%%%%%%%%---=@@%%%%%###:.:::::::::::::::::::::::::::.. ..*::%%%*..
               :%%%%%%%%%*--.=@%%%%%%##+.:::::::::::::::::::::::::--:.. ..%=+%%%:.
              .+%%%%%%%%%=-..+@%%%%%%##..::::::::::::::::::::::::---... ..%%%%%=..
              .*%%%%%%%%#-...#@%%%%%%#:::::::::::::::::::::::::----:.   ..#%%%+...
              :%%%%%%%%%:...:@%%%%%%%-::::::::::::::::::::::::-----..   ..=%%-..
             .*%%%%%%%%+....:@@%%%%%:::::::::::::::::::::::-------...     ....
             :%%%%%%%%%:....-%@@@@%:::::::::::::::::::::---------:...
            .+%%%%%%%%+.....:#@@@+::::::::::::::::::::----------:.
            :%%%%%%%%#.......+%*:::::::::::::::::::------------:..
           .%%%%%%%%@-.........:::::::::::::::----------------:.
          .*%%%%%%%@=........:::::::::::::------------------=...
         .+%%%%%%@%#:.......:::::::::::::....---------------....
        .+%%%%%@@@@-:......::::::::::::..    .=------------...
       .+%%@%@@@@@=::....:::::::::::::...    .-=--------==:..
      .#@@@@@@@@@*.::..:::::::::::::-:.       .=-=---=-=:=.
    .*@@@@@@@@=%=...:..::::::::::::-:..       .=+**=++**-..
 .:*@@%%%**+:.-.....:::::::::::-----..         .*%#######-:....
-%++-.......  ..  ...::--:::------::..         .=##*************+=:...
                     ...##**=+#*-...           .*#************#####*##*-.
                        -#######**+=-:...      .+##********#####*###%%#*+.
                        :##*********###****+=:....-+++**####*#####**+-.:+*.
                       .+#******####*****#%###*+:..        .+##**-..=#.
                        .*##**********#%####%%%#*=...        .. ...
                          ...:::-+*####*#%###=...-.
                                    .-==+=...-.

 */

bool mini_jit::TensorOperation::isUnary(prim_t prim)
{
  return prim == prim_t::copy || prim == prim_t::relu || prim == prim_t::zero;
}

bool mini_jit::TensorOperation::isBrgemm(prim_t prim)
{
  return prim == prim_t::brgemm || prim == prim_t::gemm;
}

int32_t mini_jit::TensorOperation::findMatch(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec, dim_t searchDim,
                                             exec_t searchExec, uint32_t startIndex)
{
  release_assert(dim.size() == exec.size(), "Expected the dimension types size to match the execution types size.");
  release_assert(startIndex <= dim.size(), "Expected the start index to be less than the dimension types size.");

  for (auto [iDim, iExec] = std::tuple{dim.begin() + startIndex, exec.begin() + startIndex}; iDim != dim.end(); ++iDim, ++iExec)
  {
    // std::cerr << "iDim:" << (uint32_t)*iDim << " " << std::distance(dim.begin(), iDim) << ", iExec:" << (uint32_t)*iExec << " "
    //           << std::distance(exec.begin(), iExec) << std::endl;
    if (*iDim == searchDim && *iExec == searchExec)
    {
      return std::distance(dim.begin(), iDim);
    }
  }

  return -1;
}

bool mini_jit::TensorOperation::isValidPrimConfig(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec,
                                                  const std::span<const int64_t> &strides_in0, const std::span<const int64_t> &strides_out)
{
  int32_t indexM = findMatch(dim, exec, dim_t::m, exec_t::prim);
  int32_t indexN = findMatch(dim, exec, dim_t::n, exec_t::prim);
  if (indexM == -1 || indexN == -1)
  {
    std::cerr << "1: Could not find a matching index: indexM:" << indexM << ", indexN:" << indexN << std::endl;
    return false;
  }

  if (!(isExpectedStride(1, indexM, strides_in0) && isExpectedStride(1, indexM, strides_out)))
  {
    return false;
  }

  // Search for new that fits the configuration, both should return -1
  indexM = findMatch(dim, exec, dim_t::m, exec_t::prim, indexM + 1);
  indexN = findMatch(dim, exec, dim_t::n, exec_t::prim, indexN + 1);
  if (indexM != -1 || indexN != -1)
  {
    std::cerr << "2: Could not find a matching index: indexM:" << indexM << ", indexN" << indexN << std::endl;
    return false;
  }

  return true;
}

bool mini_jit::TensorOperation::isValidKDim(const std::span<const dim_t> &dim, const std::span<const exec_t> &exec,
                                            const std::span<const int64_t> &strides_in1, prim_t prim)
{
  if (isBrgemm(prim))
  {
    int32_t indexK = findMatch(dim, exec, dim_t::k, exec_t::prim);

    if (indexK == -1)
    {
      return false;
    }

    if (prim == prim_t::brgemm)
    {
      // Another k dim should exists
      indexK = findMatch(dim, exec, dim_t::k, exec_t::prim, indexK + 1);

      if (indexK == -1)
      {
        return false;
      }
    }

    if (!isExpectedStride(1, indexK, strides_in1))
    {
      return false;
    }

    // No other k dim should exists
    indexK = findMatch(dim, exec, dim_t::k, exec_t::prim, indexK + 1);
    return indexK == -1;
  }
  else if (isUnary(prim))
  {
    // Expected to find not K dim
    int32_t indexK = findMatch(dim, exec, dim_t::k, exec_t::prim);

    return indexK == -1;
  }
  else
  {
    return true;
  }
}

bool mini_jit::TensorOperation::isSortedConfiguration(const std::span<const exec_t> &exec)
{
  bool foundPrimitive = false;
  for (auto type = exec.begin(); type != exec.end(); ++type)
  {
    if (!foundPrimitive && *type == exec_t::prim)
    {
      foundPrimitive = true;
      indexPrimitiveLoop = std::distance(type, exec.begin());
    }

    if (foundPrimitive && *type != exec_t::prim)
    {
      return false;
    }
  }

  return true;
}

bool mini_jit::TensorOperation::isExpectedStride(int64_t expected, int index, const std::span<const int64_t> &strides)
{
  if (index == -1)
  {
    return false;
  }

  return strides[index] == expected;
}

bool mini_jit::TensorOperation::isValidStride(const std::span<const dim_t> &dim, const std::span<const int64_t> &strides,
                                              const stride_t strideType)
{
  release_assert(dim.size() == strides.size(), "Expected the dim and the strides to have same size.");

  for (auto [iDim, iStride] = std::tuple{dim.begin(), strides.begin()}; iDim != dim.end(); ++iDim, ++iStride)
  {
    switch (strideType)
    {
    case stride_t::in0:
      if (*iDim == dim_t::n && *iStride != 0)
      {
        return false;
      }
      break;

    case stride_t::in1:
      if (*iDim == dim_t::m && *iStride != 0)
      {
        return false;
      }
      break;

    case stride_t::out:
      if (*iDim == dim_t::k && *iStride != 0)
      {
        return false;
      }
      break;

    default:
      release_assert(false, "Unexpected stride type to handel.");
      break;
    }
  }

  return true;
}

mini_jit::Unary::error_t mini_jit::TensorOperation::generateUnary(Unary &unary, prim_t prim, const std::span<const int64_t> &dim_sizes)
{
  release_assert(indexPrimM != -1, "Expected a match for the m primitive dimension");
  release_assert(indexPrimN != -1, "Expected a match for the n primitive dimension");

  Unary::ptype_t type;
  switch (prim)
  {
  case prim_t::zero:
    type = Unary::ptype_t::zero;
    break;

  case prim_t::copy:
    type = Unary::ptype_t::identity;
    break;

  case prim_t::relu:
    type = Unary::ptype_t::relu;
    break;

  default:
    release_assert(false, "Found a invalid type for the unary first touch.");
    break;
  }
  return unary.generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], 0, Unary::dtype_t::fp32, type);
}

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
  indexPrimitiveLoop = -1;

  // Validate dimensions
  if (dim_sizes.size() != dim_types.size() || dim_sizes.empty() || dim_types.empty())
  {
    hasSetupError = true;
    std::cerr << "Error: Dimension sizes and types must match and cannot be empty." << std::endl;
    return error_t::err_wrong_dimension;
  }

  if (!(strides_in0.size() == dim_sizes.size() && strides_out.size() == dim_sizes.size() &&
        (strides_in1.size() == dim_sizes.size()
         // strides_in1 can be empty for unary operations
         || ((isUnary(prim_first_touch) || prim_first_touch == prim_t::none) && (isUnary(prim_main) || prim_main == prim_t::none) &&
             (isUnary(prim_last_touch) || prim_last_touch == prim_t::none) && strides_in1.empty()))))
  {
    hasSetupError = true;
    std::cerr << "Error: Strides must match the number of dimensions." << std::endl;
    return error_t::err_wrong_dimension;  // Strides must match the number of dimensions
  }

  for (exec_t exec : exec_types)
  {
    if (exec == exec_t::shared)
    {
      hasSetupError = true;
      return error_t::err_execution_type_not_supported;
    }
  }

  // Validate dtype types - currently only fp32 is supported
  if (dtype != dtype_t::fp32)
  {
    hasSetupError = true;
    return error_t::err_wrong_dtype;
  }

  if (!isSortedConfiguration(exec_types))
  {
    hasSetupError = true;
    return error_t::err_invalid_execution_order;
  }

  if (!isValidPrimConfig(dim_types, exec_types, strides_in0, strides_out))
  {
    hasSetupError = true;
    std::cerr << "1: Invalid primitive configuration detected" << std::endl;
    return error_t::err_invalid_primitive_configuration;
  }

  if (!isValidKDim(dim_types, exec_types, strides_in1, prim_main))
  {
    hasSetupError = true;
    std::cerr << "2: Invalid primitive configuration detected" << std::endl;
    return error_t::err_invalid_primitive_configuration;
  }

  if (isUnary(prim_main))
  {
    if (!isValidStride(dim_types, strides_in0, stride_t::out) || !isValidStride(dim_types, strides_out, stride_t::out))
    {
      hasSetupError = true;
      std::cerr << "3: Invalid stride configuration detected for unary" << std::endl;
      return error_t::err_invalid_strides;
    }
  }
  else if (isBrgemm(prim_main))
  {
    if (!isValidStride(dim_types, strides_in0, stride_t::in0) || !isValidStride(dim_types, strides_in1, stride_t::in1) ||
        !isValidStride(dim_types, strides_out, stride_t::out))
    {
      hasSetupError = true;
      std::cerr << "3: Invalid stride configuration detected for brgemm" << std::endl;
      return error_t::err_invalid_strides;
    }
  }
  else if (prim_main == prim_t::none)
  {
    // Do nothing
  }
  else
  {
    release_assert(false, "Unexpected value for the main primitive");
  }

  // Validated through isValidPrimConfig that these indices exists
  indexPrimM = findMatch(dim_types, exec_types, dim_t::m, exec_t::prim);
  indexPrimN = findMatch(dim_types, exec_types, dim_t::n, exec_t::prim);

  release_assert(indexPrimM != -1, "Expected a valid index for the M dimension but found none.");
  release_assert(indexPrimN != -1, "Expected a valid index for the N dimension but found none.");
  release_assert(indexPrimitiveLoop != -1, "Expected a valid start of the primitive loop but found none.");

  if (prim_first_touch != prim_t::none)
  {
    if (isUnary(prim_first_touch))
    {
      first_touch.emplace<Unary>();
      TensorOperation::prim_first = prim_first_touch;

      Unary::error_t error = generateUnary(std::get<Unary>(first_touch), prim_first_touch, dim_sizes);

      if (error != Unary::error_t::success)
      {
        hasSetupError = true;
        return error_t::err_invalid_first_touch_configuration;
      }
    }
    else
    {
      hasSetupError = true;
      return error_t::err_wrong_first_touch_primitive;
    }
  }

  if (prim_main != prim_t::none)
  {
    if (isBrgemm(prim_main))
    {
      main_kernel.emplace<Brgemm>();
      TensorOperation::prim_main = prim_main;

      if (prim_main == prim_t::brgemm)
      {
        indexPrimBatch = findMatch(dim_types, exec_types, dim_t::k, exec_t::prim);
        indexPrimK = findMatch(dim_types, exec_types, dim_t::k, exec_t::prim, indexPrimBatch + 1);

        release_assert(indexPrimBatch != -1, "Expected a valid index for the Batch dimension but found none.");
        release_assert(indexPrimK != -1, "Expected a valid index for the Batch dimension but found none.");

        std::get<Brgemm>(main_kernel)
          .generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], dim_sizes[indexPrimK], dim_sizes[indexPrimBatch], 0, 0, 0,
                    Brgemm::dtype_t::fp32);
      }
      else if (prim_main == prim_t::gemm)
      {
        indexPrimK = findMatch(dim_types, exec_types, dim_t::k, exec_t::prim);

        release_assert(indexPrimK != -1, "Expected a valid index for the K dimension but found none.");

        std::get<Brgemm>(main_kernel)
          .generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], dim_sizes[indexPrimK], 1, 0, 0, 0, Brgemm::dtype_t::fp32);
      }
      else
      {
        release_assert(false, "Found missing brgemm configuration.");
      }
    }
    else if (isUnary(prim_main))
    {
      main_kernel.emplace<Unary>();
      TensorOperation::prim_main = prim_main;
      indexPrimK = indexPrimN;

      Unary::error_t error = generateUnary(std::get<Unary>(main_kernel), prim_main, dim_sizes);

      if (error != Unary::error_t::success)
      {
        hasSetupError = true;
        return error_t::err_invalid_main_configuration;
      }
    }
    else
    {
      hasSetupError = true;
      return error_t::err_wrong_main_primitive;
    }
  }

  if (prim_last_touch != prim_t::none)
  {
    if (isUnary(prim_last_touch))
    {
      last_touch.emplace<Unary>();
      TensorOperation::prim_last = prim_last_touch;

      Unary::error_t error = generateUnary(std::get<Unary>(last_touch), prim_last_touch, dim_sizes);

      if (error != Unary::error_t::success)
      {
        hasSetupError = true;
        return error_t::err_invalid_main_configuration;
      }
    }
    else
    {
      hasSetupError = true;
      return error_t::err_wrong_last_touch_primitive;
    }
  }

  TensorOperation::dtype = dtype;
  TensorOperation::dim_types = dim_types;
  TensorOperation::exec_types = exec_types;
  TensorOperation::dim_sizes = dim_sizes;
  TensorOperation::strides_in0 = strides_in0;
  TensorOperation::strides_in1 = strides_in1;
  TensorOperation::strides_out = strides_out;

  return error_t::success;
}

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

void mini_jit::TensorOperation::execute_dimension(int64_t index_dim, char const *ptr_in0, char const *ptr_in1, char *ptr_out,
                                                  bool first_access, bool last_access)
{
  release_assert(exec_types[index_dim] != exec_t::seq, "Expected a sequential loop");

  uint32_t dtype_bytes = 4;
  int64_t dim_size = dim_sizes[index_dim];
  int64_t stride_in0 = strides_in0[index_dim];
  int64_t stride_in1 = isUnary(prim_main) ? 1 : strides_in1[index_dim];
  int64_t stride_out = strides_out[index_dim];

  std::cout << "Execute check " << index_dim + 1 << " " << indexPrimitiveLoop << std::endl;
  if (index_dim + 1 < indexPrimitiveLoop)
  {
    bool is_first = first_access;
    bool is_last = last_access;

    for (int64_t iDim = 0; iDim < dim_size; iDim++)
    {
      if (dim_types[iDim] == dim_t::k)
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
  else
  {
    // call first touch kernel if necessary
    if (first_access && prim_first != prim_t::none)
    {
      if (std::holds_alternative<Unary>(first_touch))
      {
        std::cout << "First touch: indexPrimN" << indexPrimN << " " << strides_out[indexPrimN]
                  << std::endl;
        Unary::kernel_t kernel = std::get<Unary>(first_touch).get_kernel();
        kernel(ptr_out, ptr_out, strides_out[indexPrimN], strides_out[indexPrimN]);
      }
      else
      {
        release_assert(false, "Unexpected first touch primitive");
      }
    }

    // call main_kernel kernel
    if (prim_main != prim_t::none)
    {
      if (std::holds_alternative<Unary>(main_kernel))
      {
        std::cout << "Unary: indexPrimN " << indexPrimN << " " << strides_in0[indexPrimN] << " " << strides_out[indexPrimN] << std::endl;
        Unary::kernel_t kernel = std::get<Unary>(main_kernel).get_kernel();
        kernel(ptr_in0, ptr_out, strides_in0[indexPrimN], strides_out[indexPrimN]);
      }
      else if (std::holds_alternative<Brgemm>(main_kernel))
      {
        std::cout << "Gemm: indexPrimN " << indexPrimN << " " << "indexPrimK " << indexPrimK << " " << "indexPrimBatch " << indexPrimBatch
                  << " " << strides_in0[indexPrimK] << " " << strides_in1[indexPrimN] << " " << strides_out[indexPrimN] << std::endl;
        Brgemm::kernel_t kernel = std::get<Brgemm>(main_kernel).get_kernel();

        if (prim_main == prim_t::gemm)
        {
          kernel(ptr_in0, ptr_in1, ptr_out, strides_in0[indexPrimK], strides_in1[indexPrimN], strides_out[indexPrimN], 1, 1);
        }
        else if (prim_main == prim_t::brgemm)
        {
          kernel(ptr_in0, ptr_in1, ptr_out, strides_in0[indexPrimK], strides_in1[indexPrimN], strides_out[indexPrimN],
                 strides_in0[indexPrimBatch], strides_in1[indexPrimBatch]);
        }
        else
        {
          release_assert(false, "Unexpected Brgemm primitive.");
        }
      }
      else
      {
        release_assert(false, "Unexpected main primitive.");
      }
    }

    // call last touch kernel if necessary
    if (last_access && prim_last != prim_t::none)
    {
      if (std::holds_alternative<Unary>(last_touch))
      {
        std::cout << "Last touch: indexPrimK" << indexPrimK << " " << strides_in0[indexPrimK] << " " << strides_out[indexPrimN]
                  << std::endl;
        Unary::kernel_t kernel = std::get<Unary>(last_touch).get_kernel();
        kernel(ptr_out, ptr_out, strides_out[indexPrimN], strides_out[indexPrimN]);
      }
      else
      {
        release_assert(false, "Unexpected last touch primitive");
      }
    }
  }
}