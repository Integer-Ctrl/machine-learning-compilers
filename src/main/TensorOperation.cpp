#include "TensorOperation.h"
#include "TensorOptimization.h"
#include "release_assert.h"
#include <format>
#include <iostream>
#include <omp.h>
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

bool mini_jit::TensorOperation::isUnary(TensorConfig::prim_t prim)
{
  return prim == TensorConfig::prim_t::copy || prim == TensorConfig::prim_t::relu || prim == TensorConfig::prim_t::zero;
}

bool mini_jit::TensorOperation::isBrgemm(TensorConfig::prim_t prim)
{
  return prim == TensorConfig::prim_t::brgemm || prim == TensorConfig::prim_t::gemm;
}

int32_t mini_jit::TensorOperation::findMatch(const std::span<const TensorConfig::dim_t> &dim,
                                             const std::span<const TensorConfig::exec_t> &exec, TensorConfig::dim_t searchDim,
                                             TensorConfig::exec_t searchExec, uint32_t startIndex)
{
  release_assert(dim.size() == exec.size(), "Expected the dimension types size to match the execution types size.");
  release_assert(startIndex <= dim.size(), "Expected the start index to be less than the dimension types size.");

  if (startIndex >= dim.size())
  {
    return -1;
  }

  for (auto [iDim, iExec] = std::tuple{dim.begin() + startIndex, exec.begin() + startIndex}; iDim != dim.end(); ++iDim, ++iExec)
  {
    if (*iDim == searchDim && *iExec == searchExec)
    {
      return std::distance(dim.begin(), iDim);
    }
  }

  return -1;
}

bool mini_jit::TensorOperation::isValidPrimConfig(const std::span<const TensorConfig::dim_t> &dim,
                                                  const std::span<const TensorConfig::exec_t> &exec)
{
  int32_t indexM = findMatch(dim, exec, TensorConfig::dim_t::m, TensorConfig::exec_t::prim);
  int32_t indexN = findMatch(dim, exec, TensorConfig::dim_t::n, TensorConfig::exec_t::prim);
  if (indexM == -1 || indexN == -1)
  {
    std::cerr << "isValidPrimConfig 1: Could not find a matching index: indexM:" << indexM << ", indexN:" << indexN << std::endl;
    return false;
  }

  // Search for new that fits the configuration, both should return -1
  indexM = findMatch(dim, exec, TensorConfig::dim_t::m, TensorConfig::exec_t::prim, indexM + 1);
  indexN = findMatch(dim, exec, TensorConfig::dim_t::n, TensorConfig::exec_t::prim, indexN + 1);
  if (indexM != -1 || indexN != -1)
  {
    std::cerr << "isValidPrimConfig 2: Could not find a matching index: indexM:" << indexM << ", indexN" << indexN << std::endl;
    return false;
  }

  return true;
}

bool mini_jit::TensorOperation::isValidPrimStrides(const std::span<const TensorConfig::dim_t> &dim,
                                                   const std::span<const TensorConfig::exec_t> &exec,
                                                   const std::span<const int64_t> &strides_in0, const std::span<const int64_t> &strides_out,
                                                   const TensorConfig::prim_t main_prim)
{
  int32_t indexM = findMatch(dim, exec, TensorConfig::dim_t::m, TensorConfig::exec_t::prim);
  int32_t indexN = findMatch(dim, exec, TensorConfig::dim_t::n, TensorConfig::exec_t::prim);
  if (indexM == -1 || indexN == -1)
  {
    std::cerr << "isValidStride: Could not find a matching index: indexM:" << indexM << ", indexN:" << indexN << std::endl;
    return false;
  }

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

  std::cerr << "isValidStride: Could not find a valid stride: in0: m-stride: " << strides_in0[indexM]
            << ", n-stride: " << strides_in0[indexN] << "; out: m-stride: " << strides_out[indexM] << ", n-stride: " << strides_out[indexN]
            << std::endl;
  return false;
}

bool mini_jit::TensorOperation::isValidKDim(const std::span<const TensorConfig::dim_t> &dim,
                                            const std::span<const TensorConfig::exec_t> &exec, const std::span<const int64_t> &strides_in1,
                                            const TensorConfig::prim_t prim)
{
  if (isBrgemm(prim))
  {
    int32_t indexK = findMatch(dim, exec, TensorConfig::dim_t::k, TensorConfig::exec_t::prim);

    if (indexK == -1)
    {
      return false;
    }

    if (prim == TensorConfig::prim_t::brgemm)
    {
      // Another k dim should exists
      indexK = findMatch(dim, exec, TensorConfig::dim_t::k, TensorConfig::exec_t::prim, indexK + 1);

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
    indexK = findMatch(dim, exec, TensorConfig::dim_t::k, TensorConfig::exec_t::prim, indexK + 1);
    return indexK == -1;
  }
  else if (isUnary(prim))
  {
    // Expected to find not K dim
    int32_t indexK = findMatch(dim, exec, TensorConfig::dim_t::k, TensorConfig::exec_t::prim);

    return indexK == -1;
  }
  else
  {
    return true;
  }
}

bool mini_jit::TensorOperation::isSortedConfiguration(const std::span<const TensorConfig::exec_t> &exec)
{
  bool seenSequential = false;
  bool seenPrimitive = false;
  for (TensorConfig::exec_t exec_type : exec)
  {
    if (exec_type == TensorConfig::exec_t::shared && !seenSequential && !seenPrimitive)
    {
      // Nothing to do, shared must be first
    }
    else if (exec_type == TensorConfig::exec_t::shared && (seenSequential || seenPrimitive))
    {
      return false;
    }
    else if (exec_type == TensorConfig::exec_t::seq && !seenPrimitive)
    {
      seenSequential = true;
    }
    else if (exec_type == TensorConfig::exec_t::seq && seenPrimitive)
    {
      return false;
    }
    else if (exec_type == TensorConfig::exec_t::prim)
    {
      seenPrimitive = true;
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

bool mini_jit::TensorOperation::isValidStride(const std::span<const TensorConfig::dim_t> &dim, const std::span<const int64_t> &strides,
                                              const stride_t strideType)
{
  release_assert(dim.size() == strides.size(), "Expected the dim and the strides to have same size.");

  for (auto [iDim, iStride] = std::tuple{dim.begin(), strides.begin()}; iDim != dim.end(); ++iDim, ++iStride)
  {
    switch (strideType)
    {
    case stride_t::in0:
      switch (*iDim)
      {
      case TensorConfig::dim_t::c:
      case TensorConfig::dim_t::m:
      case TensorConfig::dim_t::k:
        if (*iStride == 0)
        {
          return false;
        }
        break;

      case TensorConfig::dim_t::n:
        if (*iStride != 0)
        {
          return false;
        }
        break;

      default:
        release_assert(false, "Found unhandled dimension type.");
        break;
      }
      break;

    case stride_t::in1:
      switch (*iDim)
      {
      case TensorConfig::dim_t::c:
      case TensorConfig::dim_t::n:
      case TensorConfig::dim_t::k:
        if (*iStride == 0)
        {
          return false;
        }
        break;

      case TensorConfig::dim_t::m:
        if (*iStride != 0)
        {
          return false;
        }
        break;

      default:
        release_assert(false, "Found unhandled dimension type.");
        break;
      }
      break;

    case stride_t::out:
      switch (*iDim)
      {
      case TensorConfig::dim_t::c:
      case TensorConfig::dim_t::n:
      case TensorConfig::dim_t::m:
        if (*iStride == 0)
        {
          return false;
        }
        break;

      case TensorConfig::dim_t::k:
        if (*iStride != 0)
        {
          return false;
        }
        break;

      default:
        release_assert(false, "Found unhandled dimension type.");
        break;
      }
      break;

    default:
      release_assert(false, "Unexpected stride type to handel.");
      break;
    }
  }

  return true;
}

mini_jit::Unary::error_t mini_jit::TensorOperation::generateUnary(Unary &unary, TensorConfig::prim_t prim,
                                                                  const std::span<const int64_t> &dim_sizes, bool isTranspose)
{
  release_assert(indexPrimM != -1, "Expected a match for the m primitive dimension");
  release_assert(indexPrimN != -1, "Expected a match for the n primitive dimension");

  Unary::ptype_t type;
  switch (prim)
  {
  case TensorConfig::prim_t::zero:
    type = Unary::ptype_t::zero;
    break;

  case TensorConfig::prim_t::copy:
    type = Unary::ptype_t::identity;
    break;

  case TensorConfig::prim_t::relu:
    type = Unary::ptype_t::relu;
    break;

  default:
    release_assert(false, "Found a invalid type for the unary first touch.");
    break;
  }

  return unary.generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], isTranspose, Unary::dtype_t::fp32, type);
}

mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup(const TensorConfig &config)
{
  mini_jit::TensorOptimization optimization;
  TensorOperation::config = optimization.optimize(config);

  return setup_no_optimization(TensorOperation::config.dtype, TensorOperation::config.first_touch, TensorOperation::config.main,
                               TensorOperation::config.last_touch, TensorOperation::config.dim_types, TensorOperation::config.exec_types,
                               TensorOperation::config.dim_sizes, TensorOperation::config.strides_in0, TensorOperation::config.strides_in1,
                               TensorOperation::config.strides_out);
}

mini_jit::TensorOperation::error_t mini_jit::TensorOperation::setup_no_optimization(
  TensorConfig::dtype_t dtype, TensorConfig::prim_t prim_first_touch, TensorConfig::prim_t prim_main, TensorConfig::prim_t prim_last_touch,
  std::span<const TensorConfig::dim_t> dim_types, std::span<const TensorConfig::exec_t> exec_types, std::span<const int64_t> dim_sizes,
  std::span<const int64_t> strides_in0, std::span<const int64_t> strides_in1, std::span<const int64_t> strides_out)
{
  // Reset to defaults
  hasSetupError = false;
  isParallel = false;
  isTranspose = false;
  indexPrimBatch = -1;
  indexPrimK = -1;
  indexPrimM = -1;
  indexPrimN = -1;

  // Validate dimensions
  if (dim_sizes.size() != dim_types.size() || dim_sizes.empty() || dim_types.empty())
  {
    hasSetupError = true;
    std::cerr << "Error: Dimension sizes and types must match and cannot be empty, but got dim_sizes: " << dim_sizes.size() << ", dim_types"
              << dim_types.size() << std::endl;
    return error_t::err_wrong_dimension;
  }

  if (!(strides_in0.size() == dim_sizes.size() && strides_out.size() == dim_sizes.size() &&
        (strides_in1.size() == dim_sizes.size()
         // strides_in1 can be empty for unary operations
         || ((isUnary(prim_first_touch) || prim_first_touch == TensorConfig::prim_t::none) &&
             (isUnary(prim_main) || prim_main == TensorConfig::prim_t::none) &&
             (isUnary(prim_last_touch) || prim_last_touch == TensorConfig::prim_t::none) && strides_in1.empty()))))
  {
    hasSetupError = true;
    std::cerr << "Error: Strides must match the number of dimensions, but got dim_sizes: " << dim_sizes.size()
              << ", strides_in0: " << strides_in0.size() << ", strides_in1: " << strides_in1.size()
              << ", strides_out:" << strides_out.size() << std::endl;
    return error_t::err_wrong_dimension;  // Strides must match the number of dimensions
  }

  // Check if shared exists and set parallel flag
  for (TensorConfig::exec_t exec : exec_types)
  {
    if (exec == TensorConfig::exec_t::shared)
    {
      isParallel = true;
    }
  }

  if (isParallel)
  {
    // K dimension must not be shared
    int32_t kDimExecType = findMatch(dim_types, exec_types, TensorConfig::dim_t::k, TensorConfig::exec_t::shared);
    if (kDimExecType != -1)
    {
      hasSetupError = true;
      std::cerr << "Error: Found k dimension tagged as shared, but can not execute k dimension as shared." << std::endl;
      return error_t::err_k_dimension_must_not_be_shared;
    }
  }

  // Validate dtype types - currently only fp32 is supported
  if (dtype != TensorConfig::dtype_t::fp32)
  {
    hasSetupError = true;
    std::cerr << "Error: data type must be fp32, but got " << static_cast<uint32_t>(dtype) << std::endl;
    return error_t::err_wrong_dtype;
  }

  // Validate execution type order: shared -> seq -> prim
  if (!isSortedConfiguration(exec_types))
  {
    hasSetupError = true;
    std::cerr << "Error: Expected the execution types to be sorted in the order: (shared*, sequential*, primitive*)" << std::endl;
    return error_t::err_invalid_execution_order;
  }

  if (!isValidPrimConfig(dim_types, exec_types))
  {
    hasSetupError = true;
    std::cerr << "Error: Invalid primitive configuration detected. Expected one primitive for m and one primitive for n to exist"
              << std::endl;
    return error_t::err_invalid_primitive_configuration;
  }

  if (!isValidPrimStrides(dim_types, exec_types, strides_in0, strides_out, prim_main))
  {
    hasSetupError = true;
    std::cerr << "Error: Invalid strides for the primitive m dimension (or n dimension if transpose)." << std::endl;
    return error_t::err_invalid_strides;
  }

  if (!isValidKDim(dim_types, exec_types, strides_in1, prim_main))
  {
    hasSetupError = true;
    std::cerr << (int)prim_main << std::endl;
    std::cerr << "Error: Invalid primitive configuration detected. Expected to find zero primitive k dimension for unary, one primitive k "
                 "dimension for gemm, two primitive k dimension for brgemm."
              << std::endl;
    return error_t::err_invalid_primitive_configuration;
  }

  if (isUnary(prim_main))
  {
    if (!isValidStride(dim_types, strides_in0, stride_t::out) || !isValidStride(dim_types, strides_out, stride_t::out))
    {
      hasSetupError = true;
      std::cerr << "Error: Invalid stride configuration detected for unary. Expected k-dimension to have a stride of zero." << std::endl;
      return error_t::err_invalid_strides;
    }

    if (prim_last_touch != TensorConfig::prim_t::none || prim_last_touch != TensorConfig::prim_t::none)
    {
      hasSetupError = true;
      std::cerr << "Error: A main 'Unary' primitive can not have first touch and last touch primitives." << std::endl;
      return error_t::err_invalid_main_configuration;
    }
  }
  else if (isBrgemm(prim_main))
  {
    if (!isValidStride(dim_types, strides_in0, stride_t::in0) || !isValidStride(dim_types, strides_in1, stride_t::in1) ||
        !isValidStride(dim_types, strides_out, stride_t::out))
    {
      hasSetupError = true;
      std::cerr << "Error: Invalid stride configuration detected for brgemm. Expected for in0 to have n-dimension stride of zero, in1 to "
                   "have m-dimension stride of zero and out to have k-dimension stride of zero."
                << std::endl;
      return error_t::err_invalid_strides;
    }
  }
  else if (prim_main == TensorConfig::prim_t::none)
  {
    // Do nothing
  }
  else
  {
    release_assert(false, "Unexpected value for the main primitive");
  }

  // Validated through isValidPrimConfig that these indices exists
  indexPrimM = findMatch(dim_types, exec_types, TensorConfig::dim_t::m, TensorConfig::exec_t::prim);
  indexPrimN = findMatch(dim_types, exec_types, TensorConfig::dim_t::n, TensorConfig::exec_t::prim);

  release_assert(indexPrimM != -1, "Expected a valid index for the M dimension but found none.");
  release_assert(indexPrimN != -1, "Expected a valid index for the N dimension but found none.");

  if (prim_first_touch != TensorConfig::prim_t::none)
  {
    if (isUnary(prim_first_touch))
    {
      first_touch.emplace<Unary>();
      TensorOperation::prim_first = prim_first_touch;

      Unary::error_t error = generateUnary(std::get<Unary>(first_touch), prim_first_touch, dim_sizes, false);

      if (error != Unary::error_t::success)
      {
        hasSetupError = true;
        std::cerr << "Error: while generating the first touch unary: " << static_cast<uint32_t>(error) << std::endl;
        return error_t::err_invalid_first_touch_configuration;
      }
    }
    else
    {
      hasSetupError = true;
      std::cerr << "Error: Invalid type for the first touch primitive, only support zero, copy, relu." << std::endl;
      return error_t::err_wrong_first_touch_primitive;
    }
  }

  if (prim_main != TensorConfig::prim_t::none)
  {
    if (isBrgemm(prim_main))
    {
      main_kernel.emplace<Brgemm>();
      TensorOperation::prim_main = prim_main;

      if (prim_main == TensorConfig::prim_t::brgemm)
      {
        indexPrimBatch = findMatch(dim_types, exec_types, TensorConfig::dim_t::k, TensorConfig::exec_t::prim);
        indexPrimK = findMatch(dim_types, exec_types, TensorConfig::dim_t::k, TensorConfig::exec_t::prim, indexPrimBatch + 1);

        release_assert(indexPrimBatch != -1, "Expected a valid index for the Batch dimension but found none.");
        release_assert(indexPrimK != -1, "Expected a valid index for the Batch dimension but found none.");

        Brgemm::error_t error = std::get<Brgemm>(main_kernel)
                                  .generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], dim_sizes[indexPrimK], dim_sizes[indexPrimBatch],
                                            0, 0, 0, Brgemm::dtype_t::fp32);
        if (error != Brgemm::error_t::success)
        {
          hasSetupError = true;
          std::cerr << "Error: while generating the main brgemm: " << static_cast<uint32_t>(error) << std::endl;
          return error_t::err_invalid_main_configuration;
        }
      }
      else if (prim_main == TensorConfig::prim_t::gemm)
      {
        indexPrimK = findMatch(dim_types, exec_types, TensorConfig::dim_t::k, TensorConfig::exec_t::prim);

        release_assert(indexPrimK != -1, "Expected a valid index for the K dimension but found none.");

        Brgemm::error_t error =
          std::get<Brgemm>(main_kernel)
            .generate(dim_sizes[indexPrimM], dim_sizes[indexPrimN], dim_sizes[indexPrimK], 1, 0, 0, 0, Brgemm::dtype_t::fp32);

        if (error != Brgemm::error_t::success)
        {
          hasSetupError = true;
          std::cerr << "Error: while generating the main gemm: " << static_cast<uint32_t>(error) << std::endl;
          return error_t::err_invalid_main_configuration;
        }
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

      Unary::error_t error = generateUnary(std::get<Unary>(main_kernel), prim_main, dim_sizes, isTranspose);

      if (error != Unary::error_t::success)
      {
        hasSetupError = true;
        std::cerr << "Error: while generating the main unary: " << static_cast<uint32_t>(error) << std::endl;
        return error_t::err_invalid_main_configuration;
      }
    }
    else
    {
      hasSetupError = true;
      std::cerr << "Error: Invalid type for the main primitive, only support zero, copy, relu, gemm, brgemm." << std::endl;
      return error_t::err_wrong_main_primitive;
    }
  }

  if (prim_last_touch != TensorConfig::prim_t::none)
  {
    if (isUnary(prim_last_touch))
    {
      last_touch.emplace<Unary>();
      TensorOperation::prim_last = prim_last_touch;

      Unary::error_t error = generateUnary(std::get<Unary>(last_touch), prim_last_touch, dim_sizes, false);

      if (error != Unary::error_t::success)
      {
        hasSetupError = true;
        std::cerr << "Error: while generating the last touch unary: " << static_cast<uint32_t>(error) << std::endl;
        return error_t::err_invalid_last_touch_configuration;
      }
    }
    else
    {
      hasSetupError = true;
      std::cerr << "Error: Invalid type for the last touch primitive, only support zero, copy, relu." << std::endl;
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
  uint32_t dtype_bytes = 4;
  int64_t dim_size = dim_sizes[index_dim];
  int64_t stride_in0 = strides_in0[index_dim];
  int64_t stride_in1 = isUnary(prim_main) ? 1 : strides_in1[index_dim];
  int64_t stride_out = strides_out[index_dim];

  if (exec_types[index_dim] == TensorConfig::exec_t::shared)
  {
    // Parallel execution with OpenMP
    bool is_first = first_access;
    bool is_last = last_access;

#ifdef USE_OPENMP
#pragma omp parallel for if (dim_size > 1)
#endif
    for (int64_t iDim = 0; iDim < dim_size; iDim++)
    {
      if (dim_types[index_dim] == TensorConfig::dim_t::k)
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
  else if (exec_types[index_dim] == TensorConfig::exec_t::seq)
  {

    bool is_first = first_access;
    bool is_last = last_access;

    for (int64_t iDim = 0; iDim < dim_size; iDim++)
    {
      if (dim_types[index_dim] == TensorConfig::dim_t::k)
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
    release_assert(exec_types[index_dim] == TensorConfig::exec_t::prim, "Expected a primitive loop");

    // call first touch kernel if necessary
    if (first_access && prim_first != TensorConfig::prim_t::none)
    {
      if (std::holds_alternative<Unary>(first_touch))
      {
        Unary::kernel_t kernel = std::get<Unary>(first_touch).get_kernel();
        kernel(ptr_out, ptr_out, strides_out[indexPrimN], strides_out[indexPrimN]);
      }
      else
      {
        release_assert(false, "Unexpected first touch primitive");
      }
    }

    // call main_kernel kernel
    if (prim_main != TensorConfig::prim_t::none)
    {
      if (std::holds_alternative<Unary>(main_kernel))
      {
        Unary::kernel_t kernel = std::get<Unary>(main_kernel).get_kernel();
        int32_t indexLeadingDimension = isTranspose ? indexPrimM : indexPrimN;
        kernel(ptr_in0, ptr_out, strides_in0[indexPrimN], strides_out[indexLeadingDimension]);
      }
      else if (std::holds_alternative<Brgemm>(main_kernel))
      {
        Brgemm::kernel_t kernel = std::get<Brgemm>(main_kernel).get_kernel();

        if (prim_main == TensorConfig::prim_t::gemm)
        {
          kernel(ptr_in0, ptr_in1, ptr_out, strides_in0[indexPrimK], strides_in1[indexPrimN], strides_out[indexPrimN], 1, 1);
        }
        else if (prim_main == TensorConfig::prim_t::brgemm)
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
    if (last_access && prim_last != TensorConfig::prim_t::none)
    {
      if (std::holds_alternative<Unary>(last_touch))
      {
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

mini_jit::TensorConfig mini_jit::TensorOperation::get_config()
{
  return config;
}

void mini_jit::TensorOperation::write_kernel_to_file(std::string path_no_extension) const
{
  if (prim_first != TensorConfig::prim_t::none && std::holds_alternative<Unary>(first_touch))
  {
    std::get<Unary>(first_touch).write_kernel_to_file(std::format("{}_first_touch.bin", path_no_extension).c_str());
  }

  if (prim_main != TensorConfig::prim_t::none)
  {
    if (std::holds_alternative<Brgemm>(main_kernel))
    {
      std::get<Brgemm>(main_kernel).write_kernel_to_file(std::format("{}_main.bin", path_no_extension).c_str());
    }
    else if (std::holds_alternative<Unary>(main_kernel))
    {
      std::get<Unary>(main_kernel).write_kernel_to_file(std::format("{}_main.bin", path_no_extension).c_str());
    }
  }

  if (prim_last != TensorConfig::prim_t::none && std::holds_alternative<Unary>(last_touch))
  {
    std::get<Unary>(last_touch).write_kernel_to_file(std::format("{}_first_touch.bin", path_no_extension).c_str());
  }
}