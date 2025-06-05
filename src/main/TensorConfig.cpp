#include "TensorConfig.h"
#include <algorithm>
#include <cstdint>
#include <string>

bool mini_jit::TensorConfig::equals(const TensorConfig &config1, const TensorConfig config2)
{
  return config1.first_touch == config2.first_touch && config1.main == config2.main && config1.last_touch == config2.last_touch &&
         config1.dtype == config2.dtype && config1.dim_types.size() == config2.dim_types.size() &&
         config1.exec_types.size() == config2.exec_types.size() && config1.dim_sizes.size() == config2.dim_sizes.size() &&
         config1.strides_in0.size() == config2.strides_in0.size() && config1.strides_in1.size() == config2.strides_in1.size() &&
         config1.strides_out.size() == config2.strides_out.size() &&
         std::equal(config1.dim_types.begin(), config1.dim_types.end(), config2.dim_types.begin()) &&
         std::equal(config1.exec_types.begin(), config1.exec_types.end(), config2.exec_types.begin()) &&
         std::equal(config1.dim_sizes.begin(), config1.dim_sizes.end(), config2.dim_sizes.begin()) &&
         std::equal(config1.strides_in0.begin(), config1.strides_in0.end(), config2.strides_in0.begin()) &&
         std::equal(config1.strides_in1.begin(), config1.strides_in1.end(), config2.strides_in1.begin()) &&
         std::equal(config1.strides_out.begin(), config1.strides_out.end(), config2.strides_out.begin());
}

std::string mini_jit::TensorConfig::to_string() const
{
  std::string result = "TensorConfig: {\n";
  result += "    first_touch: " + std::to_string(static_cast<uint32_t>(first_touch)) + ",\n";
  result += "    main: " + std::to_string(static_cast<uint32_t>(main)) + ",\n";
  result += "    last_touch: " + std::to_string(static_cast<uint32_t>(last_touch)) + ",\n";
  result += "    dtype: " + std::to_string(static_cast<uint32_t>(dtype)) + ",\n";

  result += "    dim_types: [ ";
  for (const auto &dim : dim_types)
    result += std::to_string(static_cast<uint32_t>(dim)) + " ";
  result += "],\n";

  result += "    exec_types: [ ";
  for (const auto &exec : exec_types)
    result += std::to_string(static_cast<uint32_t>(exec)) + " ";
  result += "],\n";

  result += "    dim_sizes: [ ";
  for (const auto &size : dim_sizes)
    result += std::to_string(size) + " ";
  result += "],\n";

  result += "    strides_in0: [ ";
  for (const auto &stride : strides_in0)
    result += std::to_string(stride) + " ";
  result += "],\n";

  result += "    strides_in1: [ ";
  for (const auto &stride : strides_in1)
    result += std::to_string(stride) + " ";
  result += "],\n";

  result += "    strides_out: [ ";
  for (const auto &stride : strides_out)
    result += std::to_string(stride) + " ";
  result += "]\n}";

  return result;
}