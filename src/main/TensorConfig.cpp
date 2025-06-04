#include "TensorConfig.h"
#include <algorithm>

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