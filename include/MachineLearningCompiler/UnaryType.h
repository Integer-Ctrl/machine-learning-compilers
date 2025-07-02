#ifndef MLC_UNARY_H
#define MLC_UNARY_H
#include <cstdint>

namespace mlc
{
  enum class UnaryType : int64_t
  {
    None = 0,
    Zero = 1,
    ReLU = 2,
    Identity = 3,
  };
}  // namespace mlc

#endif  // MLC_UNARY_H