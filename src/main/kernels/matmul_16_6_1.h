#include <cstdint>
#include "../Kernel.h"

namespace mini_jit
{
  namespace kernels
  {
    /**
     * @param kernel The kernel to be generated.
     **/
    void matmul_16_6_1(mini_jit::Kernel &kernel);
  }
}