#include <cstdint>
#include "../Kernel.h"

namespace mini_jit
{
  namespace kernels
  {
    /**
     * @param kernel The kernel to be generated.
     **/
    void matmul_16_6_k(mini_jit::Kernel &kernel, const uint32_t k_loop);
  }
}