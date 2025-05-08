#include "Brgemm.h"
#include "kernels/matmul_16_6_1.h"
#include "kernels/matmul_16_6_k.h"
#include "Kernel.h"

mini_jit::Brgemm::error_t mini_jit::Brgemm::generate(uint32_t m, uint32_t n, uint32_t k, uint32_t br_size, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, dtype_t  dtype)
{
    if (dtype != dtype_t::fp32)
    {
        return error_t::err_wrong_dtype;
    }
    if (m != 16 || n != 6)
    {
        return error_t::err_wrong_dimension;
    }
    if ((trans_a + trans_b + trans_c) != 0)
    {
        return error_t::err_row_major_order_not_supported;
    }
    if (br_size != 1)
    {
        return error_t::err_batch_reduce_size_not_supported;
    }

    if (k == 1)
    {
      kernels::matmul_16_6_1(native_kernel);
    }
    
    kernels::matmul_16_6_k(native_kernel, k);

    native_kernel.set_kernel();
    kernel = reinterpret_cast<kernel_t>(const_cast<void*>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

    return error_t::success;
}

mini_jit::Brgemm::kernel_t mini_jit::Brgemm::get_kernel() const
{
    return kernel;
}