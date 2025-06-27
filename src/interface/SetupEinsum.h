#ifndef MLC_SETUPEINSUM_H
#define MLC_SETUPEINSUM_H

#include "../../include/MachineLearningCompiler/Setup.h"
#include "../main/EinsumTree.h"

namespace mlc
{
  class SetupEinsum : public Setup
  {
  public:
    SetupEinsum(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output, const std::string &tree);
    SetupEinsum(const std::vector<Tensor *> &inputs, Tensor &output, const std::string &tree);

    virtual Error execute(const std::vector<std::reference_wrapper<Tensor>> &inputs, Tensor &output) override;
    virtual Error execute(const std::vector<Tensor *> &inputs, Tensor &output) override;
    virtual Error getSetupError() const override;

  private:
    std::vector<uint64_t> sortedDimSizes;
    Error error;
    mini_jit::EinsumTree einsumTree;
  };

}  // namespace mlc

#endif  // MLC_SETUPEINSUM_H