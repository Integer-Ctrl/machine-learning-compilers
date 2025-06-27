#ifndef MLC_ERROR_H
#define MLC_ERROR_H
#include <string>
#include <cstdint>

namespace mlc
{
  enum ErrorType : int64_t
  {
    Undefined = -1,
    None = 0,

    // Parse Errors
    ParseExpectedLeftBracket = 1,
    ParseExpectedRightBracket = 2,
    ParseExpectedArrow = 3,
    ParseExpectedComma = 4,
    ParseExpectedDimensionList = 5,
    ParseNotAllowedToParseAgain = 6,
    ParseUndefinedNode = 7,

    // Einsum Errors
    EinsumInvalidRoot = 8,
    EinsumNotEnoughInputTensors = 9,
    EinsumTooManyInputTensors = 10,
    EinsumNullPtrAsInputTensor = 11,

    // Execute Errors
    ExecuteWrongDType = 101,
    ExecuteWrongDimension = 102,
    ExecuteWrongPrimitive = 103,
    ExecuteFirstTouchPrimitive = 104,
    ExecuteWrongFirstTouchPrimitive = 104,
    ExecuteWrongMainPrimitive = 105,
    ExecuteWrongLastTouchPrimitive = 106,
    ExecuteTypeNotSupported = 107,
    ExecuteInvalidPrimitiveConfiguration = 108,
    ExecuteInvalidFirstTouchConfiguration = 109,
    ExecuteInvalidMainConfiguration = 110,
    ExecuteInvalidLastTouchConfiguration = 111,
    ExecuteInvalidExecutionOrder = 112,
    ExecuteInvalidStrides = 113,
    ExecuteKDimensionMustNotBeShared = 114,
    ExecuteSharedRequiredForParallelExecution = 115,
  };

  struct Error
  {
    ErrorType type;
    std::string message;
  };

}  // namespace mlc

#endif  // MLC_ERROR_H