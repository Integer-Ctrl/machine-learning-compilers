#ifndef MINI_JIT_UTILS_H
#define MINI_JIT_UTILS_H

#ifdef NDEBUG
#define RELEASE_ASSERT_RESTORE_NDEBUG
#endif 

#ifdef _ASSERT_H
#define RELEASE_ASSERT_RESTORE__ASSERT_H
#endif

#undef NDEBUG
#include <cassert>

#define release_assert(expr, msg) assert((void(msg), expr))

#ifdef RELEASE_ASSERT_RESTORE_NDEBUG
#define NDEBUG
#endif

#ifdef RELEASE_ASSERT_RESTORE__ASSERT_H
#undef _ASSERT_H
#include <cassert>
#endif

#endif // MINI_JIT_UTILS_H