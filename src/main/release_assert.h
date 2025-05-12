#ifndef MINI_JIT_UTILS_H
#define MINI_JIT_UTILS_H

#undef RELEASE_ASSERT_RESTORE_NDEBUG
#ifdef NDEBUG
#define RELEASE_ASSERT_RESTORE_NDEBUG 1
#endif

#undef RELEASE_ASSERT_RESTORE__ASSERT_H
#ifdef _ASSERT_H
#define RELEASE_ASSERT_RESTORE__ASSERT_H 1
#endif

#undef NDEBUG
#include <cassert>

#undef _release_assert
#undef release_assert
#define release_assert(expr, msg) _release_assert((void(msg), expr))
/// partial COPY of the assert definition to be used also in release mode
#if defined __cplusplus
#define _release_assert(expr) (static_cast<bool>(expr) ? void(0) : __assert_fail(#expr, __ASSERT_FILE, __ASSERT_LINE, __ASSERT_FUNCTION))
#elif !defined __GNUC__ || defined __STRICT_ANSI__
#define _release_assert(expr) ((expr) ? __ASSERT_VOID_CAST(0) : __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
#else
#define _release_assert(expr)                                                                                                              \
  ((void)sizeof((expr) ? 1 : 0), __extension__({                                                                                           \
     if (expr)                                                                                                                             \
       ; /* empty */                                                                                                                       \
     else                                                                                                                                  \
       __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION);                                                                        \
   }))
#endif

#ifdef RELEASE_ASSERT_RESTORE_NDEBUG
// #define NDEBUG
#endif

#ifdef RELEASE_ASSERT_RESTORE__ASSERT_H
#undef _ASSERT_H
#include <cassert>
#endif

#endif  // MINI_JIT_UTILS_H