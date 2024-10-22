// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

// NOTE: these lines may disable some kernel template parameters, which can
//       speedup compiling and reduce object file size significantly.
#define FLASHATTENTION_DISABLE_DROPOUT
// #define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_RETURNSOFTMAX
// #define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_BF16
#define FLASHATTENTION_DISABLE_UNEVEN_K
// #define FLASHATTENTION_DISABLE_BIAS

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

// TRUE/FALSE_FAKESWITCH can be used to set BOOL_SWITCH as always true/false
// without changing codes inside
#define TRUE_FAKESWITCH(COND, CONST_NAME, ...)  \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()

#define FALSE_FAKESWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_BIAS
  #define BIAS_SWITCH(COND, CONST_NAME, ...)    \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define BIAS_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_RETURNSOFTMAX
  #define RETURNSOFTMAX_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define RETURNSOFTMAX_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_BF16
  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
        using elem_type = cutlass::half_t;     \
        return __VA_ARGS__();                  \
    }()
#else
  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
      if (COND) {                              \
        using elem_type = cutlass::half_t;     \
        return __VA_ARGS__();                  \
      } else {                                 \
        using elem_type = cutlass::bfloat16_t; \
        return __VA_ARGS__();                  \
      }                                        \
    }()
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)       \
  [&] {                                    \
    if (HEADDIM <= 64) {                   \
      constexpr static int kQKHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kQKHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kQKHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kQKHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()

#define MLA_HEADDIM_SWITCH(HEADDIM, ...)      \
  [&] {                                       \
    if (HEADDIM == 192) {                     \
      constexpr static int kQKHeadDim = 192;  \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define QUANTBIT_SWITCH(QUANTBIT, ...)     \
  [&] {                                    \
    if (QUANTBIT == 0) {                   \
      constexpr static int QuantBit = 0;   \
      return __VA_ARGS__();                \
    } else if (QUANTBIT == 8) {            \
      constexpr static int QuantBit = 8;   \
      return __VA_ARGS__();                \
    }                                      \
  }()
