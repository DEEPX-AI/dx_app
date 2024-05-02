
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_DEBUG_H__
#define __ENX_DEBUG_H__

#include <stdio.h>

// printf wrapper that can be turned on and off by defining/undefining
#ifdef ENX_DEBUG_MODE
           
  #define ENX_printf(...)  do { printf("\n\r [host] " __VA_ARGS__); fflush(stdout); } while(0)

// #define __KLOCDBG__
#ifdef __KLOCDBG__
  #define ENX_assert(x)  \
  { \
    if( (x) == 0) { \
      fprintf(stderr, " ASSERT (%s|%s|%d)\r\n", __FILE__, __func__, __LINE__); \
      abort();\
    } \
  }
#else
  #define ENX_assert(x)  \
  { \
    if( (x) == 0) { \
      fprintf(stderr, " ASSERT (%s|%s|%d)\r\n", __FILE__, __func__, __LINE__); \
      while (getchar()!='q')  \
        ; \
    } \
  } 
#endif
    #define UTILS_assert(x)   ENX_assert(x)
               
#define ENX_DEBUG \
  fprintf(stderr, " %s:%s:%d Press Any key to continue !!!", __FILE__, __func__, __LINE__); 


#define ENX_DEBUG_WAIT \
  ENX_DEBUG \
  getchar();

#define ENX_COMPILETIME_ASSERT(condition)                                       \
                   do {                                                         \
                       typedef char ErrorCheck[((condition) == TRUE) ? 1 : -1]; \
                   } while(0)

#else
  
  #define ENX_printf(...)
  #define ENX_assert(x)
  #define UTILS_assert(x)  
  #define ENX_DEBUG    
  #define ENX_DEBUG_WAIT  
#endif

// printf wrapper that can be used to display errors. Prefixes all text with
// "ERROR" and inserts the file and line number of where in the code it was
// called
#define ENX_ERROR(...) \
  do \
  { \
  fprintf(stderr, " ERROR  (%s|%s|%d): ", __FILE__, __func__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  } \
  while(0);

#define ENX_assertSuccess(ret)  ENX_assert(ret==0)


#endif  /* __ENX_DEBUG_H__ */



