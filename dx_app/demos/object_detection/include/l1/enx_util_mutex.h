
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_UTIL_MUTEX_H__
#define __ENX_UTIL_MUTEX_H__

#include <stdio.h>
#include <pthread.h>

#include "enx_types.h"

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

typedef struct {
	pthread_mutex_t lock;
} MUTEX_HNDL_S;


Int32 ENX_UTIL_MutexCreate(MUTEX_HNDL_S *hndl);
Int32 ENX_UTIL_MutexDelete(MUTEX_HNDL_S *hndl);
Int32 ENX_UTIL_MutexLock(MUTEX_HNDL_S *hndl);
Int32 ENX_UTIL_MutexUnlock(MUTEX_HNDL_S *hndl);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __ENX_UTIL_MUTEX_H__ */



