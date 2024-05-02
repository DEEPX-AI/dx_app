
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_UTIL_THR_H__
#define __ENX_UTIL_THR_H__

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include "enx_types.h"
#include "dbg_prn.h"

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#define ENX_THR_PRI_MAX                 sched_get_priority_max(SCHED_FIFO)
#define ENX_THR_PRI_MIN                 sched_get_priority_min(SCHED_FIFO)

#define ENX_THR_PRI_DEFAULT             ( ENX_THR_PRI_MIN + (ENX_THR_PRI_MAX-ENX_THR_PRI_MIN)/2 )

#define ENX_THR_STACK_SIZE_DEFAULT      0

typedef void * (*ThrEntryFunc)(void *);

typedef struct {
	pthread_t      hndl;
} THR_HNDL_S;

Int32 ENX_UTIL_thrCreate(THR_HNDL_S *hndl, ThrEntryFunc entryFunc, UINT pri, UINT stackSize, void *prm);
Int32 ENX_UTIL_thrDelete(THR_HNDL_S *hndl);
Int32 ENX_UTIL_thrJoin(THR_HNDL_S *hndl);
Int32 ENX_UTIL_thrChangePri(THR_HNDL_S *hndl, UINT pri);
Int32 ENX_UTIL_thrExit(void *returnVal);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __ENX_UTIL_THR_H__ */



