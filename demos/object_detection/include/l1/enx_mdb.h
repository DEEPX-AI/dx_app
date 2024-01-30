
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_MDB_H__
#define __ENX_MDB_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_isp.h"

Int32 ENX_MDB_Init(void);
void ENX_MDB_Exit(void);

Int32 ENX_MDB_GET_MotionBitmap(MOTION_BITMAP_S *pBitmap);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_MDB_H__

