
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_AFD_H__
#define __ENX_AFD_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_isp.h"

Int32 ENX_AFD_Init(void);
void ENX_AFD_Exit(void);

Int32 ENX_AFD_GetValue(AFD_INFO_S *pAfdData);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_AFD_H__

