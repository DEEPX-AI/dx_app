
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_UYV_H__
#define __ENX_UYV_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_isp.h"

typedef struct 
{
	Int32 chId;
	Int32 width;
	Int32 height;
	Int32 size;

} UYV_CH_PARAM_S;

typedef struct 
{
    UInt32 numChannel;
	UYV_CH_PARAM_S UyvChnParams[MAX_NPU_CH];

} UYV_PARAMS_S;

typedef struct
{
	Int32 chId;
	Int32 width;
	Int32 height;
	Int32 fps;

} UYV_CH_ATTR_S;

typedef struct
{
	Int32 chId;
	BYTE *pVirtAddr;
	UInt32 PhysAddr;
	ULONG nSize;

	ULONG videoInputPTS;
	ULONG videoInputFrameNum;

} UYV_DATA_S;

Int32 ENX_UYV_CAPTURE_Init(UYV_PARAMS_S *pUyvParams);
void ENX_UYV_CAPTURE_Exit(void);

Int32 ENX_UYV_CAPTURE_CH_Start(Int32 nCh, UYV_CH_PARAM_S *pUyvChnParam);
Int32 ENX_UYV_CAPTURE_CH_Stop(Int32 nCh);

Int32 ENX_UYV_CAPTURE_CH_GetAttr(Int32 nCh, UYV_CH_ATTR_S *pUyvAttr);
Int32 ENX_UYV_CAPTURE_CH_SetAttr(Int32 nCh, UYV_CH_ATTR_S *pUyvAttr);

Int32 ENX_UYV_CAPTURE_CH_GetFd(Int32 nCh);
Int32 ENX_UYV_CAPTURE_CH_GetFrame(Int32 nCh, UYV_DATA_S *pUyvData);
Int32 ENX_UYV_CAPTURE_CH_ReleaseFrame(Int32 nCh);

Int32 ENX_UYV_CAPTURE_Debug(void);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_UYV_H__

