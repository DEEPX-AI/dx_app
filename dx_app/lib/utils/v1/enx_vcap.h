
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_VCAP_H__
#define __ENX_VCAP_H__

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
	Int32 nYcNum;

} VCAP_CH_PARAM_S;

typedef struct 
{
    UInt32 numChannel;
	VCAP_CH_PARAM_S VcapChnParams[MAX_VCAP_CH];

} VCAP_PARAMS_S;

typedef struct
{
	Int32 chId;
	Int32 width;
	Int32 height;
	Int32 fps;

} VCAP_CH_ATTR_S;

typedef struct
{
	Int32 chId;
	BYTE *pVirtAddr;
	UInt32 PhysAddr;
	ULONG nSize;

	ULONG videoInputPTS;
	ULONG videoInputFrameNum;

} VCAP_DATA_S;



Int32 ENX_VCAP_Init(VCAP_PARAMS_S *pVcapParams);
void ENX_VCAP_Exit(void);

Int32 ENX_VCAP_CH_Start(Int32 nCh, VCAP_CH_PARAM_S *pVcapChnParam);
Int32 ENX_VCAP_CH_Stop(Int32 nCh);

Int32 ENX_VCAP_CH_GetAttr(Int32 nCh, VCAP_CH_ATTR_S *pVcapAttr);
Int32 ENX_VCAP_CH_SetAttr(Int32 nCh, VCAP_CH_ATTR_S *pVcapAttr);

Int32 ENX_VCAP_CH_GetFd(Int32 nCh);
Int32 ENX_VCAP_CH_GetFrame(Int32 nCh, VCAP_DATA_S *pVcapDatar);
Int32 ENX_VCAP_CH_ReleaseFrame(Int32 nCh);

Int32 ENX_VCAP_Debug(void);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_VCAP_H__

