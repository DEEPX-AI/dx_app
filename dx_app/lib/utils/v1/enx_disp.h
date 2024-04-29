
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_DISP_H__
#define __ENX_DISP_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_isp.h"

typedef struct _DISP_DATA
{
	Int32 xoffset;
	Int32 yoffset;
	Int32 width;
	Int32 height;
	Int32 depth;	// bits per pixel
	BYTE *pVirtAddr;
	UINT PhysAddr;
	ULONG size;

} DISP_DATA_S;

Int32 ENX_DISP_Init(DISP_DATA_S *pDispInfo);
void ENX_DISP_Exit(void);

Int32 ENX_DISP_GetFd(void);
Int32 ENX_DISP_Update(Int32 nLayerIndex, DISP_DATA_S *pDispInfo);

Int32 ENX_DISP_GetTransparency(UCHAR *pTransValue);
Int32 ENX_DISP_SetTransparency(UCHAR *pTransValue);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_DISP_H__

