
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_VDEC_H__
#define __ENX_VDEC_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_isp.h"
#include "enx_codec.h"

typedef Int32* (*VdecEventFunc)(VDEC_CH_STATUS_S *);


Int32 ENX_VDEC_Init(VDEC_PARAMS_S *pVdecParams);
void ENX_VDEC_Exit(void);

Int32 ENX_VDEC_MMAP_ChBuffer(VDEC_CH_BUFF_S *pChBuff);

Int32 ENX_VDEC_CH_Create(Int32 nCh, VDEC_CH_PARAM_S *pVdecChParams);
Int32 ENX_VDEC_CH_Destroy(Int32 nCh);

Int32 ENX_VDEC_CH_Start(Int32 nCh);
Int32 ENX_VDEC_CH_Stop(Int32 nCh);

Int32 ENX_VDEC_GetParam();
Int32 ENX_VDEC_SetParam();
Int32 ENX_VDEC_CH_GetParam();
Int32 ENX_VDEC_CH_SetParam();
Int32 ENX_VDEC_CH_GetStatus(Int32 nCh, VDEC_CH_STATUS_S *pVdecStatus);
Int32 ENX_VDEC_CH_SendBitstream(VDEC_BITS_FRAMEBUF_S *pFrameBuf, BYTE *pData);
Int32 ENX_VDEC_CH_ReleaseBitstream();
Int32 ENX_VDEC_Debug(Int32 nDebugLevel);

Int32 ENX_VDEC_CH_GetFrame(Int32 nCh, VDEC_FRAME_INFO_S *pVdecFrameInfo);
void ENX_VDEC_CH_ReleaseFrame(Int32 nCh);

Int32 ENX_VDEC_RegisterEvent(VdecEventFunc callbackFunc);
void ENX_VDEC_UnRegisterEvent(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_DEC_H__
