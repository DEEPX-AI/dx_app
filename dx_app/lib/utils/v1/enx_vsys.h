
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_VSYS_H__
#define __ENX_VSYS_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include <stdio.h>

#include "enx_common.h"
#include "enx_codec.h"

Int32 ENX_VSYS_Init(void);
void ENX_VSYS_Exit(void);
Int32 ENX_VSYS_CodecReset(CODEC_MODE_PARAM_S *pCodecModeParam);

void ENX_VSYS_DumpSystemLog(Int32 nDumpType, CHAR *pFileName);
void ENX_VSYS_DumpCodecLog(Int32 nDumpType, CHAR *pFileName);

Int32 ENX_VSYS_GetChipVersion(int *pChipVersion);

Int32 ENX_VSYS_SetBaseTimeStamp(ULONG lSetPTS);
Int32 ENX_VSYS_GetCurrentTimeStamp(ULONG *pCurPTS);

Int32 ENX_VSYS_SetReg(UInt32 uAddr, UInt32 uVal);
Int32 ENX_VSYS_GetReg(UInt32 uAddr, UInt32 *pVal);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_VSYS_H__
