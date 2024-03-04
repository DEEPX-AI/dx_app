
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_AUDIO_H__
#define __ENX_AUDIO_H__

#include "enx_common.h"

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

typedef struct __ENX_AUD_REG_INFO {
	unsigned int trx_sync;	// recommand by 1

	// Control register
	unsigned int i2s_mode;	// Slave mode fixed : 0:slave, 1:master

	unsigned int txmode;	// 0:Left, 1:Right, 2:(L+R)/2, 3:Stereo
	unsigned int txcodec;	// 0,1:PCM, 2:G711-a, 3:G711-u
	unsigned int txdw;		// 0:8bits, 1:16bits, 2:24bits, 3:32bits
	unsigned int rdbyte;	// 0:128bytes, 1:256bytes, 2:512bytes, 3:1024bytes
	unsigned int txedn;		// 0:Big, 1:Little
	unsigned int txlr;		// 0:Both off, 1:Right on, 2:Left on, 3:Both on

	unsigned int rxmode;	// 0:Left, 1:Right, 2:(L+R)/2, 3:Stereo
	unsigned int rxcodec;	// 0,1:PCM, 2:G711-a, 3:G711-u
	unsigned int rxdw;		// 0:8bits, 1:16bits, 2:24bits, 3:32bits
	unsigned int wrbyte;	// 0:128bytes, 1:256bytes, 2:512bytes, 3:1024bytes
	unsigned int rxedn;		// 0:Big, 1:Little

	// Data register
	unsigned int rddw;		// 0:8bits, 1:16bits, 2:24bits, 3:32bits
	unsigned int rdlen;		// 0:128KB, 1:256KB, 2:512KB, 3:1024KB
	unsigned int wrdw;		// 0:8bits, 1:16bits, 2:24bits, 3:32bits
	unsigned int wrlen;		// 0:128KB, 1:256KB, 2:512KB, 3:1024KB

	unsigned int txframenum;	// Number of tx buffered frames (range : 2 ~ 100)

} ENX_AUD_REG_INFO_S;

typedef struct __ENX_AUD_DMA_BUFF {
	unsigned int PhysAddr;
	unsigned char *pVirtAddr;
	unsigned int TotalBuffSize;

	unsigned int Offset;
	unsigned int FrameSize;
	unsigned long AudPTS;

} ENX_AUD_DMA_BUFF_S;


Int32 ENX_AUD_Init(void);
void ENX_AUD_Exit(void);
Int32 ENX_AUD_Create(ENX_AUD_REG_INFO_S *pAudInfo);
void ENX_AUD_Destroy(void);
Int32 ENX_AUD_Start(void);
Int32 ENX_AUD_Stop(void);
Int32 ENX_AUD_Play(void);
Int32 ENX_AUD_Pause(void);

Int32 ENX_AUD_GetFd(void);
Int32 ENX_AUD_GetFrame(ENX_AUD_DMA_BUFF_S *pAudBuff);
Int32 ENX_AUD_LoopOut(ENX_AUD_DMA_BUFF_S *pAudBuff, UInt32 uMilliSec);
Int32 ENX_AUD_SendFrame(BYTE *pBuff, Int32 nSize, UInt32 uMilliSec);
Int32 ENX_AUD_GetAttr(ENX_AUD_REG_INFO_S *pAudInfo);
Int32 ENX_AUD_SetAttr(ENX_AUD_REG_INFO_S *pAudInfo);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_AUDIO_H__

