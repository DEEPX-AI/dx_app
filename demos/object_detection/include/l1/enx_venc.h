
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_VENC_H__
#define __ENX_VENC_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_codec.h"

//The two functions operate in pairs.
Int32 ENX_VENC_Init(VENC_PARAMS_S *pVencParams);
Int32 ENX_VENC_InitWithParam(VENC_PARAMS_S *pVencParams, VENC_MODE_PARAM_S *pVencModeParam);
void ENX_VENC_Exit(void);

// Allocate channel buffer
Int32 ENX_VENC_MMAP_ChBuffer(VENC_CH_BUFF_S *pChBuff);

//The two functions operate in pairs.
Int32 ENX_VENC_CH_Create(Int32 nCh, VENC_CH_PARAM_S *pVencChParams);
Int32 ENX_VENC_CH_Destroy(Int32 nCh);

//The two functions operate in pairs.
Int32 ENX_VENC_CH_Start(Int32 nCh);
Int32 ENX_VENC_CH_Stop(Int32 nCh);

// Get Streaming data functions
Int32 ENX_VENC_CH_GetFd(Int32 nCh);
Int32 ENX_VENC_CH_GetBitsStreamInfo(Int32 nCh, BITS_STREAMINFO_S *pBitsStreamInfo);
Int32 ENX_VENC_CH_GetStatus(Int32 nCh, BITS_STREAMSTATUS_S *pStreamStatus);
Int32 ENX_VENC_CH_GetBitsStream(Int32 nCh, BITS_STREAMBUF_S *pBitsStreamBuf);
void ENX_VENC_CH_ReleaseBitsStream(Int32 nCh);

// Managed VENC Parameters
Int32 ENX_VENC_GetParams(VENC_PARAMS_S *pVencParams);
Int32 ENX_VENC_SetParams(VENC_PARAMS_S *pVencParams);

Int32 ENX_VENC_CH_GetParam(Int32 nCh, VENC_CH_PARAM_S *pVencChParams);
Int32 ENX_VENC_CH_SetParam(Int32 nCh, VENC_CH_PARAM_S *pVencChParams);

// VENC Dynamic Parameters
Int32 ENX_VENC_CH_GetDynamicParam(Int32 nCh, VENC_CH_DYNAMIC_PARAM_S *pVencChParams);
Int32 ENX_VENC_CH_SetDynamicParam(VENC_CH_DYNAMIC_PARAM_S *pVencChParams);

Int32 ENX_VENC_CH_GetFrameRate(Int32 nCh, VENC_CH_FRAMERATE_PARAM_S *pVencFrameRate);
Int32 ENX_VENC_CH_GetTargetBitRate(Int32 nCh, VENC_CH_BITRATE_PARAM_S *pVencBitRate);
Int32 ENX_VENC_CH_GetIdr(Int32 nCh, VENC_CH_IDR_PARAM_S *pVencIdr);
Int32 ENX_VENC_CH_GetRateCtrl(Int32 nCh, VENC_CH_RATECTRL_PARAM_S *pVencRateCtrl);
Int32 ENX_VENC_CH_GetQPParam(Int32 nCh, VENC_CH_QP_PARAM_S *pVencQP);
Int32 ENX_VENC_CH_GetRoiParam(Int32 nCh, VENC_ROI_PARAMS_S *pVencRoi);
Int32 ENX_VENC_CH_GetSmartBackground(Int32 nCh, VENC_CH_SMARTBG_PARAM_S *pVencSmartBG);
Int32 ENX_VENC_CH_GetHeaderInc(Int32 nCh, VENC_CH_HEADERINC_PARAM_S *pVencHeaderInc);
Int32 ENX_VENC_CH_Get3DNR(Int32 nCh, VENC_CH_3DNR_PARAM_S *pVenc3DNR);
Int32 ENX_VENC_CH_GetDblk(Int32 nCh, VENC_CH_DBLK_PARAM_S *pVencDblk);
Int32 ENX_VENC_CH_GetAvcTrans(Int32 nCh, VENC_CH_AVC_TRANS_PARAM_S *pVencAvcTrans);
Int32 ENX_VENC_CH_GetAvcEntropy(Int32 nCh, VENC_CH_AVC_ENTROPY_PARAM_S *pVencAvcEntropy);
Int32 ENX_VENC_CH_GetEnableIdr(Int32 nCh, VENC_CH_ENABLE_IDR_PARAM_S *pVencEnableIdr);
Int32 ENX_VENC_CH_GetRequestIdr(Int32 nCh, VENC_CH_REQUEST_IDR_PARAM_S *pVencRequestIdr);
Int32 ENX_VENC_CH_GetUserOpt(Int32 nCh, Int32 nUserOpt);

Int32 ENX_VENC_CH_SetFrameRate(VENC_CH_FRAMERATE_PARAM_S *pVencFrameRate);
Int32 ENX_VENC_CH_SetTargetBitRate(VENC_CH_BITRATE_PARAM_S *pVencBitRate);
Int32 ENX_VENC_CH_SetIdr(VENC_CH_IDR_PARAM_S *pVencIdr);
Int32 ENX_VENC_CH_SetRateCtrl(VENC_CH_RATECTRL_PARAM_S *pVencRateCtrl);
Int32 ENX_VENC_CH_SetQPParam(VENC_CH_QP_PARAM_S *pVencQP);
Int32 ENX_VENC_CH_SetRoiParam(VENC_ROI_PARAMS_S *pVencRoi);
Int32 ENX_VENC_CH_SetSmartBackground(VENC_CH_SMARTBG_PARAM_S *pVencSmartBG);
Int32 ENX_VENC_CH_SetHeaderInc(VENC_CH_HEADERINC_PARAM_S *pVencHeaderInc);
Int32 ENX_VENC_CH_Set3DNR(VENC_CH_3DNR_PARAM_S *pVenc3DNR);
Int32 ENX_VENC_CH_SetDblk(VENC_CH_DBLK_PARAM_S *pVencDblk);
Int32 ENX_VENC_CH_SetAvcTrans(VENC_CH_AVC_TRANS_PARAM_S *pVencAvcTrans);
Int32 ENX_VENC_CH_SetAvcEntropy(VENC_CH_AVC_ENTROPY_PARAM_S *pVencAvcEntropy);
Int32 ENX_VENC_CH_SetEnableIdr(VENC_CH_ENABLE_IDR_PARAM_S *pVencEnableIdr);
Int32 ENX_VENC_CH_SetRequestIdr(VENC_CH_REQUEST_IDR_PARAM_S *pVencRequestIdr);
Int32 ENX_VENC_CH_SetUserOpt(Int32 nCh, Int32 nUserOpt);


// ========================================================================================================
// Not supported yet.
// Extended dynamic codec parameters
// ========================================================================================================
#ifdef USE_EXTEND_CODEC_PARAM
Int32 ENX_VENC_CH_GetDynParamEx(Int32 nCh, VENC_CH_DYN_PARAM_EX_S *pDynParams);
Int32 ENX_VENC_CH_SetDynParamEx(Int32 nCh, VENC_CH_DYN_PARAM_EX_S *pDynParams);

Int32 ENX_VENC_CH_GetDynFpsEx(Int32 nCh, Int32 *pFps);
Int32 ENX_VENC_CH_GetDynBpsEx(Int32 nCh, Int32 *pBps);
Int32 ENX_VENC_CH_GetDynRoiEx(Int32 nCh, VENC_CH_ROI_PARAM_S *pRoi);
Int32 ENX_VENC_CH_GetDynPPSEx(Int32 nCh, VENC_CH_DYN_PPS_S *pPPS);
Int32 ENX_VENC_CH_GetDynIndeSliceEx(Int32 nCh, VENC_CH_DYN_INDESLICE_S *pIndeSlice);
Int32 ENX_VENC_CH_GetDynDeSliceEx(Int32 nCh, VENC_CH_DYN_DESLICE_S *pDeSlice);
Int32 ENX_VENC_CH_GetDynRdoEx(Int32 nCh, VENC_CH_DYN_RDO_S *pRdo);
Int32 ENX_VENC_CH_GetDynRcEx(Int32 nCh, VENC_CH_DYN_RC_S *pRc);
Int32 ENX_VENC_CH_GetDynRcMinMaxQpEx(Int32 nCh, VENC_CH_DYN_RC_MIN_MAX_QP_S *pMinMaxQp);
Int32 ENX_VENC_CH_GetDynRcInterMinMaxQpEx(Int32 nCh, VENC_CH_DYN_RC_INTER_MIN_MAX_QP_S *pInterMinMaxQp);
Int32 ENX_VENC_CH_GetDynIntraParamEx(Int32 nCh, VENC_CH_DYN_INTRA_PARAM_S *pIntraParam);
Int32 ENX_VENC_CH_GetDynBgEx(Int32 nCh, VENC_CH_DYN_BG_S *pBg);
Int32 ENX_VENC_CH_GetDynNrEx(Int32 nCh, VENC_CH_DYN_NR_S *pNr);

Int32 ENX_VENC_CH_SetDynFpsEx(Int32 nCh, Int32 *pFps);
Int32 ENX_VENC_CH_SetDynBpsEx(Int32 nCh, Int32 *pBps);
Int32 ENX_VENC_CH_SetDynRoiEx(Int32 nCh, VENC_CH_ROI_PARAM_S *pRoi);
Int32 ENX_VENC_CH_SetDynPPSEx(Int32 nCh, VENC_CH_DYN_PPS_S *pPPS);
Int32 ENX_VENC_CH_SetDynIndeSliceEx(Int32 nCh, VENC_CH_DYN_INDESLICE_S *pIndeSlice);
Int32 ENX_VENC_CH_SetDynDeSliceEx(Int32 nCh, VENC_CH_DYN_DESLICE_S *pDeSlice);
Int32 ENX_VENC_CH_SetDynRdoEx(Int32 nCh, VENC_CH_DYN_RDO_S *pRdo);
Int32 ENX_VENC_CH_SetDynRcEx(Int32 nCh, VENC_CH_DYN_RC_S *pRc);
Int32 ENX_VENC_CH_SetDynRcMinMaxQpEx(Int32 nCh, VENC_CH_DYN_RC_MIN_MAX_QP_S *pMinMaxQp);
Int32 ENX_VENC_CH_SetDynRcInterMinMaxQpEx(Int32 nCh, VENC_CH_DYN_RC_INTER_MIN_MAX_QP_S *pInterMinMaxQp);
Int32 ENX_VENC_CH_SetDynIntraParamEx(Int32 nCh, VENC_CH_DYN_INTRA_PARAM_S *pIntraParam);
Int32 ENX_VENC_CH_SetDynBgEx(Int32 nCh, VENC_CH_DYN_BG_S *pBg);
Int32 ENX_VENC_CH_SetDynNrEx(Int32 nCh, VENC_CH_DYN_NR_S *pNr);
#endif
// ========================================================================================================


// Test Function
Int32 ENX_VENC_Debug(Int32 nDebugLevel);
Int32 ENX_VENC_DefaultReset(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_VENC_H__

