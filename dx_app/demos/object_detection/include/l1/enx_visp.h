
/*
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_VISP_H__
#define __ENX_VISP_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include "enx_common.h"
#include "enx_isp.h"


Int32 ENX_VISP_DefaultReset(void);

Int32 ENX_VISP_GetPar(VISP_PARAM_UNIT_S *pParStr, BYTE bParamCnt);
Int32 ENX_VISP_GetParAttr(VISP_PARAM_UNIT_S *pParStr, VISP_PARAM_ATTR_S *pParAttr, BYTE bParamCnt);
Int32 ENX_VISP_GetHWPar(VISP_PARAM_UNIT_S *pParStr, BYTE bParamCnt);
Int32 ENX_VISP_GetYCInfo(Int32 nYcNum, UP_MENU_YC_S *pVispYCInfo);
Int32 ENX_VISP_GetYCInfoEx(Int32 nYcNum, VISP_YCEX_PARAM_S *pVispYCEXParam);
Int32 ENX_VISP_GetFontAttr(VISP_FONT_ATTR_E eId, UP_FONT_ATTR_S *pVispFontAttr);
Int32 ENX_VISP_GetBoxPos(VISP_BOX_POS_S stBoxPos[MAX_VISP_BOX]);
Int32 ENX_VISP_GetBoxAttr(VISP_BOX_ATTR_S stBoxAttr[MAX_VISP_BOX]);
Int32 ENX_VISP_GetMotionArea(Int32 nImdNum, VISP_BOX_POS_S *pVispIMDArea);
Int32 ENX_VISP_GetStat(VISP_STAT_S *pStatStr);

Int32 ENX_VISP_SetPar(VISP_PARAM_UNIT_S *pParStr, BYTE bParamCnt);
Int32 ENX_VISP_SetYCInfo(Int32 nYcNum, VISP_YC_E eId, UP_MENU_YC_S *pVispYCInfo);
Int32 ENX_VISP_SetYCInfoEx(Int32 nYcNum, VISP_YCEX_PARAM_S *pVispYCEXParam);
Int32 ENX_VISP_SetFontText(VISP_FONT_S *pInputText);
Int32 ENX_VISP_SetFontColor(const Int32 nColorNum, Int32 nPosX, Int32 nPosY, Int32 nLen);
Int32 ENX_VISP_SetFontAttr(VISP_FONT_ATTR_E eId, UP_FONT_ATTR_S *pVispFontAttr);
Int32 ENX_VISP_SetFontChar(UINT *pFontChar, WCHAR bCharNum, BYTE bCharCnt);
Int32 ENX_VISP_SetBoxPos(VISP_BOX_POS_S stBoxPos[MAX_VISP_BOX]);
Int32 ENX_VISP_SetBoxAttr(VISP_BOX_ATTR_S stBoxAttr[MAX_VISP_BOX]);
Int32 ENX_VISP_SetMotionArea(Int32 nImdNum, VISP_BOX_POS_E eId, VISP_BOX_POS_S *pVispIMDArea);

Int32 ENX_VISP_SetNPUObjects(Int32 nType, Int32 nStart);
Int32 ENX_VISP_SendNPUObjects(Int32 nCh, NPU_RESULT_S *pNPUResult);

Int32 ENX_VISP_GetYCMapInfo(Int32 nYcNum, VISP_YCEX_PARAM_S *pVispYCEXParam);
void ENX_VISP_ReleaseYCMapInfo(Int32 nYcNum);
Int32 ENX_VISP_UpdateYCMap(Int32 nYcNum, VISP_UPDATE_YC_S *pUpdateYC);

Int32 ENX_VISP_GetYCFd(void);
Int32 ENX_VISP_GetYCPageNum(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_VISP_H__

