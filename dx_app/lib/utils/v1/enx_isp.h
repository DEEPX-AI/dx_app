
/*
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_ISP_H__
#define __ENX_ISP_H__

#include "enx_types.h"

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

typedef enum
{
	VISP_BOX_On=0,
	VISP_BOX_PosX,
	VISP_BOX_PosY,
	VISP_BOX_SizX,
	VISP_BOX_SizY,
	VISP_BOX_All,
} VISP_BOX_POS_E;

typedef enum
{
	VISP_FONT_SetByAPI=0,
	VISP_FONT_FontMute,
	VISP_FONT_BoadMode,
	VISP_FONT_FontCntX,
	VISP_FONT_FontCntY,
	VISP_FONT_BlankV,
	VISP_FONT_BlankH,
	VISP_FONT_SizeV,
	VISP_FONT_SizeH,
	VISP_FONT_OffsetV,
	VISP_FONT_OffsetH,
	VISP_FONT_ColorData0,
	VISP_FONT_ColorData1,
	VISP_FONT_ColorData2,
	VISP_FONT_ColorData3,
	VISP_FONT_All,
} VISP_FONT_ATTR_E;

typedef enum
{
	VISP_YC_In=0,
	VISP_YC_Out,
	VISP_YC_Font,
	VISP_YC_PageSize,
	VISP_YC_Width,
	VISP_YC_Height,
	VISP_YC_DownScaler,
	VISP_YC_All,
} VISP_YC_E;

typedef enum
{
	VISP_DS_AUTO=0,
	VISP_DS_SMALL,
	VISP_DS_BIG,
	VISP_DS_UNUSED,
	VISP_DS_CROP
} VISP_DOWNSCALER_TYPE_E;

typedef enum {
	YC_IN_ISP=0,
	YC_IN_DIN0=1,
	YC_IN_DIN1=2,
	YC_IN_DIN2=3,
	YC_IN_DIN3=4,
	YC_IN_JPEG=6,
	YC_IN_DEC=8,
	YC_IN_USER=8,

} VISP_YCIN_TYPE_E;

typedef enum {
	YC_OUT_ENC=0,
	YC_OUT_ISP,
} VISP_YCOUT_TYPE_E;

typedef enum {
	YC_FONT_OFF=0,
	YC_FONT_ON,
} VISP_YCFONT_TYPE_E;

typedef enum
{
	TYPE_AF1_SUM1_LOCK = 0,		// AF window1 FIR1 integral value
	TYPE_AF1_SUM2_LOCK,			// AF window1 FIR2 integral value
	TYPE_AF2_SUM1_LOCK,			// AF window2 FIR1 integral value
	TYPE_AF2_SUM2_LOCK,			// AF window2 FIR2 integral value
	TYPE_AF1_CLP_SUM1_LOCK,		// AF window1 FIR1 integral value except clip area
	TYPE_AF1_CLP_SUM2_LOCK,		// AF window1 FIR2 integral value except clip area
	TYPE_AF2_CLP_SUM1_LOCK,		// AF window2 FIR1 integral value except clip area
	TYPE_AF2_CLP_SUM2_LOCK,		// AF window2 FIR2 integral value except clip area
	TYPE_AF1_YSUM1_LOCK,		// Y integral value of applied area to AF window1 FIR1 filter
	TYPE_AF1_YSUM2_LOCK,		// Y integral value of applied area to AF window1 FIR2 filter
	TYPE_AF2_YSUM1_LOCK,		// Y integral value of applied area to AF window2 FIR1 filter
	TYPE_AF2_YSUM2_LOCK,		// Y integral value of applied area to AF window2 FIR2 filter
	TYPE_AF1_CLCNT_LOCK,		// AF window1 clip count value
	TYPE_AF2_CLCNT_LOCK,		// AF window2 clip count value
	TYPE_VAF1_SUM_LOCK,			// AF window1 vertical integral value
	TYPE_VAF2_SUM_LOCK,			// AF window2 vertical integral value

	TYPE_AFD_MAX,
} AFD_TYPE_E;

typedef struct {
	char ID[4];
	WORD VAL;
} VISP_PARAM_UNIT_S;

typedef struct {
	WORD Def;
	WORD Min;
	WORD Max;
} VISP_PARAM_ATTR_S;

typedef struct {
	UINT nPageNum;

} VISP_UPDATE_YC_S;

typedef struct {
	UINT PhysAddr;
	BYTE *pVirtAddr;
	UINT MemSize;

} VISP_YC_MAPINFO_S;

#pragma pack(push, 1)
typedef struct {
    BYTE In;
    BYTE Out;
    BYTE Font;
    BYTE PageSize;
    WORD Width;
    WORD Height;
	BYTE DownScaler; //0 auto, 1 small size, 2 big size, 3 unused.
} UP_MENU_YC_S;
#pragma pack(pop)

typedef struct {
	WORD PosX;
	WORD PosY;

} VISP_YC_CROP_POS_S;

typedef struct {
	// YC common param
	UP_MENU_YC_S stYCInfo;

	// special param
	VISP_YC_CROP_POS_S stCropPos;
	VISP_YC_MAPINFO_S stMapInfo[MAX_VISP_YC_PAGE];

	BYTE nUserOn;	// read only

} VISP_YCEX_PARAM_S;

typedef struct {
	BYTE SetByAPI;
	BYTE FontMute;
	BYTE BoadMode;
	UINT FontCntX;
	UINT FontCntY;
	UINT BlankV;
	UINT BlankH;
	UINT SizeV;
	UINT SizeH;
	UINT OffsetV;
	UINT OffsetH;
	UINT ColorData0;
	UINT ColorData1;
	UINT ColorData2;
	UINT ColorData3;
} UP_FONT_ATTR_S;

typedef struct {
	BYTE On;
	WORD PosX;
	WORD PosY;
	WORD SizX;
	WORD SizY;
} VISP_BOX_POS_S;

typedef struct {
	BYTE Pattern;
	BYTE Fill;
	BYTE Alpha;
	BYTE ColorY;
	BYTE ColorCb;
	BYTE ColorCr;
} VISP_BOX_ATTR_S;

typedef struct {
	WCHAR* pStr;
	BYTE Length;
	BYTE PosX;
	BYTE PosY;
} TEXT_S;

typedef struct {
	BYTE TextNum;
	TEXT_S pTextStr[32];
} VISP_FONT_S;

typedef struct
{
    SHORT x_min;
    SHORT x_max;
    SHORT y_min;
    SHORT y_max;
    SHORT npu_class;
    SHORT npu_score;

} NPU_OBJ_S;

typedef struct
{
    int cnt;
    NPU_OBJ_S obj[256];

} NPU_RESULT_S;

typedef struct
{
	AFD_TYPE_E eAfdIndex;
    UInt32 nAfdValue;

} AFD_INFO_S;

typedef struct
{
    UInt32 AfdVal[MAX_AFD_REG];

} VISP_STAT_AF_S;

typedef struct
{
    UInt32 Bitmap[MAX_MDB_REG];

} MOTION_BITMAP_S;

typedef struct
{
	UInt32 AeVal[MAX_AED_REG]; // Value [1] is data of default AE. Other arrays require additional control to obtain valid data.
	UInt32 AeSliceVal[MAX_AED_REG];
	UInt32 AeClipVal[MAX_AED_REG];
	UInt32 AeSht;
	UInt32 AeAgc;
	UInt32 AeStatIris;

} VISP_STAT_AE_S;

typedef struct
{
	UInt32 AwbCntVal;
	UInt32 AwbYVal;
	UInt32 AwbCbDbVal;
	UInt32 AwbCrDrVal;

} VISP_STAT_AWB_S;

typedef MOTION_BITMAP_S VISP_STAT_MDB_BITMAP_S;

typedef struct
{
	UInt32 Motion_State;

} VISP_STAT_OTHER_S;

typedef struct
{
	VISP_STAT_AE_S stAeData;
	VISP_STAT_AWB_S stAwbData;
	VISP_STAT_AF_S stAfData;
	VISP_STAT_MDB_BITMAP_S stMdbData;
	VISP_STAT_OTHER_S stOtherData;
} VISP_STAT_S;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_ISP_H__

