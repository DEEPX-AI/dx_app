
/*
 * Filename    : npu_tmp_osd.h
 * Description : EN677 temporary OSD control function
 * Author      : kimms@eyenix.com
 * Date        : 2023. 03. 15
 * NOTICE      : This file is a temporary OSD control API library written before the development 
 *               of EN677 MediaLink. It should not be used after the completion of MediaLink 
 *               configuration.
 */

#ifndef __NPU_TMP_OSD_H__
#define __NPU_TMP_OSD_H__

#include "bbox.h"

#define _REG_BASE_              (0x42100000)

/*
//#define OSD_MUTE0r            (ValSft_R31(_rd32(_REG_BASE_+(0x01a0<<2)))&BitMask_01)
//#define OSD_ISEL0r            (ValSft_R24(_rd32(_REG_BASE_+(0x01a0<<2)))&BitMask_04)
//#define OSD_IVSEL0r           (ValSft_R20(_rd32(_REG_BASE_+(0x01a0<<2)))&BitMask_04)
#define BTONE31r                (ValSft_R30(_rd32(_REG_BASE_+(0x01a3<<2)))&BitMask_02)
//#define BTONE16r              (ValSft_R00(_rd32(_REG_BASE_+(0x01a3<<2)))&BitMask_02)
//#define BTONE15r              (ValSft_R30(_rd32(_REG_BASE_+(0x01a4<<2)))&BitMask_02)
//#define BTONE0r               (ValSft_R00(_rd32(_REG_BASE_+(0x01a4<<2)))&BitMask_02)
#define BFL_ON31r               (ValSft_R31(_rd32(_REG_BASE_+(0x01a5<<2)))&BitMask_01)
//#define BFL_ON0r              (ValSft_R00(_rd32(_REG_BASE_+(0x01a5<<2)))&BitMask_01)
#define BITBOX0r                (ValSft_R00(_rd32(_REG_BASE_+(0x0206<<2)))&BitMask_32)
#define BIT_CB0r                (ValSft_R16(_rd32(_REG_BASE_+(0x0207<<2)))&BitMask_08)
//#define BIT_CR0r              (ValSft_R08(_rd32(_REG_BASE_+(0x0207<<2)))&BitMask_08)
//#define BITMAP_THRES0r        (ValSft_R00(_rd32(_REG_BASE_+(0x0207<<2)))&BitMask_08)
*/

// using mmap with 1 page, so you can access 0x4210_0000 ~ 0x4210_7FFF
#define OSD_MMAP_SIZE           (4096 * 8)

#define BOX_OFFSET              (0x01a1)
#define BOSD_ON0                (0x01a2)
#define BTONE31                 (0x01a3)
#define BFL_ON31                (0x01a5)
#define BOX0_COORD              (0x01a6)
#define BOX0_COLOR              (0x01e6)
#define BITBOX0                 (0x0206)
#define BIT_CB0                 (0x0207)

#define FONT_ONOFF              (0x0196)    // on | boad on | boad mode | mute | offY | wmod | offX
#define FONT_COL_P0             (0x0197)    // font color preset 0
#define FONT_COL_P1             (0x0198)    // font color preset 1
#define FONT_COL_P2             (0x0199)    // font color preset 2
#define FONT_COL_P3             (0x019A)    // font color preset 3
#define FONT_SIZE               (0x019C)    // Vblank | Hblank | Vsize | Hsize
#define FONT_BASE               (0x1000)    // Font space 0x1000 ~ 0x1FFF (0x4000 ~ 0x7FFC)

#define rOSD(addr, offset)      (*((volatile uint32_t*)(addr) + (offset)))
#define wOSD(addr, offset, val) {*((volatile uint32_t*)(addr) + (offset)) = (val);}

#define CLIP(min, max, val)     {(val) = (val) >= (min) ? (val) <= (max) ? (val) : (max) : (min);}

#define MAX_BOX                 (32)
#define MAX_HPOS                (0x1FFF)
#define MAX_VPOS                (0x0FFF)
#define MAX_INTESITY            (0xFF)
#define MAX_CHAR                (119)       // For 2M
#define MAX_LINE                (34)        // For 2M
#if ((MAX_CHAR+1) * MAX_LINE > 4096)
    #error "MAX FONT is 4096"
#endif

#define ISP_FONT_LINE_RET		(339)
#define ISP_FONT_PAGE_RET		(340)
#define INIT_CHAR               (' ')       //  ('A') 

typedef struct {
    uint32_t x_max      :   16; // [15:0]    
    uint32_t x_min      :   16; // [31:16]    
    uint32_t y_max      :   16; // [47:32]    
    uint32_t y_min      :   16; // [63:48]    
} BOX_POS_S;

typedef struct {
    uint32_t Cr         :   8;  // [7:0]
    uint32_t Cb         :   8;  // [15:8]
    uint32_t Y          :   8;  // [23:16]
} BOX_COLOR_S;

typedef struct {
    uint32_t Char       :   9;  // [8:0] ascii code + special char + control
    uint32_t Attr       :   2;  // [10:9] color P0 : 00, P1 : 01, P2 : 10, P3 : 11
    uint32_t Alpha      :   2;  // [12:11] 100% : 00, 75% : 01, 50% : 10, 25% : 11 
    uint32_t enAlpha    :   1;  // [29:29]
    uint32_t enAttr     :   1;  // [30:30]
    uint32_t enChar     :   1;  // [31:31]
} FONT_S;

void *InitOSDMapping();
void DeinitOSDMapping();

void InitOSD();
void DeinitOSD();

void DrawAllBoxes();
void DeleteAllBoxes();

void DrawBox(int idx);
void DrawNBox(int n);
void DeleteBox(int idx);

void setBoxPos(int idx, BOX_POS_S* pos);

void setBoxColor(int idx, BOX_COLOR_S* color);
void setBoxColorLabel(int idx, unsigned int label);

void setFont(int X, int Y, FONT_S C);
void setFontLR();
void ClearFont();
void setString(int X, int Y, const char* str);
void memcpyOSD(void *src, int size);
void EyenixOSD(std::vector< BoundingBox > &Result, std::vector<std::string> &classNames, int height, int width);
void EyenixOSD_setString(int X, int Y, const char* str);
#endif