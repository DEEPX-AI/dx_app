
/*
 * Filename    : npu_tmp_osd.h
 * Description : EN677 temporary OSD control function
 * Author      : kimms@eyenix.com
 * Date        : 2023. 03. 15
 * NOTICE      : This file is a temporary OSD control API library written before the development 
 *               of EN677 MediaLink. It should not be used after the completion of MediaLink 
 *               configuration.
 */

#ifndef __ISP_EYENIX_H__
#define __ISP_EYENIX_H__

// #include "bbox.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>

#include "enx_define.h"
#include "enx_errno.h"
#include "enx_isp.h"
#include "enx_link.h"
#include "dbg_prn.h"
#include "enx_types.h"
#include "enx_util_thr.h"
#include "enx_uyv.h"
#include "enx_vsys.h"
#include "enx_visp.h"

// #define DEMO

#ifdef DEMO
#include "../../../demos/object_detection/include/yolo.h"
#endif

//////////////////////////////////////// Util Part /////////////////////////////////////////////////
typedef enum {
	WAIT_FRAME,
    PRE_PROCESS,
	NPU_INFERENCE,
	POST_PROCESS,
	OSD_PRINT,
    TOTAL_INTERVAL,
	NUM_TimeType
} TIME_TYPE_E;

extern const char* TimeTypeString[NUM_TimeType];
extern uint64_t nputime[NUM_TimeType];

#define NPUTIMEMEMSIZE (32)
extern uint64_t nputimeMem[NPUTIMEMEMSIZE];


uint64_t GetAvgTime(uint64_t newTime);
void DumpBinary(const char* path, size_t size, void* data);

//////////////////////////////////////// Get ISP Frame Part ////////////////////////////////////////

struct npu_dma_ioctl_data_copy
{
    uint32_t input_addr;
    uint32_t uyv_addr;
    uint32_t input_size;
};

typedef struct {
	BOOL bThrRunning;
	Int32 nCh;
    UYV_DATA_S* pUyvData;
} UYV_THR_PARAM_S;

typedef struct {
    // for streamer
	int measure;
    UYV_DATA_S* pUyvData;
} NPU_THR_PARAM_S;

void SetUYVResolution(unsigned uyvWidth, unsigned uyvHeight);


//////////////////////////////////////// OSD Part //////////////////////////////////////////////////

#define _REG_BASE_              (0x42100000)

// using mmap with 8 pages, so you can access 0x4210_0000 ~ 0x4210_7FFF
#define OSD_MMAP_SIZE           (4096 * 8)

#define FRAME_IRQ               (0x06CE)
#define FRAME_IRQ_CLR           (0x03B9)

#define BOX_MUTE                (0x01A0)
#define BOX_OFFSET              (0x01A1)
#define BOSD_ON0                (0x01A2)
#define BOX0_MOD_VAL_H          (0x01A3)
#define BOX0_MOD_VAL_L          (0x01A4)
#define BOX0_FILL               (0x01A5)
#define BOX0_POS                (0x01A6)
#define BOX0_COLOR              (0x01E6)
#define BOX0_T_P_MOD            (0x0206)
#define BIT_CB0                 (0x0207)

#define BOSD_ON1                (0x0490)
#define BOX1_T_P_MOD            (0x0491)
#define BOX1_FILL               (0x0492)
#define BOX1_MOD_VAL_H          (0x0493)
#define BOX1_MOD_VAL_L          (0x0494)
#define BOX1_COLOR              (0x0495)
#define BOX1_POS                (0x04B6)

#define FONT_ONOFF              (0x0196)    // on | boad on | boad mode | mute | offY | wmod | offX
#define FONT_COL_P0             (0x0197)    // font color preset 0
#define FONT_COL_P1             (0x0198)    // font color preset 1
#define FONT_COL_P2             (0x0199)    // font color preset 2
#define FONT_COL_P3             (0x019A)    // font color preset 3
#define FONT_SIZE               (0x019C)    // Vblank | Hblank | Vsize | Hsize
#define FONT_BASE               (0x1000)    // Font space 0x1000 ~ 0x1FFF (0x4000 ~ 0x7FFC)

#define rOSD(addr, offset)      (*((volatile uint32_t*)(addr) + (offset)))
#define wOSD(addr, offset, val) do{*((volatile uint32_t*)(addr) + (offset)) = (val);}while(0)

#define CLIP(min, max, val)     {(val) = (val) >= (min) ? (val) <= (max) ? (val) : (max) : (min);}

#define MAX_BOX0                (32)
#define MAX_BOX1                (32)
#define MAX_BOX                 (MAX_BOX0 + MAX_BOX1)
#define MAX_HPOS                (0x1FFF)
#define MAX_VPOS                (0x0FFF)
#define MAX_INTESITY            (0xFF)
#define MAX_CHAR                (119)       // For 2M
#define MAX_LINE                (34)        // For 2M
#if ((MAX_CHAR+1) * MAX_LINE > 4096)
    #error "MAX FONT is 4096"
#endif

#define BOX_OFFSET_2M           (0x00130003) // 0x13, 0x3

#define ISP_FONT_LINE_RET		(339)
#define ISP_FONT_PAGE_RET		(340)
#define INIT_CHAR               (' ')       //  ('A') 

#define SCREEN_WIDTH        (1920)
#define SCREEN_HEIGHT       (1080)

typedef struct {
    bool        BoxOn       ;
    uint16_t    x_min   :13 ;
    uint16_t    y_min   :12 ;
    uint16_t    x_max   :13 ;
    uint16_t    y_max   :12 ;
} BOX_POS_S;

typedef struct {
    uint8_t     Mode    :1  ;   // 0 : Tone, 1 : Pattern
    bool        Fill        ;   // 0 : Line, 1 : Filled
    uint8_t     modVal  :2  ;   // 0 : (100%, Bit), 1 : (50%, BW), 2 : (25%, NEG), 3 : (0%, invBit)
    uint8_t     ColorY      ;   // 
    uint8_t     ColorCb     ;   // 
    uint8_t     ColorCr     ;   // 
} BOX_ATTR_S;

typedef struct {
    bool        enChar      ;  // 
    bool        enAttr      ;  // 
    bool        enAlpha     ;  // 
    uint8_t     Alpha   :2  ;  // 0 : 100%, 1 : 50%, 2 : 25%, 3 : 0%
    uint8_t     Attr    :2  ;  // color P0 ~ P3
    uint16_t    Char    :9  ;  // ascii code + special char + control
} FONT_S;

extern BOX_POS_S BOXBUF[MAX_BOX];
extern BOX_ATTR_S BOXATTRBUF[MAX_BOX];
extern FONT_S FONTBUF[MAX_LINE][MAX_CHAR + 1];

void *InitOSDMapping();
void DeinitOSDMapping();

void InitOSD();
void DeinitOSD();

void setFontLR();
void ClearFontBuf();
void ClearBoxBuf();
void setString(int X, int Y, const char* str);

void UpdateOSD();
#ifdef DEMO
void SendResultOSD(uint64_t* NPUTimeStamp, string usrStr, YoloParam cfg, vector< BoundingBox > Result);
#endif
void WaitNewFrame();
void ClearFrameFlag();
#endif
