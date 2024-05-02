
/*
 * Filename    : npu_tmp_osd.h
 * Description : EN677 temporary OSD control function
 * Author      : kimms@eyenix.com
 * Date        : 2023. 03. 15
 * NOTICE      : This file is a temporary OSD control API library written before the development 
 *               of EN677 MediaLink. It should not be used after the completion of MediaLink 
 *               configuration.
 */

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>

#include "osd_eyenix.h"

using namespace std;

static int memfd;
static void *OSDBuf = NULL;

uint8_t box_colors[100][3] = {
    {160, 0, 0},
    {42, 25, 75},
    {239, 120, 192},
    {33, 244, 128},
    {251, 220, 138},
    {31, 232, 79},
    {160, 197, 122},
    {35, 32, 66},
    {28, 177, 185},
    {107, 185, 161},
    {179, 154, 95},
    {118, 4, 50},
    {229, 210, 154},
    {193, 161, 39},
    {23, 85, 212},
    {59, 6, 13},
    {36, 217, 100},
    {152, 36, 120},
    {29, 16, 99},
    {120, 208, 156},
    {201, 189, 239},
    {221, 178, 52},
    {29, 88, 88},
    {22, 248, 30},
    {252, 197, 115},
    {230, 24, 195},
    {55, 50, 132},
    {188, 70, 4},
    {126, 133, 194},
    {89, 205, 182},
    {221, 146, 81},
    {10, 85, 61},
    {39, 255, 128},
    {160, 232, 81},
    {249, 224, 21},
    {211, 224, 119},
    {126, 77, 174},
    {97, 190, 30},
    {115, 140, 39},
    {233, 0, 192},
    {209, 0, 128},
    {62, 149, 105},
    {234, 89, 197},
    {235, 103, 137},
    {148, 111, 12},
    {196, 75, 92},
    {108, 186, 191},
    {244, 44, 150},
    {227, 66, 41},
    {152, 216, 79},
    {69, 231, 165},
    {235, 174, 101},
    {74, 160, 95},
    {80, 45, 68},
    {34, 44, 252},
    {157, 191, 44},
    {160, 147, 235},
    {221, 152, 134},
    {60, 72, 133},
    {160, 24, 62},
    {214, 122, 167},
    {206, 37, 141},
    {202, 255, 0},
    {120, 255, 255},
    {119, 133, 31},
    {188, 100, 95},
    {201, 18, 249},
    {167, 192, 0},
    {25, 181, 130},
    {32, 155, 75},
    {160, 55, 116},
    {162, 140, 236},
    {198, 215, 70},
    {126, 128, 0},
    {96, 90, 131},
    {57, 139, 128},
    {124, 221, 152},
    {111, 78, 142},
    {186, 14, 188},
    {200, 195, 247},
    {96, 11, 47},
    {254, 249, 213},
    {82, 242, 28},
    {188, 179, 37},
    {160, 138, 221},
    {115, 60, 161},
    {29, 58, 166},
    {212, 53, 224},
    {243, 32, 128},
    {160, 170, 85},
    {35, 92, 60},
    {227, 200, 71},
    {168, 50, 43},
    {209, 132, 65},
    {224, 201, 175},
    {185, 151, 30},
    {145, 129, 7},
    {15, 218, 112},
    {184, 163, 79},
    {69, 78, 1},
};

void *InitOSDMapping()
{
    memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if(memfd < 0){
        printf("mem open error\n");
        return nullptr;
    }
    OSDBuf = mmap(
        0,						// addr
        OSD_MMAP_SIZE,			// len
        PROT_READ|PROT_WRITE,	// prot
        MAP_SHARED,				// flags
        memfd,					// fd
        _REG_BASE_		    // offset
    );
    if(OSDBuf == MAP_FAILED){
        perror("mmap");
        return NULL;
    }
    return OSDBuf;
};

void DeinitOSDMapping()
{
    munmap(OSDBuf, OSD_MMAP_SIZE);
    close(memfd);
};

void InitOSD(){

    InitOSDMapping();

    // BOX Init
    wOSD(OSDBuf, BOX_OFFSET, 0x00130003); // trial & error value
    BOX_COLOR_S stdColor;   // green
    stdColor.Y = 133;
    stdColor.Cb = 63;
    stdColor.Cr = 51;
    for(int i=0;i<MAX_BOX;++i){
        setBoxColor(i,&stdColor);
    }
    DeleteAllBoxes();

    // Font Init
    wOSD(OSDBuf, FONT_ONOFF, 0xC006000C); // trial & error value
    wOSD(OSDBuf, FONT_COL_P0, 0x00F08080); // preset : white
    wOSD(OSDBuf, FONT_COL_P1, 0x00853F33); // preset : green
    wOSD(OSDBuf, FONT_COL_P2, 0x0091932C); // preset : yellow
    wOSD(OSDBuf, FONT_COL_P3, 0x001BD478); // preset : red
    wOSD(OSDBuf, FONT_SIZE, 0x00888011); // min Vsize, min Hsize, trial & error value

    // Font space Init
    setFontLR();
    FONT_S stdFont;
    stdFont.Char = INIT_CHAR;
    stdFont.Attr = 0;       // preset 0 color
    stdFont.Alpha = 0;      // 100%
    stdFont.enAlpha = 1;
    stdFont.enAttr = 1;
    stdFont.enChar = 1;
    for(int i = 0; i < MAX_LINE; ++i){
        stdFont.Char = INIT_CHAR;
        for(int j=0; j < MAX_CHAR; ++j){
            setFont(j, i, stdFont);
        }
    }

    ClearFont();
};

void DeinitOSD(){
    DeleteAllBoxes();
    ClearFont();
    DeinitOSDMapping();
};

void DrawAllBoxes(){
    wOSD(OSDBuf, BOSD_ON0, 0xFFFFFFFF);
};

void DeleteAllBoxes(){
    wOSD(OSDBuf, BOSD_ON0, 0x00000000);
};

void DrawBox(int idx){
    if(idx < 0 || idx > (MAX_BOX - 1)) return;
    wOSD(OSDBuf, BOSD_ON0, (0x1 << idx) | rOSD(OSDBuf, BOSD_ON0));
};

void DrawNBox(int n){
    if(n < 0 || n > (MAX_BOX)) return;
    if(n == 0) return DeleteAllBoxes(); // BECAUSE result of 32bit shift is not 0 !!!
    wOSD(OSDBuf, BOSD_ON0, 0xFFFFFFFF >> (32 - n));
};

void DeleteBox(int idx){
    if(idx < 0 || idx > (MAX_BOX - 1)) return;
    wOSD(OSDBuf, BOSD_ON0, ~(0x1 << idx) & rOSD(OSDBuf, BOSD_ON0));
};

void setBoxPos(int idx, BOX_POS_S* pos){
    if(idx < 0 || idx > (MAX_BOX - 1)) return;

    CLIP(0, MAX_HPOS, pos->x_min);
    CLIP(0, MAX_HPOS, pos->x_max);
    CLIP(0, MAX_VPOS, pos->y_min);
    CLIP(0, MAX_VPOS, pos->y_max);

    uint32_t HspHep = (pos->x_min << 16) | pos->x_max;
    uint32_t VspVep = (pos->y_min << 16) | pos->y_max;

    wOSD(OSDBuf, BOX0_COORD + idx * 2, VspVep);
    wOSD(OSDBuf, BOX0_COORD + idx * 2 + 1, HspHep);
};

void setBoxColor(int idx, BOX_COLOR_S* color){
    if(idx < 0 || idx > (MAX_BOX - 1)) return;

    CLIP(0, MAX_INTESITY, color->Y);
    CLIP(0, MAX_INTESITY, color->Cb);
    CLIP(0, MAX_INTESITY, color->Cr);

    uint32_t YCbCr = (color->Y << 16) | (color->Cb << 8) | color->Cr;

    wOSD(OSDBuf, BOX0_COLOR + idx, YCbCr);
}

void setBoxColorLabel(int idx, unsigned int label){
    if(idx < 0 || idx > (MAX_BOX - 1)) return;

    uint8_t box_y = box_colors[label][0];
    uint8_t box_cb = box_colors[label][1];
    uint8_t box_cr = box_colors[label][2];

    uint32_t YCbCr = (box_y << 16) | (box_cb << 8) | box_cr;

    wOSD(OSDBuf, BOX0_COLOR + idx, YCbCr);
}

void setFont(int X, int Y, FONT_S C){
    CLIP(0, MAX_CHAR - 1, X);
    CLIP(0, MAX_LINE - 1, Y);

    uint32_t Font = (C.enChar << 31) | (C.enAttr << 30) | (C.enAlpha << 29)
        | (C.Alpha << 11) | (C.Attr << 9) | C.Char;

    wOSD(OSDBuf, FONT_BASE + (MAX_CHAR + 1) * Y + X, Font);
}

// Setting Line return & Page return
void setFontLR(){
    FONT_S C;
    C.Char = ISP_FONT_LINE_RET;
    C.enAlpha = 0;
    C.enAttr = 0;
    C.enChar = 1;

    uint32_t Font = (C.enChar << 31) | (C.enAttr << 30) | (C.enAlpha << 29) | C.Char;
    for(int i=0;i<MAX_LINE;++i){
        wOSD(OSDBuf, FONT_BASE + i * (MAX_CHAR + 1) + MAX_CHAR, Font);
    }

    // page return
    C.Char = ISP_FONT_PAGE_RET;
    Font = (C.enChar << 31) | (C.enAttr << 30) | (C.enAlpha << 29) | C.Char;
    wOSD(OSDBuf, FONT_BASE + (MAX_CHAR + 1) * MAX_LINE, Font);
}

void ClearFont(){
    FONT_S C;
    C.Char = ' ';
    C.enAlpha = 0;
    C.enAttr = 0;
    C.enChar = 1;

    for(int i = 0; i < MAX_LINE; ++i){
        for(int j=0; j < MAX_CHAR; ++j){
            setFont(j, i, C);
        }
    }
}

void setString(int X, int Y, const char* str){
    CLIP(0, MAX_LINE - 1, Y);
    CLIP(0, MAX_CHAR - (int)strlen(str), X);
    
    FONT_S C;
    C.enAlpha = 0;
    C.enAttr = 0;
    C.enChar = 1;

    for(int i=0; (i < (int)strlen(str)) && (*(str+i)); i++) {
        C.Char = *(str+i);
        setFont(X + i, Y, C);
	}
    
};

void memcpyOSD(void *src, int size)
{
    memcpy(OSDBuf, src, size);
}

void EyenixOSD(vector< BoundingBox > &Result, vector<std::string> &classNames, int height, int width)
{
    // font clear for erase all old results
    ClearFont();

    DrawNBox((int)Result.size());

    for(int i=0;i<(int)Result.size();i++)
    {
        // set box colors according to labels
        setBoxColorLabel(i, Result[i].label);

        BOX_POS_S pos;
        pos.x_min = int(Result[i].box[0] * 1920 / width); //xmin
        pos.y_min = int(Result[i].box[1] * 1080 / height); //ymin
        pos.x_max = int(Result[i].box[2] * 1920 / width); //xmax
        pos.y_max = int(Result[i].box[3] * 1080 / height); //ymax
        setBoxPos(i, &pos);

        char cstr[25];
        sprintf(cstr,"%s(%02d%%)", classNames[Result[i].label].c_str(), int(100 * Result[i].score));

        int strXp = int((Result[i].box[0] + Result[i].box[2]) * MAX_CHAR / width / 2);
        int strYp = int((Result[i].box[1] + Result[i].box[3]) * MAX_LINE / height / 2);

        setString(strXp - strlen(cstr)/2, strYp, cstr);
    }    
}
void EyenixOSD_setString(int X, int Y, const char* str){
    CLIP(0, MAX_LINE - 1, Y);
    CLIP(0, MAX_CHAR - (int)strlen(str), X);
    
    FONT_S C;
    C.enAlpha = 0;
    C.enAttr = 0;
    C.enChar = 1;

    for(int i=0; (i < (int)strlen(str)) && (*(str+i)); i++) {
        C.Char = *(str+i);
        setFont(X + i, Y, C);
	}
    
};