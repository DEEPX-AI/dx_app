
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

#include "l1/isp_eyenix.h"

using namespace std;


//////////////////////////////////////// Util Part /////////////////////////////////////////////////

const char* TimeTypeString[NUM_TimeType] = {
	"Wait FRAME      ",
    "Pre Process     ",
	"NPU Inference   ",
	"Post Process    ",
	"OSD API         ",
    "Total interval  "
};

uint64_t nputime[NUM_TimeType];
uint64_t nputimeMem[NPUTIMEMEMSIZE] = {0};
unsigned char nputimeMemIdx = 0;


uint64_t GetAvgTime(uint64_t newTime){
    nputimeMem[((nputimeMemIdx++) % NPUTIMEMEMSIZE)] = newTime;

    uint64_t sum = 0;
    for(int i=0; i< NPUTIMEMEMSIZE; ++i){
        sum += nputimeMem[i]; 
    }


    return (sum / NPUTIMEMEMSIZE);
};


void DumpBinary(const char* path, size_t size, void* data){
    FILE* outfp = fopen(path,"wb");
    if(outfp == NULL) {
        printf("Can't write file. (%s)",path);
        return;
    }

    int ret = fwrite(data,size,1,outfp);

    if(size != ret){
        printf("There is problem in fwrite. (%d)",ret);
    }
    fclose(outfp);
    return;
}


//////////////////////////////////////// Get ISP Frame Part ////////////////////////////////////////
static int memfd;
static unsigned char *ispBuf = NULL;

void SetUYVResolution(unsigned uyvWidth, unsigned uyvHeight){
	UYV_CH_ATTR_S stUYV;
	stUYV.chId = 0;
	stUYV.width = uyvWidth;
	stUYV.height = uyvHeight;
	stUYV.fps = 30;

	ENX_UYV_CAPTURE_CH_SetAttr(0,&stUYV);

	VISP_PARAM_UNIT_S stPar;
	strcpy(stPar.ID,"UYSH");
    ENX_VISP_GetPar(&stPar, 1);
    int UYVH = stPar.VAL;
	strcpy(stPar.ID,"UYSV");
    ENX_VISP_GetPar(&stPar, 1);
    int UYVV = stPar.VAL;

	printf("UYV Resolution (W, H) : (%d, %d)\n", UYVH, UYVV); 
};

extern int uyv_cnt;
extern pthread_mutex_t lock_uyv;

void *get_uyv_thread(void *pParam)
{
	Int32 ret = 0;
	UYV_DATA_S *pUyvData = NULL;
	struct timeval timeout;
	fd_set readFds;
	Int32 uyvFds[MAX_NPU_CH];
	Int32 maxFd = 0;
	Int32 nCh = 0;

	UYV_THR_PARAM_S *pThrParam = NULL;
	pThrParam = (UYV_THR_PARAM_S *)pParam;

	pUyvData = pThrParam->pUyvData;

	nCh = pThrParam->nCh;

	FD_ZERO(&readFds);
	
	printf("%s start ch=%d\n", __func__, nCh);

	while(pThrParam->bThrRunning == TRUE)
	{
		FD_ZERO(&readFds);

		while((uyvFds[nCh] = ENX_UYV_CAPTURE_CH_GetFd(nCh)) < 0) {
			printf("wait stream from video .. \n");
			usleep(100);
			continue;
		}

		if (maxFd < uyvFds[nCh]) {
			maxFd = uyvFds[nCh];
		}

		FD_SET(uyvFds[nCh], &readFds);

		timeout.tv_sec  = 0;
		timeout.tv_usec = 100 * 1000; //100ms

		ret = select(maxFd+1, &readFds, NULL, NULL, &timeout);

		if (ret < 0) { 
			printf("Sample Channel select err %d \n", ret);
			break;
		} else if (0 == ret) { 
			printf("Sample Channel select timeout %d \n", ret);
			continue;
		} else { 

			if(uyvFds[nCh] < 0 || uyvFds[nCh] > maxFd) {
				printf("Sample Channel fd set failed uyvFds %d \n", uyvFds[nCh]);
				break;
			}

			if (FD_ISSET(uyvFds[nCh], &readFds)) 
			{ 
				pthread_mutex_lock(&lock_uyv);
				ret = ENX_UYV_CAPTURE_CH_GetFrame(nCh, pUyvData);	// Lock
				if(ret != 0) {
					printf("ENX_VENC_CH_GetFrame failed\n");
					pthread_mutex_unlock(&lock_uyv);
					continue;
				}
				uyv_cnt++;
				pthread_mutex_unlock(&lock_uyv);
			} // FD_ISSET
		} // select ret
	} // while

	if(pUyvData != NULL) {
		free(pUyvData);
		pUyvData = NULL;
	}

	pThrParam->bThrRunning = FALSE;
	printf("**************** UYV THREAD EXIT ******************\n");

	UNUSED(pParam);
	return (void *)0;
}

//////////////////////////////////////// OSD Part //////////////////////////////////////////////////
static void *OSDBuf = NULL;
static int wait_cnt = 0;
BOX_POS_S BOXBUF[MAX_BOX];
BOX_ATTR_S BOXATTRBUF[MAX_BOX];
FONT_S FONTBUF[MAX_LINE][MAX_CHAR + 1];

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
    // Mapping Physical Memory
    InitOSDMapping();

    wOSD(OSDBuf, BOX_OFFSET, 0x00130003); // trial & error value

    // Font Init
    wOSD(OSDBuf, FONT_ONOFF, 0xC006000C); // trial & error value
    wOSD(OSDBuf, FONT_COL_P0, 0x00F08080); // preset : white    // 00 | Y | Cb | Cr
    //wOSD(OSDBuf, FONT_COL_P1, 0x00F08080); // preset : white    // 00 | Y | Cb | Cr
    wOSD(OSDBuf, FONT_COL_P1, 0x00853F33); // preset : green    // 00 | Y | Cb | Cr
    wOSD(OSDBuf, FONT_COL_P2, 0x0091932C); // preset : yellow   // 00 | Y | Cb | Cr
    wOSD(OSDBuf, FONT_COL_P3, 0x001BD478); // preset : red      // 00 | Y | Cb | Cr
#if 1 //old
    wOSD(OSDBuf, FONT_SIZE, 0x00888011); // min Vsize, min Hsize, trial & error value
#else
    wOSD(OSDBuf, FONT_SIZE, 0x00888022); // min Vsize, min Hsize, trial & error value
#endif
    // OSD buffer clear
    memset(FONTBUF,0,sizeof(BOXBUF));
    memset(FONTBUF,0,sizeof(FONTBUF));

    // Font space Init
    setFontLR();
    // Clear font
    ClearFontBuf();
};

void DeinitOSD(){
    ClearBoxBuf();
    ClearFontBuf();
    UpdateOSD();
    DeinitOSDMapping();
};

void ClearFontBuf(){
    FONT_S initSpace;
    initSpace.Char = INIT_CHAR;
    initSpace.Attr = 0;
    initSpace.Alpha = 0;
    initSpace.enAlpha = 1;
    initSpace.enAttr = 1;
    initSpace.enChar = 1;

    for(int i = 0; i < MAX_LINE; ++i){
        for(int j = 0; j < MAX_CHAR; ++j){
            FONTBUF[i][j] = initSpace;
        }
    }
};

void writeBoxPos(int idx, BOX_POS_S pos){
    // skip index check and range check
    /* 
    if(idx < 0 || idx > (MAX_BOX - 1)){
        return;
    }

    CLIP(0, MAX_HPOS, pos.x_min);
    CLIP(0, MAX_HPOS, pos.x_max);
    CLIP(0, MAX_VPOS, pos.y_min);
    CLIP(0, MAX_VPOS, pos.y_max);
    */

    uint32_t HspHep = (pos.x_min << 16) | pos.x_max;
    uint32_t VspVep = (pos.y_min << 16) | pos.y_max;

    if(idx < MAX_BOX0){
        wOSD(OSDBuf, BOX0_POS + idx * 2, VspVep);
        wOSD(OSDBuf, BOX0_POS + idx * 2 + 1, HspHep);
    }else{
        wOSD(OSDBuf, BOX1_POS + (idx - MAX_BOX0) * 2, VspVep);
        wOSD(OSDBuf, BOX1_POS + (idx - MAX_BOX0) * 2 + 1, HspHep);
    }
    
};

void writeBoxColor(int idx, BOX_ATTR_S attr){
    // skip index check
    /*
    if(idx < 0 || idx > (MAX_BOX - 1)) return;
    */

    uint32_t YCbCr = (attr.ColorY << 16) | (attr.ColorCb << 8) | (attr.ColorCr << 0);

    // (color of 46th box), (APIBoxNum[31:0]) is overlaped
    if(idx == 46){
        YCbCr &= 0xFFFF81FF; // bit[14:9] reset
        YCbCr |= 0x00004000; // APIBoxNUM = 32 (0x20)
    }

    if(idx < MAX_BOX0){
        wOSD(OSDBuf, BOX0_COLOR + idx, YCbCr);
    }
    else{
        wOSD(OSDBuf, BOX1_COLOR + (idx - MAX_BOX0), YCbCr);
    }

};

void ClearBoxBuf(){
    for(int i=0;i<MAX_BOX;++i){
        BOXBUF[i].BoxOn = false;
    }
};

// Setting Line return & Page return
void setFontLR(){
    FONT_S LR;
    LR.Char = ISP_FONT_LINE_RET;
    LR.Attr = 0;
    LR.Alpha = 0;
    LR.enAlpha = 1;
    LR.enAttr = 1;
    LR.enChar = 1;

    for(int i=0;i<MAX_LINE;++i){
        FONTBUF[i][MAX_CHAR] = LR;
    }

    // page return
    FONT_S PR = LR;
    PR.Char = ISP_FONT_PAGE_RET;
    FONTBUF[MAX_LINE - 1][MAX_CHAR] = PR;
}



void UpdateOSD(){
    // write registor

    uint64_t boxOnFlags = 0;
    uint64_t boxModFlags = 0;
    uint32_t boxFillFlags[2] = {0, 0};
    uint64_t boxModValFlags[2] = {0, 0};

    for(int i=0; i< MAX_BOX; ++i){
        boxOnFlags = (boxOnFlags >> 1) | ((uint64_t)BOXBUF[i].BoxOn << 63);
        boxModFlags = (boxModFlags >> 1) | ((uint64_t)BOXATTRBUF[i].Mode << 63);
        if(i < MAX_BOX0){
            boxFillFlags[0] = (boxFillFlags[0] >> 1) | ((uint32_t)BOXATTRBUF[i].Fill << 31);
            boxModValFlags[0] = (boxModValFlags[0] >> 2) | ((uint64_t)BOXATTRBUF[i].modVal << 62);
        }else{
            // Seriously, I can't believe that BOX1_FIL REG bit order is reverse...
            boxFillFlags[1] = (boxFillFlags[1] << 1) | ((uint32_t)BOXATTRBUF[i].Fill);
            boxModValFlags[1] = (boxModValFlags[1] >> 2) | ((uint64_t)BOXATTRBUF[i].modVal << 62);
        }
    }

    wOSD(OSDBuf, BOSD_ON0, boxOnFlags & 0xFFFFFFFF);
    wOSD(OSDBuf, BOSD_ON1, (boxOnFlags >> 32) & 0xFFFFFFFF);

    wOSD(OSDBuf, BOX0_T_P_MOD, boxModFlags & 0xFFFFFFFF);
    wOSD(OSDBuf, BOX1_T_P_MOD, (boxModFlags >> 32) & 0xFFFFFFFF);

    wOSD(OSDBuf, BOX0_FILL, boxFillFlags[0] & 0xFFFFFFFF);
    wOSD(OSDBuf, BOX1_FILL, boxFillFlags[1] & 0xFFFFFFFF);

    wOSD(OSDBuf, BOX0_MOD_VAL_L, boxModValFlags[0] & 0xFFFFFFFF);
    wOSD(OSDBuf, BOX0_MOD_VAL_H, (boxModValFlags[0] >> 32) & 0xFFFFFFFF);
    wOSD(OSDBuf, BOX1_MOD_VAL_L, boxModValFlags[1] & 0xFFFFFFFF);
    wOSD(OSDBuf, BOX1_MOD_VAL_H, (boxModValFlags[1] >> 32) & 0xFFFFFFFF);

    for(int i = 0; i < MAX_BOX ; ++i){
        writeBoxPos(i, BOXBUF[i]);
        writeBoxColor(i, BOXATTRBUF[i]);
    }

    for(int i = 0; i < MAX_LINE; ++i){
        for(int j = 0; j < MAX_CHAR + 1; ++j){
            uint32_t Font = ((uint32_t)FONTBUF[i][j].enChar << 31)
                | ((uint32_t)FONTBUF[i][j].enAttr << 30)
                | ((uint32_t)FONTBUF[i][j].enAlpha<< 29)
                | ((uint32_t)FONTBUF[i][j].Alpha  << 11)
                | ((uint32_t)FONTBUF[i][j].Attr   << 9)
                | ((uint32_t)FONTBUF[i][j].Char   << 0);

            wOSD(OSDBuf, FONT_BASE + (MAX_CHAR + 1) * i + j, Font);
        }
    }
    

}

void SendResultOSD(uint64_t* NPUTimeStamp, string usrStr, YoloParam cfg,  vector< BoundingBox > Result){
    // NOTICE :
    // Earlier OSD settings have lower priority.
    // later OSD settings have higher priority.
    // So, Fixed OSD must be placed end of this func. 

#define SCREEN_WIDTH        (1920)
#define SCREEN_HEIGHT       (1080)
#define USE_COLOR_CONF_BOX  (0)
#define USE_ALPHA_CONF_BOX  (1)
#define USE_ALPHA_CONF_CHAR (1)

    // ##################################################
    // Store Box Pos, Size
    // ##################################################

    static double pixelScaledWidth = (double)SCREEN_WIDTH / cfg.width;
    static double pixelScaledHeight = (double)SCREEN_HEIGHT / cfg.height;

    int nObjs = (int)Result.size();
    for(int i = 0; i < MAX_BOX; ++i){
        BOXBUF[i].BoxOn = (i < nObjs) ? true                                        :   false  ; //OnOff
        BOXBUF[i].x_min = (i < nObjs) ? int(Result[i].box[0] * pixelScaledWidth)    :   0      ; //xmin
        BOXBUF[i].y_min = (i < nObjs) ? int(Result[i].box[1] * pixelScaledHeight)   :   0      ; //ymin
        BOXBUF[i].x_max = (i < nObjs) ? int(Result[i].box[2] * pixelScaledWidth)    :   0      ; //xmax
        BOXBUF[i].y_max = (i < nObjs) ? int(Result[i].box[3] * pixelScaledHeight)   :   0      ; //ymax
    }

#if 1
    // ##################################################
    // Store Box Attributte
    // ##################################################

#define TH_ALPHA100 (0.70)
#define TH_ALPHA50  (0.40)
#define TH_ALPHA25  (0.25)

#define TH_P_GREEN  (0.85)  // pure green   (149, 43, 21)
#define TH_M_GREEN  (0.55)  // mild green   (202, 85, 74)
#define TH_W_GREEN  (0.25)  // white green  (228, 101, 21)
    for(int i = 0; i < MAX_BOX; ++i){
        if(i >= nObjs) break;
        BOXATTRBUF[i].Mode    = 0;        // Tone
        BOXATTRBUF[i].Fill    = false;    // not filled
#if USE_ALPHA_CONF_BOX
        BOXATTRBUF[i].modVal  = i < nObjs ?
            Result[i].score > TH_ALPHA25 ?
            Result[i].score > TH_ALPHA50 ?
            Result[i].score > TH_ALPHA100 ?
            0 : 1 : 2 : 3 : 0;       
#else
        BOXATTRBUF[i].modVal  = 0;        // tone 100%
#endif
#if USE_COLOR_CONF_BOX
        BOXATTRBUF[i].ColorY  = i < nObjs ?
            Result[i].score > TH_W_GREEN ?
            Result[i].score > TH_M_GREEN ?
            Result[i].score > TH_P_GREEN ?
                149 : 202 : 228 : 242 : 149;
        BOXATTRBUF[i].ColorCb = i < nObjs ?
            Result[i].score > TH_W_GREEN ?
            Result[i].score > TH_M_GREEN ?
            Result[i].score > TH_P_GREEN ?
                43 : 85 : 107 : 117 : 43;
        BOXATTRBUF[i].ColorCr = i < nObjs ?
            Result[i].score > TH_W_GREEN ?
            Result[i].score > TH_M_GREEN ?
            Result[i].score > TH_P_GREEN ?
                21 : 74 : 101 : 115 : 21;
#else
        BOXATTRBUF[i].ColorY  = 0x85;     // 133, green
        BOXATTRBUF[i].ColorCb = 0x3F;     // 63, green
        BOXATTRBUF[i].ColorCr = 0x33;     // 51, green
        unsigned char label_idx = Result[i].label > 99 ? 99 : Result[i].label;


        // extern const unsigned char box_colors[100][3];
        BOXATTRBUF[i].ColorY  = box_colors[label_idx][0];
        BOXATTRBUF[i].ColorCb = box_colors[label_idx][1];
        BOXATTRBUF[i].ColorCr = box_colors[label_idx][2];
#endif
    }
#endif


    // ##################################################
    // Clear Font Buffer
    // ##################################################

    for(int i = 0; i < MAX_LINE; ++i){
        for(int j = 0; j < MAX_CHAR; ++j ){
            FONTBUF[i][j].enChar  = true;   // Change(erase) char
            FONTBUF[i][j].enAttr  = true;   // Change(erase) color
            FONTBUF[i][j].enAlpha = true;   // Change(erase) alpha or 
            FONTBUF[i][j].Alpha   = 0;      // tone 100%
            FONTBUF[i][j].Attr    = 0;      // preset color 0
            FONTBUF[i][j].Char    = ' ';    // space
        }
    }

    // ##################################################
    // Store Font char by char
    // ##################################################


    static double stringScaledWidth = (double)MAX_CHAR / cfg.width;
    static double lineScaledHeight = (double)MAX_LINE / cfg.height;

    for(int i = 0; i < /*MAX_BOX*/nObjs; ++i){
        char cstr[25];
        sprintf(cstr,"%s(%02d%%)",cfg.classNames[Result[i].label].c_str(), int(100 * Result[i].score));

        int strXp = int((Result[i].box[0] + Result[i].box[2]) * stringScaledWidth / 2);
        int strYp = int((Result[i].box[1] + Result[i].box[3]) * lineScaledHeight / 2 - 0.5);

        strXp -= strlen(cstr)/2;

        CLIP(0, MAX_CHAR - strlen(cstr), strXp);
        CLIP(0, MAX_LINE - 1, strYp);

        for(int j=0;j<strlen(cstr);++j){
            FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
            FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
            FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
#if USE_ALPHA_CONF_CHAR
            FONTBUF[strYp][strXp + j].Alpha   =
                Result[i].score > TH_ALPHA25 ?
                Result[i].score > TH_ALPHA50 ?
                Result[i].score > TH_ALPHA100 ?
                0 : 1 : 2 : 3;
#else
            FONTBUF[strYp][strXp + j].Alpha   = 0;          // Not Used          
#endif
            FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Char    = cstr[j];    // space
        }
    }


    // ##################################################
    // Store User input Box (Fixed line, Box)
    // ##################################################

#if USE_COLOR_CONF_BOX | USE_ALPHA_CONF_BOX | USE_ALPHA_CONF_CHAR
    // legend for alpha diffence of OSD
    // NOTICE : 63~ 59 th boxes CANNOT be used for OBJ DET.
    for(int i = 59; i <= 63; ++i){
        BOXBUF[i].BoxOn = true; //OnOff
        BOXBUF[i].x_min = SCREEN_WIDTH - 320;
        BOXBUF[i].y_min = 95 + (i - 59) * 32;
        BOXBUF[i].x_max = SCREEN_WIDTH - 1;
        BOXBUF[i].y_max = 127 + (i - 59) * 32;
    }

    for(int i = 59; i <= 63; ++i){
        BOXATTRBUF[i].Mode    = 0;
        BOXATTRBUF[i].Fill    = true;
        BOXATTRBUF[i].modVal  = (i - 59) /2 ;
        BOXATTRBUF[i].ColorY  = 
            (i - 59) < 5 ?
            (i - 59) < 3 ?
            (i - 59) < 1 ?
                149 : 202 : 228 : 242;
        BOXATTRBUF[i].ColorCb = 
            (i - 59) < 5 ?
            (i - 59) < 3 ?
            (i - 59) < 1 ?
                43 : 85 : 107 : 117;
        BOXATTRBUF[i].ColorCr = 
            (i - 59) < 5 ?
            (i - 59) < 3 ?
            (i - 59) < 1 ?
                21 : 74 : 101 : 115;
    }


#endif
    // ##################################################
    // Store User input String
    // ##################################################

    // Title
    {
        char titlestr[MAX_CHAR];
        sprintf(titlestr,"DEEPX-L1 NPUAPP  : %s", usrStr.c_str());
        int strXp = 1;
        int strYp = 1;

        for(int j=0;j<strlen(titlestr);++j){
            FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
            FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
            FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
            FONTBUF[strYp][strXp + j].Alpha   = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Char    = titlestr[j];    // space
        }
    }

#if PRINT_TOTAL_INFO
    // Time stamp
    for(int i = 0; i < NUM_TimeType; ++i){
#if 0
        if(i == WAIT_FRAME) continue;
#endif
        char timestr[40];
        sprintf(timestr,"%s : %6u us",TimeTypeString[i], NPUTimeStamp[i]);

        int strXp = 1;
        int strYp = 3 + i;

        for(int j=0;j<strlen(timestr);++j){
            FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
            FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
            FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
            FONTBUF[strYp][strXp + j].Alpha   = 0;          // Not Used          
            FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Char    = timestr[j];    // space
        }
    }
#else
    {
        char timestr[40];
#if 1
        sprintf(timestr,"%s : %6u us",TimeTypeString[NPU_INFERENCE], NPUTimeStamp[NPU_INFERENCE]);
#else
        sprintf(timestr,"%s : %6u us",TimeTypeString[NPU_INFERENCE], ie.GetNpuPerf(0));
#endif
        int strXp = 1;
        int strYp = 3;

        for(int j=0;j<strlen(timestr);++j){
            FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
            FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
            FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
            FONTBUF[strYp][strXp + j].Alpha   = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Char    = timestr[j];    // space
        }
    }
#endif

#if USE_COLOR_CONF_BOX | USE_ALPHA_CONF_BOX | USE_ALPHA_CONF_CHAR
    // legend for alpha diffence of OSD
    for(int i = 0; i < 5; ++i){
        char alphastr[25];
        sprintf(alphastr,"%d - %d %% score", 100 - (15 * i) - 1, 100 - 15 * (i + 1));

        int strXp = 100;
        int strYp = 3 + i;

        for(int j=0;j<strlen(alphastr);++j){
            FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
            FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
            FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
            FONTBUF[strYp][strXp + j].Alpha   = i/2;          // Not Used          
            FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Char    = alphastr[j];    // space
        }
    }
    


#endif

    // Amount of Detected Objs
    {
        char NumObjstr[40];
        sprintf(NumObjstr,"Detected Objs    : %6d Objs", nObjs /* (int)Result.size()*/);
        int strXp = 1;
#if PRINT_TOTAL_INFO
        int strYp = 10;
#else
        int strYp = 4;
#endif
        for(int j=0;j<strlen(NumObjstr);++j){
            FONTBUF[strYp][strXp + j].enChar  = true;       // Change char
            FONTBUF[strYp][strXp + j].enAttr  = true;       // Change color
            FONTBUF[strYp][strXp + j].enAlpha = true;       // Change alpha
            FONTBUF[strYp][strXp + j].Alpha   = 0;          // Not Used          
            FONTBUF[strYp][strXp + j].Attr    = 0;          // Not Used
            FONTBUF[strYp][strXp + j].Char    = NumObjstr[j];    // space
        }
    }


    // ##################################################
    // write buf data into reg
    // ##################################################

    UpdateOSD();


}

void WaitNewFrame(){
    while(!rOSD(OSDBuf, FRAME_IRQ)){
        
        wait_cnt++;
        if(wait_cnt == 10000000){
            printf("Can't get New Frame ; %d\n", wait_cnt);
            break;
        }
    }
    wait_cnt = 0;
};

void ClearFrameFlag(){
    wOSD(OSDBuf, FRAME_IRQ_CLR, 1);
};

