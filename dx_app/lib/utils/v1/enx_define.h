
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_DEFINE_H__
#define __ENX_DEFINE_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#define MAX_VIDEO_CH	5
#define MAX_FILE_PATH 256
#define MAX_PACKET_NUM 30

#define MAX_ECMM_CNT	10
#define MAX_ECMM_SIZE	16*1024*1024//20*1024*1024	// byte
#define DEF_VENC_CH_BUFF 4*1024*1024
#define DEF_VDEC_CH_BUFF 4*1024*1024

#define MAX_MDB_REG		268
#define MAX_AED_REG		6
#define MAX_AFD_REG		16
#define MAX_VCAP_CH		3
#define MAX_NPU_CH		2
#define MAX_VDEC_CH		4
#define MAX_VISP_PAR	232
#define MAX_VISP_BOX	32

#define TICK_COUNT_PER_SEC		90000
#define DEF_SEC_UNIT	1000000
//#define DEF_PTS_UNIT	3000

#define MAX_VISP_PVC_NUM	16
#define MAX_VISP_IMD_NUM	4
#define MAX_VISP_YC_NUM		8
#define MAX_VISP_YC_PAGE	3
#define MAX_AUD_FRAME_NUM	100

#define MAX_DUMP_PATH_LEN	128

#ifndef KB
#define KB ((UINT)1024)
#endif

#ifndef MB
#define MB (KB*KB)
#endif

#define ENX_TIMEOUT_NONE        ((UINT) 0)  // no timeout
#define ENX_TIMEOUT_FOREVER     ((UINT)-1)  // wait forever

#define ENX_memAlloc(size)      (void*)malloc((size))
#define ENX_memFree(ptr)        free(ptr)

#define ENX_align(value, align)   ((( (value) + ( (align) - 1 ) ) / (align) ) * (align) )

#define ENX_floor(value, align)   (( (value) / (align) ) * (align) )
#define ENX_ceil(value, align)    ENX_align(value, align)



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif	// __ENX_DEFINE_H__
