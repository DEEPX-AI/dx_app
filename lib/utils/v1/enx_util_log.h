
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_UTIL_DBG_H__
#define __ENX_UTIL_DBG_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */


#include <stdarg.h>
#include "enx_types.h"
#include "dbg_prn.h"

Int32 openlog();
void closelog();
void writelog(Int32 lvl, Int32 type, const char *fmt, ...);

// Color test success ==========================================================================================
// writelog( lvl, 1, ESC_GREEN "[ MSG][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args );
// =============================================================================================================


#define	SYS_PRINTF( lvl, fmt, args... )	\
	do { \
		writelog( lvl, LOG_TYPE_SYS, ESC_WHITE_BOLD "[ SYS][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	MSG_PRINTF( lvl, fmt, args... )	\
	do { \
		writelog( lvl, LOG_TYPE_MSG, ESC_WHITE_BOLD "[ MSG][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	VENC_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_VENC, ESC_WHITE_BOLD "[VENC][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	VISP_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_VISP, ESC_WHITE_BOLD "[VISP][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	ACAP_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_ACAP, ESC_WHITE_BOLD "[ AUD][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	VCAP_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_VCAP, ESC_WHITE_BOLD "[VCAP][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	UYV_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_UYV, ESC_WHITE_BOLD "[ UYV][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	DISP_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_DISP, ESC_WHITE_BOLD "[DISP][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)

#define	VDEC_PRINTF( lvl, fmt, args... ) \
	do { \
		writelog( lvl, LOG_TYPE_VDEC, ESC_WHITE_BOLD "[VDEC][%s:%5d]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, ##args ); \
	} while(0)



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // #define __ENX_UTIL_DBG_H__
