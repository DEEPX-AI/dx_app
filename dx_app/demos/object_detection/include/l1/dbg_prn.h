
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __DBG_PRN_H__
#define __DBG_PRN_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>

#ifndef FOR_RELEASE
	#define FOR_RELEASE
#endif

//
// Defines escape sequence character codes to control color.
//
#define		ESC_BLACK			"\033[0;33m"
#define		ESC_RED				"\033[0;31m"
#define		ESC_GREEN			"\033[0;32m"
#define		ESC_YELLOW			"\033[0;33m"
#define		ESC_BLUE			"\033[0;34m"
#define		ESC_MAGENTA			"\033[0;35m"
#define		ESC_CYAN			"\033[0;36m"
#define		ESC_WHITE			"\033[0;37m"

#define		ESC_BLACK_BOLD		"\033[1;33m"
#define		ESC_RED_BOLD		"\033[1;31m"
#define		ESC_GREEN_BOLD		"\033[1;32m"
#define		ESC_YELLOW_BOLD		"\033[1;33m"
#define		ESC_BLUE_BOLD		"\033[1;34m"
#define		ESC_MAGENTA_BOLD	"\033[1;35m"
#define		ESC_CYAN_BOLD		"\033[1;36m"
#define		ESC_WHITE_BOLD		"\033[1;37m"

#define		ESC_ENDCOLOR		"\033[0m"

#include	<unistd.h>
#include	<sys/syscall.h>	//	for SYS_xxxx definitions
//#include	<sys/types.h>	//	for pid_t

#include "enx_types.h"

Int32 gettid(void);

#define	RED_PRINTF( fmt, args... )		\
     printf( ESC_RED "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	GREEN_PRINTF( fmt, args... )	\
     printf( ESC_GREEN "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	YELLOW_PRINTF( fmt, args... )	\
     printf( ESC_YELLOW "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	BLUE_PRINTF( fmt, args... )		\
     printf( ESC_BLUE "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	MAGENTA_PRINTF( fmt, args... )	\
     printf( ESC_MAGENTA "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	CYAN_PRINTF( fmt, args... )		\
     printf( ESC_CYAN "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	WHITE_PRINTF( fmt, args... )	\
     printf( ESC_WHITE "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

#define	RED_BOLD_PRINTF( fmt, args... )		\
     printf( ESC_RED_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	GREEN_BOLD_PRINTF( fmt, args... )	\
     printf( ESC_GREEN_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	YELLOW_BOLD_PRINTF( fmt, args... )	\
     printf( ESC_YELLOW_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	BLUE_BOLD_PRINTF( fmt, args... )	\
     printf( ESC_BLUE_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	MAGENTA_BOLD_PRINTF( fmt, args... )	\
     printf( ESC_MAGENTA_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	CYAN_BOLD_PRINTF( fmt, args... )	\
     printf( ESC_CYAN_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	WHITE_BOLD_PRINTF( fmt, args... )	\
     printf( ESC_WHITE_BOLD "[%s:%5d P(%d) T(%d)]: " ESC_ENDCOLOR fmt, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

#define	RED_FULL_PRINTF( fmt, args... )		\
     printf( ESC_RED "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	GREEN_FULL_PRINTF( fmt, args... )	\
     printf( ESC_GREEN "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	YELLOW_FULL_PRINTF( fmt, args... )	\
     printf( ESC_YELLOW "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	BLUE_FULL_PRINTF( fmt, args... )		\
     printf( ESC_BLUE "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	MAGENTA_FULL_PRINTF( fmt, args... )	\
     printf( ESC_MAGENTA "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	CYAN_FULL_PRINTF( fmt, args... )		\
     printf( ESC_CYAN "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	WHITE_FULL_PRINTF( fmt, args... )	\
     printf( ESC_WHITE "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

#define	RED_BOLD_FULL_PRINTF( fmt, args... )		\
     printf( ESC_RED_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	GREEN_BOLD_FULL_PRINTF( fmt, args... )	\
     printf( ESC_GREEN_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	YELLOW_BOLD_FULL_PRINTF( fmt, args... )	\
     printf( ESC_YELLOW_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	BLUE_BOLD_FULL_PRINTF( fmt, args... )	\
     printf( ESC_BLUE_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	MAGENTA_BOLD_FULL_PRINTF( fmt, args... )	\
     printf( ESC_MAGENTA_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	CYAN_BOLD_FULL_PRINTF( fmt, args... )	\
     printf( ESC_CYAN_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )
#define	WHITE_BOLD_FULL_PRINTF( fmt, args... )	\
     printf( ESC_WHITE_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

// We always need error output.
#define	EPRINTF( fmt, args... ) printf( ESC_RED_BOLD "[%s:%5d P(%d) T(%d)]: ERROR " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

// for normal log
#define	HPRINTF( fmt, args... ) printf( ESC_WHITE_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

// for warning log
#define	YPRINTF( fmt, args... ) printf( ESC_YELLOW_BOLD "[%s:%5d P(%d) T(%d)]: " fmt ESC_ENDCOLOR, __FUNCTION__, __LINE__, getpid(), gettid(), ##args )

#ifdef FOR_RELEASE
    #define	DPRINTF( fmt, args... )

	#define	RPRINTF( fmt, args... )
    #define	GPRINTF( fmt, args... )
	#define	BPRINTF( fmt, args... )
    #define	MPRINTF( fmt, args... )
    #define	CPRINTF( fmt, args... )
    #define	WPRINTF( fmt, args... )

	#define	RBPRINTF( fmt, args... )
	#define	GBPRINTF( fmt, args... )
    #define	YBPRINTF( fmt, args... )
    #define	BBPRINTF( fmt, args... )
	#define	MBPRINTF( fmt, args... )
	#define	CBPRINTF( fmt, args... )
	#define	WBPRINTF( fmt, args... )

	#define	RFPRINTF( fmt, args... )
	#define	GFPRINTF( fmt, args... )
	#define	YFPRINTF( fmt, args... )
	#define	BFPRINTF( fmt, args... )
	#define	MFPRINTF( fmt, args... )
	#define	CFPRINTF( fmt, args... )
	#define	WFPRINTF( fmt, args... )

    #define	RBFPRINTF( fmt, args... )
    #define	GBFPRINTF( fmt, args... )
    #define	YBFPRINTF( fmt, args... )
    #define	BBFPRINTF( fmt, args... )
    #define	MBFPRINTF( fmt, args... )
    #define	CBFPRINTF( fmt, args... )
    #define	WBFPRINTF( fmt, args... )

#else	/*	FOR_RELEASE		*/

    #define	DPRINTF( fmt, args... )

	#define	RPRINTF			RED_PRINTF
	#define	GPRINTF			GREEN_PRINTF
	#define	BPRINTF			BLUE_PRINTF
	#define	MPRINTF			MAGENTA_PRINTF
	#define	CPRINTF			CYAN_PRINTF
	#define	WPRINTF			WHITE_PRINTF

	#define	RBPRINTF		RED_BOLD_PRINTF
	#define	GBPRINTF		GREEN_BOLD_PRINTF
	#define	YBPRINTF		YELLOW_BOLD_PRINTF
	#define	BBPRINTF		BLUE_BOLD_PRINTF
	#define	MBPRINTF		MAGENTA_BOLD_PRINTF
	#define	CBPRINTF		CYAN_BOLD_PRINTF
	#define	WBPRINTF		WHITE_BOLD_PRINTF

	#define	RFPRINTF		RED_FULL_PRINTF
	#define	GFPRINTF		GREEN_FULL_PRINTF
	#define	YFPRINTF		YELLOW_FULL_PRINTF
	#define	BFPRINTF		BLUE_FULL_PRINTF
	#define	MFPRINTF		MAGENTA_FULL_PRINTF
	#define	CFPRINTF		CYAN_FULL_PRINTF
	#define	WFPRINTF		WHITE_FULL_PRINTF

	#define	RBFPRINTF		RED_BOLD_FULL_PRINTF
	#define	GBFPRINTF		GREEN_BOLD_FULL_PRINTF
	#define	YBFPRINTF		YELLOW_BOLD_FULL_PRINTF
	#define	BBFPRINTF		BLUE_BOLD_FULL_PRINTF
	#define	MBFPRINTF		MAGENTA_BOLD_FULL_PRINTF
	#define	CBFPRINTF		CYAN_BOLD_FULL_PRINTF
	#define	WBFPRINTF		WHITE_BOLD_FULL_PRINTF

#endif	/*	FOR_RELEASE		*/

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // __DBG_PRN_H__
