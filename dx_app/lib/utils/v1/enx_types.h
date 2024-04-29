
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_TYPES_H__
#define	__ENX_TYPES_H__

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

//#include <stdarg.h>			// for va_list
#if 0
typedef int						INT;		// 16 bit or 32 bit according to system
typedef unsigned int			UINT;		//	"

typedef char					CHAR;		// 8 bit
//typedef char					bool;		//	"
typedef unsigned char			UCHAR;		//	"
typedef unsigned char			uchar;		//	"
typedef unsigned char			BYTE;		//	"
typedef unsigned char			BOOL;		//	"
typedef unsigned char			INT8U;		//	"
typedef unsigned char			u8;			//	"
typedef unsigned char			__u8;		//	"
typedef signed	 char			INT8S;		//	"
typedef signed   char			int8;		//	"
typedef signed char				s8;			//	"
typedef signed char				__s8;		//	"
typedef unsigned char			uint8;		//	"
typedef unsigned char			u_int8;		//	"
typedef volatile uint8			vuint8;		//	"

typedef short					SHORT;		// 16 bit
typedef unsigned short			USHORT;		//	"
typedef unsigned short			WORD;		//	"
typedef unsigned short			WCHAR;		//	"
typedef unsigned short			INT16U;		//	"
typedef unsigned short  		u16;		//	"
typedef unsigned short  		__u16;		//	"
typedef unsigned short			uint16;		//	"
typedef unsigned short	    	u_int16;	//	"
typedef signed	 short			INT16S;		//	"
typedef signed   short			int16;		//	"
typedef signed 	 short			s16;		//	"
typedef signed 	 short			__s16;		//	"
typedef volatile uint16			vuint16;	//	"

typedef long					LONG;		// 32 bit
typedef unsigned long			ULONG;		//	"
typedef unsigned long			ulong;		//	"
typedef unsigned long			DWORD;		//	"
typedef unsigned int			INT32U;		//	"
typedef signed   int			INT32S;		//	"
typedef signed   int			int32;		//	"
typedef signed   int 			Int32;		//	"

typedef unsigned long			uint32;		//	"
typedef unsigned long			u_int32;	//	"
typedef volatile uint32			vuint32;	//	"
typedef volatile unsigned int	UInt32;		//	"

typedef	long long				LLONG;		// 64 bit
typedef	long long				llong;		//	"
typedef	unsigned long long		ULLONG;		//	"
typedef unsigned long long		ullong;		//	"
typedef unsigned long long		QWORD;		//	"
typedef unsigned long long		INT64U;		//	"
typedef signed	 long long		INT64S;		//	"
typedef signed   long long		int64;		//	"
typedef unsigned long long		uint64;		//	"

typedef unsigned int 			float32;
typedef unsigned long long 		float64;
#else

//typedef _Bool					bool;		// 1 bit
typedef unsigned char			BOOL;		//  "

typedef char					CHAR;		// 8 bit
typedef unsigned char			UCHAR;		//	"
typedef unsigned char			BYTE;		//	"
typedef unsigned char			uchar;		//	"
typedef signed   char			INT8S;		//	"
typedef unsigned char			INT8U;		//	"
typedef signed   char			s8;			//	"
typedef unsigned char			u8;			//	"
typedef signed   char			__s8;		//	"
typedef unsigned char			__u8;		//	"
typedef signed   char			s_int8;		//	"
typedef unsigned char			u_int8;		//	"
typedef signed   char			int8;		//	"
typedef unsigned char			uint8;		//	"
typedef volatile uint8			vuint8;		//	"

typedef short					SHORT;		// 16 bit
typedef unsigned short			USHORT;		//	"
typedef unsigned short			WORD;		//	"
typedef unsigned short			WCHAR;		//	"
typedef unsigned short			ushort;		//	"
typedef signed   short			INT16S;		//	"
typedef unsigned short			INT16U;		//	"
typedef signed   short			s16;		//	"
typedef unsigned short  		u16;		//	"
typedef signed   short			__s16;		//	"
typedef unsigned short  		__u16;		//	"
typedef signed   short			s_int16;	//	"
typedef unsigned short			u_int16;	//	"
typedef signed   short			int16;		//	"
typedef unsigned short			uint16;		//	"
typedef volatile uint16			vuint16;	//	"

typedef int						INT;		// 32 bit
typedef unsigned int			UINT;		//	"
typedef unsigned int			DWORD;		//	"
typedef unsigned int			uint;		//	"
typedef signed   int			INT32S;		//	"
typedef unsigned int			INT32U;		//	"
typedef signed   int			s32;		//	"
typedef unsigned int			u32;		//	"
typedef signed   int			__s32;		//	"
typedef unsigned int			__u32;		//	"
typedef signed   int			s_int32;	//	"
typedef unsigned int			u_int32;	//	"
typedef signed   int			int32;		//	"
typedef unsigned int			uint32;		//	"
typedef volatile uint32			vuint32;	//	"
typedef int32 					Int32;		//	"
typedef uint32					UInt32;		//	"

typedef long					LONG;		// 64 bit
typedef unsigned long			ULONG;		//	"
typedef unsigned long			QWORD;		//	"
typedef unsigned long			ulong;		//	"
typedef signed   long			INT64S;		//	"
typedef unsigned long			INT64U;		//	"
//typedef signed   long			s64;		//	"
//typedef unsigned long			u64;		//	"
//typedef signed   long			__s64;		//	"
//typedef unsigned long			__u64;		//	"
typedef signed   long			s_int64;	//	"
typedef unsigned long			u_int64;	//	"
typedef signed   long			int64;		//	"
typedef unsigned long			uint64;		//	"
typedef volatile uint64			vuint64;	//	"
typedef unsigned long 			UInt64;		//	"
typedef	long long				LLONG;		// 64 bit
#endif

//******************************************************************************
//
//------------------------------------------------------------------------------
//
#define TRUE					1
#define FALSE					0

#ifndef NULL
#define NULL 					((void*)0)
#endif

#define	UNUSED(x)				((void)(x))

#define ENX_SUCCESS				0
#define ENX_FAIL				1

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // __ENX_TYPES_H__

