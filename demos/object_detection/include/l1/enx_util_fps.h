
/* 
 * Copyright (C) 2019 Eyenix Corporation
 * dev-team2, Eyenix <support6@eyenix.com>
 */

#ifndef __ENX_UTIL_FPS_H__
#define __ENX_UTIL_FPS_H__

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#include "enx_types.h"
#include "dbg_prn.h"

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */


#define FPS_BUFFER_SIZE		30

typedef struct {
	// These members manages the circular buffer.
	int m_numCounters; // number of valid counters in m_counters
	int m_curPosition; // the position to write in m_counters
	clock_t m_counters[ FPS_BUFFER_SIZE]; // actual buffer to store clock ticks
	int m_frameSizes[ FPS_BUFFER_SIZE]; // buffer to store frame size to calculate kbps.
} ENX_FPS_S;



// call this when you get a new frame with a known data size
void SetFpsCount( int numBytes);

// gets frame-per-second value using the timestamp of previous FPS_BUFFER_SIZE frames.
double getFps(void);



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __ENX_UTIL_FPS_H__ */



