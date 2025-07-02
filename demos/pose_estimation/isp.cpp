#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>
#include "isp.h"

using namespace std;

static int memfd;
static unsigned char *ispBuf = NULL;

unsigned char *InitISPMapping(size_t mapSize, off_t phyAddr)
{
    memfd = open("/dev/mem", O_RDWR);
    if(memfd < 0){
        printf("mem open error\n");
        return nullptr;
    }
    ispBuf = (unsigned char *)mmap(
        0,						// addr
        mapSize,			// len
        PROT_READ|PROT_WRITE,	// prot
        MAP_SHARED,				// flags
        memfd,					// fd
        phyAddr		    // offset
    );
    return ispBuf;
};

void DeinitISPMapping(size_t imgSize)
{
    munmap(ispBuf, imgSize);
    close(memfd);
};
