#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#ifdef __linux__
#include <syslog.h>
#endif
#ifdef __linux__
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

#ifdef __linux__
ssize_t sendDataToSocket(int sock, void* data, ssize_t size, int flags);
ssize_t receiveDataFromSocket(int sock, void* data, ssize_t size);
#endif