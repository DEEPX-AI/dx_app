#include "dxrt/dxrt_api.h"
#include "socket.h"

using namespace std;

ssize_t sendDataToSocket(int sock, void *data, ssize_t size, int flags)
{
    ssize_t totalBytes = 0;
    while(1)
    {
        ssize_t bytes = send(sock, (void *)((uintptr_t)data + totalBytes), size - totalBytes, flags);
        if(bytes<0)
        {
            return bytes;
            perror("send");
        }
        totalBytes += bytes;
        if(totalBytes>=size) break;
    }
    return totalBytes;
}
ssize_t receiveDataFromSocket(int sock, void *data, ssize_t size)
{
    ssize_t totalBytes = 0;
    while(true)
    {
        ssize_t bytes = recv(sock, (void *)((uintptr_t)data + totalBytes), size - totalBytes, 0);
        if(bytes<0)
        {
            return bytes;
            perror("receive");
        }
        totalBytes += bytes;
        if(totalBytes >= size) break;
    }
    cout << "Received " << totalBytes << " bytes." << endl;
    return totalBytes;
}