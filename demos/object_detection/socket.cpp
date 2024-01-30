#include "dxrt/dxrt_api.h"
#include "socket.h"

using namespace std;

ssize_t sendDataToSocket(int sock, void *data, size_t size, int flags)
{
    ssize_t totalBytes = 0;
    while(1)
    {
        ssize_t bytes = send(sock, data+totalBytes, size - totalBytes, flags);
        LOG_VALUE(bytes);
        if(bytes<0)
        {
            return bytes;
            perror("send");
        }
        // DXRT_ASSERT(bytes>=0, "write failed.");
        totalBytes += bytes;
        if(totalBytes>=size) break;
    }
    // cout << "Sent " << totalBytes << " bytes." << endl;
    return totalBytes;
}
ssize_t receiveDataFromSocket(int sock, void *data, size_t size, int flags)
{
    ssize_t totalBytes = 0;
    while(1)
    {
        ssize_t bytes = recv(sock, data+totalBytes, size - totalBytes, 0);
        if(bytes<0)
        {
            return bytes;
            perror("receive");
        }
        // LOG_VALUE(bytes);
        // DXRT_ASSERT(bytes>=0, "read failed.");
        totalBytes += bytes;
        if(totalBytes>=size) break;
    }
    cout << "Received " << totalBytes << " bytes." << endl;
    return totalBytes;
}