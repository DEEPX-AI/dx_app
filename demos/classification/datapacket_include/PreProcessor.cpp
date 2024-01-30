/* KHU */
#include "IDS.h"
#include "KHU_Utils.h"
#include "Rolling.h"
#include "PacketFeature.h"

#include <fstream>

/*******************************/
/*  KHU logics                 */
bool PreProc(void* src, void* dest, bool hazardMode = false, void* portAddr = nullptr);
int PATCHSIZE = 32;
Rolling RW = Rolling();
KHU_Utils KHU = KHU_Utils();
PacketFeature Frame = PacketFeature();
list<string> originalInfo = list<string>(); 
list<string> rollingTableInfo = list<string>(); 
list<string> concatenatedInfo = list<string>(); 
int** patch;
map<string, string> testList;
/*******************************/

/*
temporary buffer to covert from int to uint8 
npu input data shape [224, 224], type : uint8
*/
uint8_t npu_in_u8[224][224];

void DataDumpTxt(string filename, uint8_t *data, size_t ch, size_t row, size_t col)
{
    ofstream f_out(filename, ios::out);
    for(size_t c=0; c<ch;c++)
    {
        for(size_t h=0; h<row;h++)
        {
            for(size_t w=0; w<col;w++)
            {
                f_out << (int)*data << " ";
                data++;
            }
            f_out << endl;
        }
    }
    f_out.close();
}
void DataDumpBin(string filename, uint8_t *data, int32_t size)
{
    FILE *fp;
    fp = fopen(filename.c_str(), "wb");
    fwrite(data, sizeof(uint8_t)*size, 1, fp);
    fclose(fp);
}

void* PreProcessor(eth_header_data_t* src, bool hazardMode, void* portAddr)
{
	void* dest = nullptr;
	src->local_time = to_string(KHU.getTime());

	(void)PreProc(src, &dest, hazardMode);

	{
		int** tmpDest = (int**)dest;
		// printf("=======================================================================\n");
		// printf("===========================show frame==================================\n");
		for (int i = 0; i < 224; i++) {
			for (int j = 0; j < 224; j++) {
				npu_in_u8[i][j] = tmpDest[i][j];
			}
		}
		// printf("=======================================================================\n");
		// DataDumpBin("npu_input_pre-processed.bin", (uint8_t *)npu_in_u8, 224*224);
		// DataDumpTxt("npu_input_pre-processed.txt", (uint8_t *)npu_in_u8, 1, 224, 224);
	}
	dest = (void *)npu_in_u8;

	return dest;
}

bool PreProc(void* src, void* dest, bool hazardMode, void* portAddr) {
	bool result = false;

#ifdef DS_PACKET_CONVERT
  PacketConverter conv; 
  conv = *(ds_eth_header_data_t*) src;
  tagEthHeaderDataType* packet = &conv.getHdr();
	originalInfo = KHU.packetIntoList(packet);
#else
	tagEthHeaderDataType* packet = (tagEthHeaderDataType*)src; // Type Casting into Previous Packet from void *src pointer
	originalInfo = KHU.packetIntoList(packet);
#endif

#if 1
	string testList[7];
	testList[0] = packet->src_ip_addr;
	testList[1] = packet->dst_ip_addr;
	testList[2] = packet->src_port;
	testList[3] = packet->dst_port;
	testList[4] = packet->flag;
	testList[5] = packet->payload_length;
	testList[6] = packet->local_time;
#endif  

	RW.add(testList);

	rollingTableInfo = RW.extractInfo(); // Rolling Window


	concatenatedInfo = KHU.concatenate(originalInfo, rollingTableInfo); // Rolling Window

	patch = KHU.makePatch(concatenatedInfo, PATCHSIZE); // Patch

	Frame.append(patch, hazardMode);
	
	int** feature = Frame.getFeature();
	
	*(void**)dest = feature;

	if (hazardMode && portAddr != nullptr) {
		*(int*)portAddr = stoi(packet->dst_port);
	}

	result = true;
	return result;
}
