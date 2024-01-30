#define _CRT_SECURE_NO_WARNINGS  

#include "IDS.h"
#include <iostream>
using namespace std;

IDS::IDS()
{
	RW = Rolling();
	KHU = KHU_Utils();

	Frame = PacketFeature();

	originalInfo = list<string>();
	rollingTableInfo = list<string>();
	concatenatedInfo = list<string>();
	cout << "Intrusion Detection System Class Initialized" << endl;
}

IDS::~IDS()
{
	cout << "Intrustion Detection System Class bye" << endl;
}

int** IDS::intrusionDetect(tagEthHeaderDataType* packet)
{
	long long startTime = KHU.getTime();

	originalInfo = KHU.packetIntoList(packet);
	
	testList["SRC_IP"] = "127.0.0.1";
	testList["SRC_PORT"] = "9999";
	testList["DST_PORT"] = "9999";
	testList["FLAG"] = "S";
	testList["TIME"] = to_string(KHU.getTime());
	
	int** feature = Frame.getFeature();

	long long endTime = KHU.getTime();
	printf("Packet Preprocessing : %d \n", endTime - startTime);

	return feature;

}
