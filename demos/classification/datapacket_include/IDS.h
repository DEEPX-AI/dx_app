#pragma once

#include<time.h>
#include<chrono>
#include<ctime>
using namespace std;

#include "KHU_Utils.h"
#include "TagEthHeaderDataType.h"
#include "Rolling.h"
#include "PacketFeature.h"

class IDS
{
public:
	int PATCHSIZE = 32;

	IDS();
	~IDS();

	Rolling RW;
	KHU_Utils KHU;
	PacketFeature Frame;

	list<string> originalInfo; // incoming packet information
	list<string> rollingTableInfo; // rolling window output data
	list<string> concatenatedInfo; // stack rolling table under incoming data 

	int** patch;

	map<string, string> testList;

	int** intrusionDetect(tagEthHeaderDataType* packet);
};