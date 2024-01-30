#pragma once
#define _CRT_SECURE_NO_WARNINGS 

#include <iostream>
#include <list>
#include <map>
#include <time.h>
#include <string>
using namespace std;

#include "KHU_Utils.h"
struct additional {
	string sbytes;
	string packet_cnt; 
	string syn_time; 
	string ack_time;
	string syn_ack_time;
};
class Rolling
{
private:
	int curProcessingSrcIpNum = 0;  		// incoming ip count in 2 sec
	int curProcessingSrcPortNum = 0; 		// incoming port count in 2sec
	int curProcessingDstPortNum = 0; 		
	int curProcessingPacketDstPort = 9999; 	// destination of incoming packet

	int windowTime = 2000;					// ms, rolling window data size

	int synCount = 0;
	int ackCount = 0; 

	float synRatio = 0.0; 
	float ackRatio = 0.0; 

	string currentTime;  
	string lastSynTime = "0"; 
	string lastAckTime = "0"; 

	long long syn2ackTime = 0; 

	list<map<string, string>>* table;

	KHU_Utils KHU;

	map<string, string> lastComp;
	map<pair<string, string>, additional> transactions;

	string currentSRCIP;
	string currentDSTIP;
	string currentSRCPORT;
	string currentDSTPORT;
	string currentFLAG;

public:
	Rolling();
	~Rolling();

	void print(); 
	void add(string item[7]);  	// add new items to rooling window
	list<string> extractInfo();
};