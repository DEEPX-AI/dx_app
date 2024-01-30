#define _CRT_SECURE_NO_WARNINGS  
#include "KHU_Utils.h"

#include <string>
#include <iostream>
using namespace std;

KHU_Utils::KHU_Utils()
{
	returnPatch = new int* [PATCHSIZE];
	for (int i = 0; i < PATCHSIZE; i++) {
		returnPatch[i] = new int[PATCHSIZE];
	}
}

KHU_Utils::~KHU_Utils()
{
	cout << "KHU_Utils Bye" << endl;
}

list<string> KHU_Utils::packetIntoList(tagEthHeaderDataType* packet)
{
	list<string> returnList;

	returnList.push_back(packet->src_ip_addr); 
	returnList.push_back(packet->src_port); 
	returnList.push_back(packet->dst_ip_addr); 
	returnList.push_back(packet->dst_port); 
	returnList.push_back(packet->flag); 
	returnList.push_back(packet->ip_protocol); 
	returnList.push_back(packet->ttl); 
	returnList.push_back(packet->service); 
	returnList.push_back(packet->payload_length); 

	returnList.push_back(packet->payload_data);
	returnList.push_back(packet->local_time); 

	return returnList; 
}

/***
 * concatenabte rolling tabel under original list
*/
list<string> KHU_Utils::concatenate(list<string> originalList, list<string> rollingTableList)
{
	list<string>::iterator it;

	for (it = rollingTableList.begin(); it != rollingTableList.end(); it++) {
		originalList.push_back(*it);
	}

	return originalList;
}

/***
 * string list to 2D integer ptr
*/
int** KHU_Utils::makePatch(list<string> target, int patchsize)
{
	PATCHSIZE = patchsize;

	list<string>::iterator it;
	int i = 0;
	for (it = target.begin(); it != target.end(); it++) {
		returnPatch[i] = convertToBinary(*it);
		i++;
	}

	while (i < PATCHSIZE) {
		int* garbageRow = new int[PATCHSIZE];

		for (int j = 0; j < PATCHSIZE; j++) {
			garbageRow[j] = 0;
		}
		returnPatch[i] = garbageRow;
		i++;
	}

	return returnPatch;
}

int* KHU_Utils::convertToBinary(string data)
{
	string binaryData = "";

	for (int i = 0; i < data.length(); i++) {
		int letterAscII = int(data.at(i));
		while (letterAscII != 0) {binaryData = (letterAscII % 2 == 0 ? "0" : "1") + binaryData; letterAscII /= 2; }
	}

	while (binaryData.length() < PATCHSIZE) {
		binaryData += "0";
	}

	int* returnBinary = new int[PATCHSIZE];

	for (int i = 0; i < PATCHSIZE; i++) {
		returnBinary[i] = int(stoi(binaryData.substr(i, 1)));
	}

	return returnBinary;
}

long long KHU_Utils::getTime()
{
	auto time = std::chrono::system_clock::now();
	auto mill = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch());

	long long currentTimeMillis = mill.count();
	return currentTimeMillis;
}

void KHU_Utils::showPatch()
{
	printf("=======================================================================\n");
	printf("===========================SHOW PACKET=================================\n");
	for (int i = 0; i < PATCHSIZE; i++) {
		for (int j = 0; j < PATCHSIZE; j++) {
			printf("%d", returnPatch[i][j]);
		}
	}
	printf("=======================================================================\n");
}

void KHU_Utils::printList(list<string> wantToSee)
{
	list<string>::iterator it;
	for (it = wantToSee.begin(); it != wantToSee.end(); it++) {
		cout << *it << endl;
	}
}


