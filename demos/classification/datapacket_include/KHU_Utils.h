#pragma once
#include<iostream>
#include<list>
#include<time.h>
#include<chrono>
#include<ctime>
using namespace std;

#include "TagEthHeaderDataType.h"

class KHU_Utils
{
private:
	int PATCHSIZE = 32;
	int** returnPatch = 0;
	int* garbageRow[32] = { 0 };

public:
	KHU_Utils();
	~KHU_Utils();
		
	list<string> packetIntoList(tagEthHeaderDataType* packet); 
	list<string> concatenate(list<string> originalList, list<string> rollingTableList); 

	int** makePatch(list<string> target, int patchsize); 
	int* convertToBinary(string data); 
	long long getTime(); 

	
	void showPatch(); 
	void printList(list<string> wantToSee); 

};

