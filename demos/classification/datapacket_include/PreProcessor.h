#pragma once
#define _CRT_SECURE_NO_WARNINGS 

#include <string>
#include "TagEthHeaderDataType.h"

using namespace std;

void* PreProcessor(eth_header_data_t* src, bool hazardMode, void* portAddr = nullptr);