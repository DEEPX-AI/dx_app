#pragma once
#include <string>
#include <list>
#include <array>
#include <iostream>
#include <cmath>
using namespace std;

class PacketFeature
{
private:
	int patchCount = 0;

	int FEATURESIZE = 224;

	int stride = 32;
	int row = 0;
	int col = 0;

	int current_row = 0;
	int current_col_start = 0;
	int current_col_end = 0;

	bool hazardMode = false;

	int** frame;

public:

	PacketFeature();
	~PacketFeature();

	void append(int** patch, bool hazard); 
	void showFrame(); 

	int** getFeature(); 

};

