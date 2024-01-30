#include "PacketFeature.h"

PacketFeature::PacketFeature()
{
	frame = new int* [FEATURESIZE];
	for (int i = 0; i < FEATURESIZE; i++) {
		frame[i] = new int[FEATURESIZE];
		for (int j = 0; j < FEATURESIZE; j++) {
			frame[i][j] = 0;
		}
	}
}

PacketFeature::~PacketFeature()
{
	cout << "PacketFeature Bye" << endl;
}

/***
 * append patch data to frame
*/
void PacketFeature::append(int** patch, bool hazard)
{
	if (hazardMode != hazard) {
		if (hazard) {
			for (int i = 0; i < FEATURESIZE; i++) {
				frame[i] = new int[FEATURESIZE];
				for (int j = 0; j < FEATURESIZE; j++) {
					frame[i][j] = 0;
				}
			}
		}
	}
	hazardMode = hazard; 

	int count = FEATURESIZE / stride; 

	
	if (patchCount > std::pow((FEATURESIZE / stride),2)) {
		patchCount = 0;
	}

	if (hazardMode) {
		patchCount = 24;
	}

	row = patchCount / count; 
	col = patchCount % count; 

	if (row > count - 1) {
		row = 0;
	}

	for (int nth_row = 0; nth_row < stride; nth_row++) { 
		int current_row = row * stride + nth_row; 
		int current_col_start = col * stride; 
		int current_col_end = current_col_start + stride; 

		for (int column = current_col_start; column < current_col_end; column++) { 
			frame[current_row][column] = patch[nth_row][column - current_col_start]; 
		}
	}
	patchCount++; 
}

void PacketFeature::showFrame()
{
	printf("=======================================================================\n");
	printf("===========================show frame==================================\n");

	for (int i = 0; i < FEATURESIZE; i++) {
		for (int j = 0; j < FEATURESIZE; j++) {
			printf("%d", frame[i][j]);
		}
		printf("\n");
	}
	printf("=======================================================================\n");

}

int** PacketFeature::getFeature()
{
	return frame;
}
