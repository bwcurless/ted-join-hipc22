#ifndef DATASET_H
#define DATASET_H

#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <vector>

#include "params.h"

void importDataset(std::vector<std::vector<INPUT_DATA_TYPE>> *inputVector,
                   char *filename);

void importDataset_v2(std::vector<std::vector<INPUT_DATA_TYPE>> *inputVector,
                      char *filename);

#endif
