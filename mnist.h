#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Mnist{
public:
  vector<vector<float> > readTrainingFile(string filename);
  vector<float> readLabelFile(string filename);
};
