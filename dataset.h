#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include<thread>
#include <chrono>

using namespace std;

#ifndef _Dataset_
#define _Dataset_

class Dataset {
public:
    void standrize(vector<vector<float> > *s);

    void normalize(vector<vector<float> > *s, float max);

    void calcSTD(vector<float> data, float &mean, float &std);

    void createMiniBatch(vector<vector<float> >&s, vector<float>&d, float *X,
            float *D, int batchSize, int d_size, int l);

    void shuffle(vector<vector<float> > *s, vector<float> *d);
    void shuffle(vector<vector<float> > *s);
};

#endif

