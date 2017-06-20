#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include<thread>
#include <chrono>
#include <string.h>

#include "dataset.h"

#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

void Dataset::calcSTD(vector<float> data, float &mean, float &std) {
    mean = 0;
    std = 0;
    for (int i = 0; i < data.size(); i++) {
        mean += data[i];
    }
    mean /= data.size();

    for (int i = 0; i < data.size(); i++) {
        std += (data[i] - mean) * (data[i] - mean);
    }
    if (data.size() > 1) {
        std /= (data.size() - 1);
        std = sqrt(std);
    } else {
        std = -1;
    }
}

void Dataset::shuffle(vector<vector<float> > *s, vector<float> *d) {
    std::random_device rnd;
    std::mt19937 mt(rnd());

    int data_size = (*s).size();

    for (int i = 0; i < data_size; i++) {
        int j = mt() % data_size;
        vector<float> t = (*s)[i];
        float td = (*d)[i];
        (*s)[i] = (*s)[j];
        (*d)[i] = (*d)[j];
        (*s)[j] = t;
        (*d)[j] = td;
    }
}
void Dataset::shuffle(vector<vector<float> > *s) {
    std::random_device rnd;
    std::mt19937 mt(rnd());

    int data_size = (*s).size();

    for (int i = 0; i < data_size; i++) {
        int j = mt() % data_size;
        vector<float> t = (*s)[i];
        (*s)[i] = (*s)[j];
        (*s)[j] = t;
    }
}

void Dataset::standrize(vector<vector<float> > *s) {

    vector<float> mean;
    vector<float> std;

    mean.resize((*s).size());
    std.resize((*s).size());

    for (int i = 0; i < (*s).size(); i++) {
        float m1, std1;
        vector<float> dt = (*s)[i];
        calcSTD(dt, m1, std1);
        mean[i] = m1;
        std[i] = std1;
    }
    for (int i = 0; i < (*s).size(); i++) {
        for (int j = 0; j < (*s)[0].size(); j++) {
            (*s)[i][j] = ((*s)[i][j] - mean[i])
                    / (std::max((double) std[i], 1e-8));
        }
    }
}

void Dataset::normalize(vector<vector<float> > *s, float max){
    for (int i = 0; i < (*s).size(); i++) {
        for (int j = 0; j < (*s)[0].size(); j++) {
            (*s)[i][j] /= max;
        }
    }
}

void Dataset::createMiniBatch(vector<vector<float> >&s, vector<float>&d,
        float *X, float *D, int batchSize, int d_size, int l) {

    int M = s[0].size();

    int batchStartIdx = l * batchSize;
    for (int i = 0; i < M; i++) {
        int k = 0;
        for (int j = batchStartIdx; j < batchStartIdx + batchSize; j++) {

            X[IDX2F(i, k, M)] = s[j][i];

            k++;
        }
    }
    for (int i = 0; i < d_size; i++) {
        int k = 0;
        for (int j = batchStartIdx; j < batchStartIdx + batchSize; j++) {

            if (d[j] == i)
                D[IDX2F(i, k, d_size)] = 1;
            else
                D[IDX2F(i, k, d_size)] = 0;

            k++;
        }
    }
}

