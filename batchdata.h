/*
 * batchdata.h
 *
 *  Created on: 2016/01/15
 *      Author: takeshi.fujita
 */

#ifndef BATCHDATA_H_
#define BATCHDATA_H_

class BatchData {
public:
    float *X = NULL;
    float *D = NULL;

    int m, n, batchSize;

    BatchData(int n, int m, int batchSize){
        X = (float *) malloc(sizeof(*X) * n * batchSize);
        D = (float *) malloc(sizeof(*D) * m * batchSize);
        this->m = m;
        this->n = n;
        this->batchSize = batchSize;
    }

    ~BatchData(){
        if (X!=NULL){
            free(X);
            X = NULL;
        }
        if (D!=NULL){
            free(D);
            D = NULL;
        }
    }

    float *getX(){
        return X;
    }
    float *getD(){
        return D;
    }
};



#endif /* BATCHDATA_H_ */
