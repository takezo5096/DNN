/*
 * cuMat.h
 *
 *  Created on: 2016/01/12
 *      Author: takeshi.fujita
 */

#ifndef CUMAT_H_
#define CUMAT_H_

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
//#include <chrono>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>


#include <cublas_v2.h>

#include "mat_mul_elementwise_kernel.h"
#include "matmod_kernel.h"
#include "mat_log_kernel.h"
#include "mat_sqrt_kernel.h"
#include "relu_kernel.h"
#include "relu_d_kernel.h"
#include "sigmoid_kernel.h"
#include "sigmoid_d_kernel.h"
#include "tanh_kernel.h"
#include "tanh_d_kernel.h"
#include "softmax_kernel.h"
#include "dropout_kernel.h"
#include "mat_ones_kernel.h"
#include "mat_sum_kernel.h"
#include "mat_div_kernel.h"
#include "mat_mul_elementwise_plus_kernel.h"
#include "adam_kernel.h"
#include "adam2_kernel.h"
#include "mat_sin_kernel.h"
#include "mat_cos_kernel.h"
#include "softmax_cross_entropy_kernel.h"
#include "mat_l2_kernel.h"


using namespace std;

#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

class MallocCounter {
public:
    int num = 0;
    void up(){
        num++;
    }
    void down(){
        num--;
    }

    int get(){
        return num;
    }

};

extern MallocCounter mallocCounter;

class cuMat {

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {
        ar & mHostArray;
        ar & rows;
        ar & cols;
    }

public:
    float *mDevice = NULL;
    float *mHost = NULL;
    vector<float> mHostArray;
    int rows = 0;
    int cols = 0;

    cublasHandle_t cudaHandle;



    cuMat() {

        cublasCreate(&cudaHandle);
        cudaThreadSynchronize();
    }

    cuMat(int rows, int cols) {

        cublasCreate(&cudaHandle);
        cudaThreadSynchronize();

        //cout << "cuMat constractor rows cols" << endl;
        new_matrix(rows, cols);

    }

    cuMat(const cuMat &a) {
        cublasCreate(&cudaHandle);
        cudaThreadSynchronize();
        //cout << "cuMat copy constractor" << endl;
        new_matrix(a.rows, a.cols);

        cudaError_t error = cudaMemcpy(mDevice, a.mDevice,
                rows * cols * sizeof(*mDevice), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            printf("cuMat copy constractor cudaMemcpy error\n");

    }

    ~cuMat() {
        //cout << "cuMat ~cuMat" << endl;
        del_matrix();
        cublasDestroy(cudaHandle);
    }


    int getRows() {
        return this->rows;
    }
    int getCols() {
        return this->cols;
    }

    void memMallocHost() {
        mHost = (float *) malloc(rows * cols * sizeof(*mHost));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mHost[IDX2F(i, j, rows)] = 0.0;
            }
        }
    }
    void memMallocDevice() {
        cudaError_t error = cudaMalloc((void**) &mDevice,
                rows * cols * sizeof(*mDevice));
        if (error != cudaSuccess) printf("cudaMemcpy error\n");
        cudaMemset(mDevice, 0x00, rows * cols * sizeof(*mDevice));
        cudaThreadSynchronize();

    }

    void new_matrix(int rows, int cols) {
        //cout << "new_matrix" << endl;
        if (this->rows != rows || this->cols != cols) {
            if (mDevice != NULL || mHost != NULL){
                //cout << "new_matrix 2" << endl;
                del_matrix();
            }
            this->rows = rows;
            this->cols = cols;

            cudaError_t error;
            cublasStatus_t stat;

            error = cudaMalloc((void**) &mDevice,
                    rows * cols * sizeof(*mDevice));
            if (error != cudaSuccess) printf("cuMat::new_matrix cudaMalloc error\n");

            cudaMemset(mDevice, 0x00, rows * cols * sizeof(*mDevice));
            cudaThreadSynchronize();
            mallocCounter.up();
        }
    }

    void del_matrix() {
        //cout << "del_matrix" << endl;
        if (mDevice != NULL){
            cudaFree(mDevice);
            mDevice = NULL;
            mallocCounter.down();
        }
        if (mHost != NULL){
            free(mHost);
            mHost = NULL;
        }
        cudaThreadSynchronize();
    }

    void memHostToDevice() {
        cudaError_t error = cudaMemcpy(mDevice, mHost,
                rows * cols * sizeof(*mDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) printf("cudaMemcpy error\n");
    }
    void memDeviceToHost() {
        cudaError_t error = cudaMemcpy(mHost, mDevice,
                rows * cols * sizeof(*mDevice), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            printf("cudaMemcpy error\n");
    }
    void memSetHost(int i, int j, float val) {
        if (mHost == NULL)
            this->memMallocHost();
        mHost[IDX2F(i, j, rows)] = val;
    }
    void memSetHost(float *v) {
        cudaError_t error = cudaMemcpy(mDevice, v,
                rows * cols * sizeof(*mDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            printf("cudaMemcpy error\n");
    }

    void toHostArray(){
        //cout << "toHostArray" << endl;
        if (mHost == NULL) this->memMallocHost();
        memDeviceToHost();

        mHostArray.resize(rows*cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++){
                mHostArray[IDX2F(i, j, rows)] = mHost[IDX2F(i, j, rows)];
            }
        }
    }
    void fromHostArray(){
        //cout << "fromHostArray" << endl;
        if (mDevice == NULL) this->memMallocDevice();
        if (mHost == NULL) this->memMallocHost();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++){
                mHost[IDX2F(i, j, rows)] = mHostArray[IDX2F(i, j, rows)];
            }
        }

        memHostToDevice();
        //cout << *this;
    }


    cuMat &operator=(const cuMat &a) {
        //cout << "cuMat operator=" << endl;
        new_matrix(a.rows, a.cols);

        cudaError_t error = cudaMemcpy(mDevice, a.mDevice,
                rows * cols * sizeof(*mDevice), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            printf("cuMat operator= cudaMemcpy error\n");

        return *this;
    }

    float operator()(int i, int j) {
        if (mHost == NULL)
            this->memMallocHost();

        this->memDeviceToHost();

        return mHost[IDX2F(i, j, rows)];

    }

    friend void printRows(ostream &output, cuMat &a, int i){
        output << "[";
        if (a.cols < 6){
            for (int j = 0; j < a.cols; j++)  output << a.mHost[IDX2F(i, j, a.rows)] << " ";
        }
        else{
            for (int j = 0; j < 3; j++)  output << a.mHost[IDX2F(i, j, a.rows)] << " ";
            cout << "..., ";
            for (int j = a.cols-2; j < a.cols; j++)  output << a.mHost[IDX2F(i, j, a.rows)] << " ";
        }
        output << "]";
    }


    friend ostream &operator<<(ostream &output, cuMat &a) {

        if (a.mDevice == NULL){
            printf("cuMat operator<< a.mDevice is NULL\n");
            if (a.mHost == NULL){
                printf("also cuMat operator<< a.mHost is NULL\n");
            }
        }
        if (a.mHost == NULL) a.memMallocHost();


        cudaError_t error = cudaMemcpy(a.mHost, a.mDevice,
                a.rows * a.cols * sizeof(*a.mDevice), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            printf("cuMat operator<< cudaMemcpy error\n");

        output << "matrix rows:" << a.rows << " cols:" << a.cols << endl;
        output << "[";
        if (a.rows < 11){
            for (int i = 0; i < a.rows; i++) {
                printRows(output, a, i);
                if (i!=a.rows-1) output << endl;
                else output << "]" << endl;
            }
        }
        else{
            for (int i = 0; i < 5; i++) {
                printRows(output, a, i);
                output << endl;
            }
            output << "...," << endl;
            for (int i = a.rows -5; i < a.rows; i++) {
                printRows(output, a, i);
                if (i!=a.rows-1) output << endl;
                else output << "]" << endl;
            }
        }

        return output;
    }

    friend cuMat operator+(const cuMat &a, const cuMat &b) {
        cuMat r = a;
        r.plus(b, r);

        return r;
    }
    friend cuMat operator+(float a, cuMat &b) {
        cuMat r = b;
        b.plus(a, r);

        return r;
    }
    friend cuMat operator+(const cuMat &b, float a) {
        cuMat r = b;
        r.plus(a, r);

        return r;
    }

    friend cuMat operator-(const cuMat &a, const cuMat &b) {
        cuMat r = a;
        r.minus(b, r);

        return r;
    }

    friend cuMat operator*(const cuMat &a, const cuMat &b) {
        cuMat r = a;
        r.mul(b, r); //dotではなくmulとする

        return r;
    }
    friend cuMat operator*(float a, const cuMat &b) {
        cuMat r = b;
        r.mul(a, r);

        return r;
    }
    friend cuMat operator*(const cuMat &b, float a) {
        cuMat r = b;
        r.mul(a, r);

        return r;
    }
    friend cuMat operator/(float p, cuMat &b) {
        cuMat r = b;
        b.div(p, r);

        return r;
    }
    friend cuMat operator/(const cuMat &b, float p) {
            cuMat r = b;
            r.mul(1.0/p, r);

            return r;
    }
    friend cuMat operator/(const cuMat &a, const cuMat &b) {
            cuMat r = a;
            r.div(b, r);

            return r;
        }

    cuMat &operator+=(const cuMat &a) {
        plus(a, *this);
        return *this;
    }
    cuMat &operator+=(float a) {
        plus(a, *this);
        return *this;
    }
    cuMat &operator-=(const cuMat &a) {
        minus(a, *this);
        return *this;
    }
    cuMat &operator-=(float a) {
        plus(-a, *this);
        return *this;
    }
    cuMat &operator*=(cuMat &a) {
        mul(a, *this);
        return *this;
    }
    cuMat &operator*=(float a) {
        mul(a, *this);
        return *this;
    }

    void copy(const cuMat &a) {
        if (rows != a.rows || cols != a.cols) {
            cout << "cuMat copy error rows != a.rows || cols != a.cols" << endl;
        }
        cudaError_t error = cudaMemcpy(mDevice, a.mDevice,
                rows * cols * sizeof(*mDevice), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            printf("cudaMemcpy error\n");
    }

    void ones() {
        mat_ones_kernel_exec(mDevice, mDevice, cols, rows);
    }

    void plus(const cuMat &b, cuMat &r) {

        float alpha = 1;
        float beta = 1;
        cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
                CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
                b.mDevice, rows, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }

    void minus(const cuMat &b, cuMat &r) {

        float alpha = 1;
        float beta = -1;
        cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
                CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
                b.mDevice, rows, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }

    void mul(const float alpha, cuMat &r) {

        float beta = 0;
        cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
                CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
                r.mDevice, r.rows, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void mul_plus(const float alpha, cuMat &r) {

        float beta = 1;
        cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
                CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
                r.mDevice, r.rows, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }



    void plus(const float beta, cuMat &r) {

        cuMat i(rows, cols);
        i.ones();
        //r.ones();

        float alpha = 1;
        cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
                CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
                i.mDevice, i.rows, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void plus(const float beta, cuMat &i, cuMat &r) {

        //i.ones();
        //r.ones();

        float alpha = 1;
        cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
                CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
                i.mDevice, i.rows, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }

    void div(const float p, cuMat &r) {

        matmod_kernel_exec(mDevice, r.mDevice, cols, rows, p);
    }

    void div(const cuMat &b, cuMat &r){
        mat_div_kernel_exec(mDevice, b.mDevice, r.mDevice, cols, rows);
    }

    cuMat dot(const cuMat &b) {
        cuMat r(this->rows, b.cols);
        dot(b, r);
        return r;
    }
    void dot(const cuMat &b, cuMat &r) {

        if (cols != b.rows) {
            cout << "operator dot error => a.rows != b.cols || a.cols != b.rows"
                    << endl;
            return;
        }

        float alpha = 1;
        float beta = 0;

        cublasStatus_t stat = cublasSgemm(cudaHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                rows, b.cols, cols, &alpha, mDevice, rows, b.mDevice, b.rows,
                &beta, r.mDevice, r.rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgemm" << endl;
        cudaThreadSynchronize();
    }
    void dot_plus(const cuMat &b, cuMat &r) {

                float alpha = 1;
                float beta = 1;

                cublasStatus_t stat = cublasSgemm(cudaHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        rows, b.cols, cols,
                        &alpha, mDevice, rows,
                        b.mDevice, b.rows,
                        &beta, r.mDevice, r.rows);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    cout << "cannot cublasSgemm" << endl;
                cudaThreadSynchronize();
        }

    void transpose_dot_plus(const cuMat &b, cuMat &r) {

            float alpha = 1;
            float beta = 1;

            cublasStatus_t stat = cublasSgemm(cudaHandle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    cols, b.cols, rows,
                    &alpha, mDevice, rows,
                    b.mDevice, b.rows,
                    &beta, r.mDevice, r.rows);
            if (stat != CUBLAS_STATUS_SUCCESS)
                cout << "cannot cublasSgemm" << endl;
            cudaThreadSynchronize();
    }
    void dot_transpose_plus(const cuMat &b, cuMat &r) {

            float alpha = 1;
            float beta = 1;

            cublasStatus_t stat = cublasSgemm(cudaHandle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    rows, b.rows, cols,
                    &alpha, mDevice, rows,
                    b.mDevice, b.rows,
                    &beta, r.mDevice, r.rows);
            if (stat != CUBLAS_STATUS_SUCCESS)
                cout << "cannot cublasSgemm" << endl;
            cudaThreadSynchronize();
    }

    cuMat transpose() {
        cuMat r(cols, rows);
        transpose(r);
        return r;
    }
    void transpose(cuMat &r) {
        //reverse
        float alpha = 1;
        float beta = 0;
        cublasStatus_t stat = cublasSgeam(cudaHandle,
                CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &alpha, mDevice, rows, &beta, r.mDevice, cols,
                r.mDevice, cols);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }

    void plus_util(float alpha, float beta, cuMat &b, cuMat &r) {

        cublasStatus_t stat = cublasSgeam(cudaHandle,
                CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta, b.mDevice, rows,
                r.mDevice, rows);
        if (stat != CUBLAS_STATUS_SUCCESS)
            cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }

    void mul(const cuMat &m, cuMat &r) {

        //matmul2_kernel_exec(mDevice, m.mDevice, r.mDevice, cols, rows);
        mat_mul_elementwise_kernel_exec(mDevice, m.mDevice, r.mDevice, cols, rows);
    }
    void mul_plus(const cuMat &m, cuMat &r, float alpha, float beta) {

        mat_mul_elementwise_plus_kernel_exec(mDevice, m.mDevice, r.mDevice, alpha, beta, cols, rows);
    }

    cuMat log() {
        cuMat r(rows, cols);
        log(r, 0.0);
        return r;
    }
    void log(cuMat &r, float alpha) {

        mat_log_kernel_exec(mDevice, r.mDevice, cols, rows, alpha);

    }

    cuMat sqrt() {
        cuMat r(rows, cols);
        sqrt(r, 0.0);
        return r;
    }
    void sqrt(cuMat &r, float alpha) {

        mat_sqrt_kernel_exec(mDevice, r.mDevice, cols, rows, alpha);
    }

    cuMat sin(){
        cuMat r(rows, cols);
        sin(r);
        return r;
    }
    void sin(cuMat &r){
        mat_sin_kernel_exec(mDevice, r.mDevice, cols, rows, 0);
    }
    cuMat cos(){
         cuMat r(rows, cols);
         cos(r);
         return r;
     }
     void cos(cuMat &r){
         mat_cos_kernel_exec(mDevice, r.mDevice, cols, rows, 0);
     }

    cuMat relu() {
        cuMat r(rows, cols);
        relu(r);
        return r;
    }
    void relu(cuMat &r) {

        relu_kernel_exec(mDevice, r.mDevice, cols, rows);
    }

    cuMat relu_d() {
        cuMat r(rows, cols);
        relu_d(r);
        return r;
    }
    void relu_d(cuMat &r) {

        relu_d_kernel_exec(mDevice, r.mDevice, cols, rows);
    }


    cuMat sigmoid() {
        cuMat r(rows, cols);
        sigmoid(r);
        return r;
    }
    void sigmoid(cuMat &r) {

        sigmoid_kernel_exec(mDevice, r.mDevice, cols, rows);
    }

    cuMat sigmoid_d() {
        cuMat r(rows, cols);
        sigmoid_d(r);
        return r;
    }
    void sigmoid_d(cuMat &r) {

        sigmoid_d_kernel_exec(mDevice, r.mDevice, cols, rows);
    }


    cuMat tanh() {
        cuMat r(rows, cols);
        tanh(r);
        return r;
    }
    void tanh(cuMat &r) {

        tanh_kernel_exec(mDevice, r.mDevice, cols, rows);
    }

    cuMat tanh_d() {
        cuMat r(rows, cols);
        tanh_d(r);
        return r;
    }
    void tanh_d(cuMat &r) {

        tanh_d_kernel_exec(mDevice, r.mDevice, cols, rows);
    }



    cuMat softmax() {
        cuMat r(rows, cols);
        softmax(r);
        return r;
    }
    void softmax(cuMat &r) {

        softmax_kernel_exec(mDevice, r.mDevice, cols, rows);
    }

    float sum() {
        float *sum_d;
        float sum_h=0;
        cudaError_t error = cudaMalloc((void**) &sum_d, sizeof(*sum_d));
        if (error != cudaSuccess) printf("cudaMemcpy error\n");
        cudaThreadSynchronize();
        cudaMemset(sum_d, 0x00, sizeof(*sum_d));
        mat_sum_kernel_exec(mDevice, sum_d, cols, rows);

        error = cudaMemcpy(&sum_h, sum_d, sizeof(*sum_d),
                cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            printf("cudaMemcpy error\n");

        return sum_h;
    }

    float l2() {
            float *sum_d;
            float sum_h=0;
            cudaError_t error = cudaMalloc((void**) &sum_d, sizeof(*sum_d));
            if (error != cudaSuccess) printf("cudaMemcpy error\n");
            cudaThreadSynchronize();
            cudaMemset(sum_d, 0x00, sizeof(*sum_d));
            mat_l2_kernel_exec(mDevice, sum_d, cols, rows);

            error = cudaMemcpy(&sum_h, sum_d, sizeof(*sum_d),
                    cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
                printf("cudaMemcpy error\n");
            return std::sqrt(sum_h);
        }

    void maxRowIndex(int *idx) {
        //std::chrono::system_clock::time_point  start, end;
        //start = std::chrono::system_clock::now();

        if (mHost == NULL)
            this->memMallocHost();
        memDeviceToHost();
        float max[cols];
        for (int j = 0; j < cols; j++) {
            max[j] = 0;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mHost[IDX2F(i, j, rows)] > max[j]) {
                    idx[j] = i;
                    max[j] = mHost[IDX2F(i, j, rows)];
                }
            }
        }
        //end = std::chrono::system_clock::now();
        //int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        //cout << elapsed << endl;
    }

    void dropout(cuMat &r, cuMat &idx, float p) {
        dropout_kernel_exec(mDevice, r.mDevice, idx.mDevice, cols, rows, p);
    }

    void adam(cuMat &b, cuMat &r, float lr, float e){
        adam_kernel_exec(mDevice, b.mDevice, r.mDevice, lr, e, cols, rows);
    }
    void adam2(cuMat &mv, cuMat &mg, cuMat &r, float beta1, float beta2, float lr, float e){
        adam2_kernel_exec(mDevice, mv.mDevice, mg.mDevice, r.mDevice, beta1, beta2, lr, e, cols, rows);
    }

    void softmax_cross_entropy(cuMat &t, cuMat &r){
        softmax_cross_entropy_kernel_exec(mDevice, t.mDevice, r.mDevice, cols, rows);
    }

    void fill(float a){
        this->ones();
        this->mul(a, *this);
    }
};

#endif /* CUMAT_H_ */
