/*
 * variable.cpp
 *
 *  Created on: 2015/12/25
 *      Author: takeshi.fujita
 */

#include <list>
#include <random>
#include <iostream>
#include <chrono>

#include "variable.h"
#include "function.h"

using namespace std;




//global variable for id
int gVariableId = 0;

// Variable class //////////////////////////////////////////////////////
Variable::Variable() {
    id = gVariableId;
    gVariableId++;
}

Variable::Variable(const Variable &a){
    id = gVariableId;
    gVariableId++;

    data = a.data;
    grad = a.grad;
    data_sparse = a.data_sparse;

    seed = a.seed;

    creator = a.creator;


    this->isGetGrad = a.isGetGrad;
    this->isSparse = a.isSparse;


}

Variable::Variable(int rows, int cols) {
    id = gVariableId;
    gVariableId++;

    data = cuMat(rows, cols);
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;

}

Variable::Variable(int rows, int cols, bool is_get_grad){

    this->isGetGrad = is_get_grad;

    id = gVariableId;
    gVariableId++;

    data = cuMat(rows, cols);
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;


}

Variable::Variable(cuMat &input) {
    id = gVariableId;
    gVariableId++;

    data = input;
    grad = cuMat(input.rows, input.cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;
}

Variable::Variable(Function *f, int rows, int cols) {
    id = gVariableId;
    gVariableId++;

    data = cuMat(rows, cols);
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = f;
}

Variable::Variable(Function *f, cuMat &input) {
    id = gVariableId;
    gVariableId++;

    data = input;
    grad = cuMat(input.rows, input.cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = f;
}

Variable::Variable(vector<float> &ids, int nums){

    id = gVariableId;
    gVariableId++;

    data_sparse = cuMatSparse(ids, nums);
    //cuMat tmp = data_sparse.toDense();
    grad = cuMat(data_sparse.rows, data_sparse.cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;

    this->isGetGrad = false;
    this->isSparse = true;

}


Variable::~Variable() {
    //cout << "Variable::~Variable()" << endl;
}



Variable &Variable::operator=(const Variable &a) {

    id = gVariableId;
    gVariableId++;

    data = a.data;
    grad = a.grad;
    //data_sparse = a.data_sparse;

    seed = a.seed;

    creator = a.creator;


    this->isGetGrad = a.isGetGrad;
    this->isSparse = a.isSparse;

    return *this;
}


void Variable::creatorSet(Function *f) {
    this->creator = f;
}

void Variable::backward() {

    this->grad = seed;

    this->backward(this);

}
void Variable::backward(Variable *v) {

    if (v == NULL) {
        return;
    }

    if (v->creator != NULL) {

        //cout << "backward name1:" << v->creator->name << endl;


        if (v->last_opt != NULL && v->opt == *v->last_opt){
            *v->is_last_backward = true;
            //cout << "is backwoard v->opt:" << v->opt << " v->creator->name:" << v->creator->name << endl;
        }

        if (v->forward_count >0) v->forward_count--;

        if (v->is_last_backward != NULL && *v->is_last_backward == false) return;

        if (v->forward_count != 0) return;


        v->creator->backward(v->grad);

        //cout << "backward name2:" << v->creator->name << endl;

        for (int i = 0; i< v->creator->inputs.size(); i++) {

            PVariable nv = v->creator->inputs[i];

            this->backward(nv.get());
        }
    }
    else{
    }
}


void Variable::zero_grads() {

    this->zero_grads(this);
}
void Variable::zero_grads(Variable *v) {

    if (v == NULL)
        return;

    v->grad.mul(0, v->grad);
    v->forward_count = 0;

    if (v->creator != NULL) {


        for (int i = 0; i < v->creator->inputs.size(); i++) {
            PVariable nv = v->creator->inputs[i];

            this->zero_grads(nv.get());
        }

    }
}


void Variable::ones() {
    data.ones();
    grad.mul(0, grad);

}
void Variable::zeros() {
    data.mul(0, data);
    grad.mul(0, grad);
    forward_count = 0;
    last_opt = NULL;
    is_last_backward = NULL;
    this->creator = NULL;
}
void Variable::unchain(){
    this->creator = NULL;
}

void Variable::zero_grad(){
    grad.mul(0, grad);
}

void Variable::randoms(float m, float a) {
    random_device rd;
    mt19937 mt(rd());
    //uniform_real_distribution<float> initd1(-a, a);
    normal_distribution<float> initd1(m, a);
    //gamma_distribution<float> initd1(a, 1.0);


    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols; j++) {
            data.memSetHost(i, j, initd1(mt));
        }
    }
    data.memHostToDevice();
}

float Variable::val(){
    return data(0, 0);
}


