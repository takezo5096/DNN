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
int gVariableId = -1;

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

    //cout << "Variable constractor1" << endl;
    data = cuMat(rows, cols);
    //cout << "Variable constractor2" << endl;
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;

}

Variable::Variable(int rows, int cols, bool is_get_grad){

    this->isGetGrad = is_get_grad;

    id = gVariableId;
    gVariableId++;

    //cout << "Variable constractor1" << endl;
    data = cuMat(rows, cols);
    //cout << "Variable constractor2" << endl;
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

    //cout << "Variable::Variable embed 1 nums:" << nums << endl;
    //for(int id : ids) cout << id  << ",";
    //cout << endl;
    data_sparse = cuMatSparse(ids, nums);
    //cuMat tmp = data_sparse.toDense();
    //cout << tmp;
    //cout << "Variable::Variable embed 2" << endl;
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

/*
 Variable Variable::sin(){
     vector<Variable *> inputs;
     inputs.push_back(this);
     Function *f = new FunctionSin();
     Variable vn = *f->forward(inputs);
     return vn;
 }
 Variable Variable::log(){
      vector<Variable *> inputs;
      inputs.push_back(this);
      Function *f = new FunctionLog();
      Variable *vn = f->forward(inputs);
      return *vn;
  }
*/

void Variable::creatorSet(Function *f) {
    this->creator = f;
}

void Variable::backward() {

    this->grad = seed;

    this->backward(this);

}
void Variable::backward(Variable *v) {
    if (v == NULL)
        return;

    if (v->creator != NULL) {
        //cout << "Variable::backward  1" << endl;
        //cout << v->grad;
        //cout << "Variable::backward v->grad.sum():" << v->grad.sum() << endl;

        //std::chrono::system_clock::time_point  start, end;

        //start = std::chrono::system_clock::now();

        v->creator->backward(v->grad);

        //end = std::chrono::system_clock::now();
        //int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        //cout << "v->creator->name:" << v->creator->name << " v->creator->id:" << v->creator->id << " time:" << elapsed << endl;

        if (v->creator->paramsStackNums.empty()) return;

        //cout << "&&&&&&&&&&&&&&&1 Variable::backward  v->creator->id:" << v->creator->id << " v->creator->paramsStackNums.size():" << v->creator->paramsStackNums.size() << endl;
        int paramNums = v->creator->paramsStackNums.back();

        FunctionParam *p  = v->creator->paramsStack[paramNums];
        bool isPop = v->creator->popParamStack();
        //cout << "&&&&&&&&&&&&&&&2 Variable::backward  v->creator->id:" << v->creator->id << " paramNums:" << paramNums << " v->creator->paramsStackNums.size():" << v->creator->paramsStackNums.size() << endl;


        for (int i = p->inputs.size()-1; i >= 0; i--) {
            PVariable nv = p->inputs[i];
            //cout << "Variable::backward 2" << endl;
            //cout << nv->grad;

            this->backward(nv.get());
        }
        //v->creator->clearParamStack(isPop);

    }
}


void Variable::zero_grads() {

    this->zero_grads(this);
}
void Variable::zero_grads(Variable *v) {

    for (Function *f : v->functions_history){
        f->init();
    }
}

void Variable::unchain() {

    this->unchain(this);

}

void Variable::unchain(Variable *vv) {

    //cout << "Variable::unchain vv->functions_history.size():" << vv->functions_history.size() << endl;
    for (Function *f : vv->functions_history){

        //if (f->paramsStackNums.size() > 0){
        //    cout << "Variable::unchain id:" << f->id << " paramsStackNums.size():" << f->paramsStackNums.size() << endl;
        //    exit(1);
        //}

        f->paramsStackNums.clear();

        for (FunctionParam *p : f->paramsStack){
            if (p != NULL){
                delete p; p = NULL;
            }
        }
        f->paramsStack.clear();
    }
    vv->functions_history.clear();
    vv->creator = NULL;
}

/*
void Variable::unchain(Variable *vv) {

    vv->functions_history.clear();
    vv->creator = NULL;
}
*/



void Variable::ones() {
    data.ones();
    grad.mul(0, grad);
}
void Variable::zeros() {
    data.mul(0, data);
    grad.mul(0, grad);
}
void Variable::randoms(float m, float a) {
    random_device rd;
    mt19937 mt(rd());
    //uniform_real_distribution<float> initd1(-a, a);
    normal_distribution<float> initd1(m, a);

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


/*
Variable operator+(const Variable &v1, const Variable &v2) {
     //vector<Variable *> inputs;
     //inputs.push_back(&v1);
     //inputs.push_back(&v2);
     Function *f = new FunctionPlus();
     //Variable vn = *f->forward(inputs);
     Variable vn = *f->forward(&v1, &v2);
     return vn;
 }

Variable operator-(const Variable &v1, const Variable &v2) {
     //vector<Variable *> inputs;
     //inputs.push_back(&v1);
     //inputs.push_back(&v2);
     Function *f = new FunctionMinus();
     //Variable vn = *f->forward(inputs);
     Variable vn = *f->forward(&v1, &v2);
     return vn;
 }

Variable operator*(const Variable &v1, const Variable &v2) {
     //vector<Variable *> inputs;
     //inputs.push_back(&v1);
     //inputs.push_back(&v2);
     Function *f = new FunctionMul();
     //Variable vn = *f->forward(inputs);
     Variable vn = *f->forward(&v1, &v2);
     return vn;
 }
*/
