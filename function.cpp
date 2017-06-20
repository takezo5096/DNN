/*
 * function.cpp
 *
 */

#include <list>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include <math.h>
#include <boost/pool/object_pool.hpp>
#include <sstream>

#include "function.h"


using namespace std;

int func_id = 0;

PVariable variable_construct_for_function(Function *f, int rows, int cols) {
    PVariable r = PVariable(variable_construct(rows, cols), variable_destroy);
    r->creator = f;

    return r;
}


// Function class //////////////////////////////////////////////////////////////
Function::Function(){
    name = "Function";
    this->id = func_id;
    func_id++;
    count_function++;
}
Function::~Function(){
    init();
    count_function--;

}
void Function::init() {

    inputs.clear();
    outputs.clear();
}



PVariable Function::forward(vector<PVariable> &inputs, vector<PVariable > &outputs){return NULL;}
void Function::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){}


PVariable Function::forward(PVariable v){

    v->forward_count++;

    inputs.push_back(v);
    PVariable r = forward(inputs, outputs);

    return r;
}


PVariable Function::forward(PVariable v1, PVariable v2){

    v1->forward_count++;
    v2->forward_count++;

    inputs.push_back(v1);
    inputs.push_back(v2);
    PVariable r = forward(inputs, outputs);

    return r;
}

PVariable Function::forward(PVariable v1, PVariable v2, PVariable v3){

    v1->forward_count++;
    v2->forward_count++;
    v3->forward_count++;

    inputs.push_back(v1);
    inputs.push_back(v2);
    inputs.push_back(v3);
    PVariable r = forward(inputs, outputs);

    return r;
}

PVariable Function::forward(PVariable v1, PVariable v2, PVariable v3, PVariable v4){
    v1->forward_count++;
    v2->forward_count++;

    inputs.push_back(v1);
    inputs.push_back(v2);
    inputs.push_back(v3);
    inputs.push_back(v4);
    PVariable r = forward(inputs, outputs);

    return r;
}

PVariable Function::forward(PVariable v1, PVariable v2, PVariable v3, PVariable v4,
                            PVariable v5, PVariable v6, PVariable v7, PVariable v8,
                            PVariable v9, PVariable v10, PVariable v11, PVariable v12
){
    v1->forward_count++;
    v2->forward_count++;


    inputs.push_back(v1);
    inputs.push_back(v2);
    inputs.push_back(v3);
    inputs.push_back(v4);
    inputs.push_back(v5);
    inputs.push_back(v6);
    inputs.push_back(v7);
    inputs.push_back(v8);
    inputs.push_back(v9);
    inputs.push_back(v10);
    inputs.push_back(v11);
    inputs.push_back(v12);
    PVariable r = forward(inputs, outputs);

    return r;
}



void Function::backward(cuMat &p_grad){

    backward(p_grad, inputs, outputs);
}




void Function::clip_grad(Variable *v){
    float clip_grad_threshold = 5.0;
    float sq = v->grad.l2();
    float rate = clip_grad_threshold/sq;
    if (rate < 1.){
        v->grad.mul(rate, v->grad);
    }
}

void Function::reset_state(){}



FunctionPlus::FunctionPlus() : Function() {
    name = "FunctionPlus";
}



PVariable FunctionPlus::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    PVariable r = variable_construct_for_function(this, v1->data.rows, v1->data.cols);

    v1->data.plus(v2->data, r->data);

    return r;
}
void FunctionPlus::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);
    //v1->grad += p_grad*1.0;
    //v2->grad += p_grad*1.0;
    if (v1->isGetGrad) p_grad.mul_plus(1.0, v1->grad);
    if (v2->isGetGrad) p_grad.mul_plus(1.0, v2->grad);
}

FunctionMinus::FunctionMinus() : Function() {
    name = "FunctionMinus";
}
PVariable FunctionMinus::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    PVariable r = variable_construct_for_function(this, v1->data.rows, v1->data.cols);

    outputs.push_back(r);

    v1->data.minus(v2->data, r->data);

    return r;

}
void FunctionMinus::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);
    //v1->grad += p_grad*1.0;
    //v2->grad += p_grad*(-1.0);
    if (v1->isGetGrad) p_grad.mul_plus(1.0, v1->grad);
    if (v2->isGetGrad) p_grad.mul_plus(-1.0, v2->grad);
}


FunctionMul::FunctionMul() : Function() {
    name = "FunctionMul";
}
PVariable FunctionMul::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    PVariable r = variable_construct_for_function(this, v1->data.rows, v1->data.cols);


    outputs.push_back(r);
    v1->data.mul(v2->data, r->data);

    return r;

}
void FunctionMul::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    //v1->grad += p_grad * v2->data;
    //v2->grad += p_grad * v1->data;
    if (v1->isGetGrad) p_grad.mul_plus(v2->data, v1->grad, 1.0, 1.0);
    if (v2->isGetGrad) p_grad.mul_plus(v1->data, v2->grad, 1.0, 1.0);
}


FunctionInverse::FunctionInverse() : Function() {
    name = "FunctionInverse";
}
PVariable FunctionInverse::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v = inputs.at(0);

    PVariable r = variable_construct_for_function(this, v->data.rows, v->data.cols);


    outputs.push_back(r);
    v->data.inverse(r->data);

    return r;

}
void FunctionInverse::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v = inputs.at(0);

    if (v->isGetGrad) v->grad += p_grad * v->data.inverse_d();
}

FunctionSqrt::FunctionSqrt() : Function() {
    name = "FunctionSqrt";
}
PVariable FunctionSqrt::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v = inputs.at(0);

    PVariable r = variable_construct_for_function(this, v->data.rows, v->data.cols);


    outputs.push_back(r);
    v->data.sqrt(r->data, 1e-8);

    return r;

}
void FunctionSqrt::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v = inputs.at(0);

    if (v->isGetGrad) v->grad += p_grad * v->data.sqrt_d();
}


FunctionSin::FunctionSin() : Function() { }
PVariable FunctionSin::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);

    PVariable r;
    r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    outputs.push_back(r);

    v1->data.sin(r->data);
    return r;
}
void FunctionSin::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);

    if (rr.get() == NULL) rr = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    v1->data.cos(rr->data);
    if (v1->isGetGrad) v1->grad += p_grad * rr->data;
}

FunctionCos::FunctionCos() : Function() { }
PVariable FunctionCos::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);

    PVariable r;
    r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    outputs.push_back(r);
    v1->data.cos(r->data);
    return r;

}
void FunctionCos::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);

    if (rr.get() == NULL) rr = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    v1->data.sin(rr->data);
    if (v1->isGetGrad) v1->grad += p_grad * rr->data * (-1.0);
}

FunctionLog::FunctionLog() : Function() {}
PVariable FunctionLog::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);

    PVariable r;
    r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    outputs.push_back(r);
    v1->data.log(r->data, 0);
    return r;

}
void FunctionLog::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);

    if (v1->isGetGrad) v1->grad += p_grad * 1.0/v1->data;
}






FunctionLinear::FunctionLinear() : Function() {
    name = "FunctionLinear";
}
FunctionLinear::FunctionLinear(Variable *w, Variable *b,  bool isTranspose) : Function() {
    name = "FunctionLinear";
    this->w  = w;
    this->b = b;
    this->isTranspose = isTranspose;
}
FunctionLinear::FunctionLinear(Variable *w,  bool isTranspose) : Function() {
    name = "FunctionLinear";
    noBias = true;
    this->w  = w;
    this->isTranspose = isTranspose;

}
FunctionLinear::FunctionLinear(int output_size, int input_size) : Function() {
    name = "FunctionLinear";

    this->w = new Variable(output_size, input_size);
    this->b = new Variable(output_size, 1);
    this->w->randoms(0., sqrt((1./(float)input_size)));

}

FunctionLinear::FunctionLinear(int output_size, int input_size, bool no_bias) : Function() {
    name = "FunctionLinear";

    noBias = no_bias;

    this->w = new Variable(output_size, input_size);
    this->w->randoms(0., sqrt((1./(float)input_size)));

    if (!noBias){
        this->b = new Variable(output_size, 1);
    }
}


void FunctionLinear::toHostArray(){
    i1.toHostArray();
    w->data.toHostArray();
    w->grad.toHostArray();
    if (!noBias){
        b->data.toHostArray();
        b->grad.toHostArray();
    }
}
void FunctionLinear::fromHostArray(){
    i1.fromHostArray();
    w->data.fromHostArray();
    w->grad.fromHostArray();
    if (!noBias){
        b->data.fromHostArray();
        b->grad.fromHostArray();
    }
}


PVariable FunctionLinear::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable x = inputs.at(0);

    int w_size = w->data.rows;
    if (isTranspose) w_size = w->data.cols;

    PVariable r = PVariable(new Variable(this, w_size, x->data.cols));



    if (i1.cols == 0 || i1.cols != x->data.cols){
        i1 = cuMat(1, x->data.cols);
        i1.ones();
    }



    if (!noBias) b->data.dot(i1, r->data);
    if (!isTranspose) w->data.dot_plus(x->data, r->data);
    else w->data.transpose().dot_plus(x->data, r->data);
    //r->data = w->data.dot(x->data) + b->data.dot(i1);

    return r;
}
void FunctionLinear::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    if (x->isGetGrad){
        if (!isTranspose) w->data.transpose_dot_plus(p_grad, x->grad);
        else w->data.transpose().transpose_dot_plus(p_grad, x->grad);
    }
    //x->grad += w->data.transpose().dot(p_grad);

    if (!isTranspose) p_grad.dot_transpose_plus(x->data, w->grad);
    else {
        cuMat tmpwg = w->grad.transpose();
        tmpwg *= 0;
        p_grad.dot_transpose_plus(x->data, tmpwg);
        w->grad += tmpwg.transpose();
    }
    //w->grad += p_grad.dot(x->data.transpose());


    if (!noBias){
        p_grad.dot_transpose_plus(i1, b->grad);
    }
    //b->grad += p_grad.dot(i1.transpose());
}



FunctionSparseLinear::FunctionSparseLinear() : Function() {
    name = "FunctionSparseLinear";
}
FunctionSparseLinear::FunctionSparseLinear(Variable *w, Variable *b, float beta, float p, Variable *ph) : Function() {
    name = "FunctionSparseLinear";
    this->w  = w;
    this->b = b;

    this->beta = beta;
    this->p = p;
    this->ph = ph;
}
FunctionSparseLinear::FunctionSparseLinear(Variable *w, float beta, float p, Variable *ph) : Function() {
    name = "FunctionSparseLinear";
    noBias = true;
    this->w  = w;

    this->beta = beta;
    this->p = p;
    this->ph = ph;
}
void FunctionSparseLinear::toHostArray(){
    i1.toHostArray();
    w->data.toHostArray();
    w->grad.toHostArray();
    if (!noBias){
        b->data.toHostArray();
        b->grad.toHostArray();
    }
}
void FunctionSparseLinear::fromHostArray(){
    i1.fromHostArray();
    w->data.fromHostArray();
    w->grad.fromHostArray();
    if (!noBias){
        b->data.fromHostArray();
        b->grad.fromHostArray();
    }
}


PVariable FunctionSparseLinear::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);
    PVariable r = PVariable(new Variable(this, w->data.rows, x->data.cols));

    if (i1.cols == 0 || i1.cols != x->data.cols){
        i1 = cuMat(1, x->data.cols);
        i1.ones();
    }

    if (!noBias) b->data.dot(i1, r->data);
    w->data.dot_plus(x->data, r->data);

    outputs.push_back(r);

    PVariable r2 = PVariable(new Variable(this, w->data.rows, x->data.cols));
    r->data.relu(r2->data);
    //r->data.sigmoid(r2->data);

    return r2;
}

void FunctionSparseLinear::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable o = outputs.at(0);
    cuMat sg = o->data.relu_d();
    //cuMat sg = o->data.sigmoid_d();


    PVariable x = inputs.at(0);

    if (x->isGetGrad){
        w->data.transpose_dot_plus(p_grad, x->grad);
    }

    // KL
    cuMat ones(ph->data.rows, ph->data.cols);
    ones.ones();
    cuMat p_tmp(ph->data.rows, ph->data.cols);
    p_tmp.ones();
    p_tmp *= p;

    cuMat kl = beta * (-1.0 * p_tmp / ph->data + (ones - p_tmp) / (ones - ph->data));

    p_grad += kl;
    p_grad *= sg;

    p_grad.dot_transpose_plus(x->data, w->grad);

    if (!noBias){
        p_grad.dot_transpose_plus(i1, b->grad);
    }


}



FunctionEmbed::FunctionEmbed() : Function() {
    name = "FunctionEmbed";
}
FunctionEmbed::FunctionEmbed(int output_size, int input_size, bool no_bias){
    name = "FunctionEmbed";

    noBias = no_bias;

    Variable w(output_size, input_size);
    this->w = w;
    this->w.randoms(0., sqrt((1./(float)input_size)));

    if (!noBias){
        Variable b(output_size, 1);
        this->b = b;
    }

}
PVariable FunctionEmbed::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    int x_cols = 0;
    if (!x->isSparse) x_cols = x->data.cols;
    else x_cols = x->data_sparse.rows;

    PVariable r = variable_construct_for_function(this, w.data.rows, x_cols);

    outputs.push_back(r);


    if (i1.cols == 0 || i1.cols != x_cols){
        i1 = cuMat(1, x_cols);
        i1.ones();
    }

    if (!noBias) b.data.dot(i1, r->data);

    if (!x->isSparse){
        w.data.dot_plus(x->data, r->data);
    }
    else{
        if (wt.rows == 0) wt = cuMat(w.data.cols, w.data.rows);
        w.data.transpose(wt);
        if (rt.rows == 0) rt = cuMat(r->data.cols, r->data.rows);
        x->data_sparse.s_d_dot(wt, rt);
        if (rtmp.rows == 0) rtmp = cuMat(r->data.rows, r->data.cols);
        rt.transpose(rtmp);
        r->data.plus(rtmp, r->data);
    }

    return r;
}
void FunctionEmbed::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable x = inputs.at(0);

    float batchSizeNorm = 1.0/((float)x->data.cols);

    if (x->isGetGrad) w.data.transpose_dot_plus(p_grad, x->grad);

    if (!x->isSparse){
        p_grad.dot_transpose_plus(x->data, w.grad);
    }
    else{
        cuMat xd = x->data_sparse.toDense();
        p_grad.dot_plus(xd, w.grad);
    }
    w.grad.mul(batchSizeNorm, w.grad);

    if (!noBias){
        p_grad.dot_transpose_plus(i1, b.grad);
        b.grad.mul(batchSizeNorm, b.grad);
    }


}
void FunctionEmbed::toHostArray(){
    i1.toHostArray();
    w.data.toHostArray();
    w.grad.toHostArray();
    if (!noBias){
        b.data.toHostArray();
        b.grad.toHostArray();
    }
}
void FunctionEmbed::fromHostArray(){
    i1.fromHostArray();
    w.data.fromHostArray();
    w.grad.fromHostArray();
    if (!noBias){
        b.data.fromHostArray();
        b.grad.fromHostArray();
    }
}



FunctionReLU::FunctionReLU() : Function() {
    name = "FunctionReLU";
}
PVariable FunctionReLU::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable x = inputs.at(0);

    PVariable r = PVariable(new Variable(this, x->data.rows, x->data.cols));

    x->data.relu(r->data);

    return r;
}
void FunctionReLU::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);


    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.relu_d(rr->data);

    if (x->isGetGrad) rr->data.mul_plus(p_grad, x->grad, 1.0, 1.0);
}

FunctionPReLU::FunctionPReLU() : Function() {
    name = "FunctionPReLU";
}
FunctionPReLU::FunctionPReLU(Variable *a) {
    name = "FunctionPReLU";
    this->a = a;
}

PVariable FunctionPReLU::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    PVariable r = variable_construct_for_function(this, x->data.rows, x->data.cols);

    outputs.push_back(r);

    x->data.prelu(a->data, r->data);

    return r;
}
void FunctionPReLU::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);


    if (xd.get() == NULL || xd->data.cols != x->data.cols){
        xd = PVariable(new Variable(x->data.rows, x->data.cols));
        ad = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.prelu_d(a->data, xd->data, ad->data);

    ad->data.mul_plus(p_grad, a->grad, 1.0, 1.0);
    if (x->isGetGrad) xd->data.mul_plus(p_grad, x->grad, 1.0, 1.0);

}


FunctionSigmoid::FunctionSigmoid() : Function() {
    name = "FunctionSigmoid";
}
PVariable FunctionSigmoid::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    PVariable r = variable_construct_for_function(this, x->data.rows, x->data.cols);


    outputs.push_back(r);
    x->data.sigmoid(r->data);

    return r;
}
void FunctionSigmoid::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);


    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }
    x->data.sigmoid_d(rr->data);

    if (x->isGetGrad) rr->data.mul_plus(p_grad, x->grad, 1.0, 1.0);
}

FunctionTanh::FunctionTanh() : Function() {
    name = "FunctionTanh";
}
PVariable FunctionTanh::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    PVariable r = variable_construct_for_function(this, x->data.rows, x->data.cols);

    outputs.push_back(r);

    x->data.tanh(r->data);

    return r;
}
void FunctionTanh::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);


    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }
    x->data.tanh_d(rr->data);

    if (x->isGetGrad) rr->data.mul_plus(p_grad, x->grad, 1.0, 1.0);
}

FunctionSoftmax::FunctionSoftmax() : Function() {
    name = "FunctionSoftmax";
}
PVariable FunctionSoftmax::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    PVariable r = PVariable(new Variable(x->data.rows, x->data.cols));

    outputs.push_back(r);

    x->data.softmax(r->data);

    return r;
}

FunctionSoftmaxCrossEntropy::FunctionSoftmaxCrossEntropy() : Function() {
    name = "FunctionSoftmaxCrossEntropy";
    loss = cuMat(1,1);
    loss.ones();

}
PVariable FunctionSoftmaxCrossEntropy::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);

    if (rr3.get() == NULL || rr3->data.cols != x->data.cols){
        rr3 = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.softmax(rr3->data);

    if (rr.get() == NULL || rr->data.cols != rr3->data.cols){
        rr = PVariable(new Variable(rr3->data));
    }
    else rr->data = rr3->data;

    rr3->data.softmax_cross_entropy(t->data, rr3->data);
    float sum = rr3->data.sum();

    sum /= rr3->data.cols;

    PVariable r = PVariable(new Variable(this, loss.rows, loss.cols));

    outputs.push_back(r);

    r->data = loss * sum;

    return r;
}
void FunctionSoftmaxCrossEntropy::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);
    PVariable y = rr;

    if (x->isGetGrad) x->grad += y->data - t->data;
}


FunctionMeanSquaredError::FunctionMeanSquaredError() : Function() {
    name = "FunctionMeanSquaredError";
    loss = cuMat(1,1);
    loss.ones();
}

PVariable FunctionMeanSquaredError::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);

    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data));
    }

    x->data.minus(t->data, rr->data);

    rr->data.mul(rr->data, rr->data);

    float sum = rr->data.sum();
    sum /= (2*rr->data.cols);

    PVariable r = variable_construct_for_function(this, loss.rows, loss.cols);

    outputs.push_back(r);

    loss.mul(sum, r->data);

    return r;
}
void FunctionMeanSquaredError::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);

    if (rr.get() == NULL || rr->data.cols != x->data.cols){

        rr = PVariable(new Variable(x->data));
    }

    x->data.minus(t->data, rr->data);
    float batch_size = rr->data.cols;
    if (x->isGetGrad) x->grad.plus(rr->data, x->grad);
}



FunctionDropout::FunctionDropout(float p) : Function() {
    name = "FunctionDropout";
    this->p = p;
}
PVariable FunctionDropout::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable x = inputs.at(0);

    PVariable r = PVariable(new Variable(this, x->data.rows, x->data.cols));

    outputs.push_back(r);

    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.dropout(r->data, rr->data, p);

    return r;
}
void FunctionDropout::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);
    PVariable idx = rr;

    if (x->isGetGrad) idx->data.mul_plus(p_grad, x->grad, 1.0, 1.0);
}


FunctionIdentity::FunctionIdentity() : Function() {
    name = "FunctionIdentity";
}

PVariable FunctionIdentity::forward(vector<PVariable> &inputs, vector<PVariable> &outputs){

    PVariable x = inputs.at(0);

    PVariable r = variable_construct_for_function(this, x->data.rows, x->data.cols);

    r->data = x->data;

    return r;
}

void FunctionIdentity::backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs){
    PVariable x = inputs.at(0);

    if (x->isGetGrad) p_grad.mul_plus(1.0, x->grad);
}



// LSTM ----------------------------------
FunctionLSTM::FunctionLSTM() : Function() {
    name = "FunctionLSTM";
}

PVariable FunctionLSTM::forward(vector<PVariable> &inputs, vector<PVariable> &outputs){

    PVariable x = inputs.at(0);
    PVariable c = inputs.at(1);
    PVariable c_next = inputs.at(2);

    int offset = x->data.rows/4;

    this->i = x->data.sliceRows(0, offset);
    this->f = x->data.sliceRows(offset, offset);
    this->g = x->data.sliceRows(offset*2, offset);
    this->o = x->data.sliceRows(offset*3, offset);

    cuMat _i = this->i;
    cuMat _f = this->f;
    cuMat _g = this->g;
    cuMat _o = this->o;

    this->i.sigmoid(_i);
    this->f.sigmoid(_f);
    this->g.tanh(_g);
    this->o.sigmoid(_o);

    c_next->data = _g * _i + _f * c->data;

    cuMat tmp = c_next->data;
    c_next->data.tanh(tmp);

    PVariable r = variable_construct_for_function(this, x->data.rows, x->data.cols);


    r->data = _o * tmp;

    return r;
}


void FunctionLSTM::backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs){

    PVariable x = inputs.at(0);
    PVariable c = inputs.at(1);
    PVariable c_next = inputs.at(2);

    int offset = x->data.rows/4;

    cuMat co = c_next->data.tanh();

    c->grad = gh * this->o * co.tanh_d() + c_next->grad;

    cuMat gg = c->grad * this->i * this->g.tanh_d();

    cuMat gi = c->grad * this->g * this->i.sigmoid_d();

    cuMat gf = c->grad * c->data * this->f.sigmoid_d();

    cuMat go = gh * co * this->o.sigmoid_d();

    c->grad *= this->f;

    cuMat tmp = x->grad;
    tmp.joinRows(gi, 0, offset);
    tmp.joinRows(gf, offset, offset);
    tmp.joinRows(gg, offset*2, offset);
    tmp.joinRows(go, offset*3, offset);
    if (x->isGetGrad) x->grad += tmp;
}


//FullLSTM
FunctionFullLSTM::FunctionFullLSTM(
        Variable *f_c_w, Variable *f_h_w, Variable *f_x_w, Variable *f_x_b,
        Variable *i_c_w, Variable *i_h_w, Variable *i_x_w, Variable *i_x_b,
        Variable *o_c_w, Variable *o_h_w, Variable *o_x_w, Variable *o_x_b,
        Variable *g_h_w, Variable *g_x_w, Variable *g_x_b) : Function(){

    name = "FunctionFullLSTM";

    this->f_c_w = f_c_w; this->f_h_w = f_h_w; this->f_x_w = f_x_w; this->f_x_b = f_x_b;
    this->i_c_w = i_c_w; this->i_h_w = i_h_w; this->i_x_w = i_x_w; this->i_x_b = i_x_b;
    this->o_c_w = o_c_w; this->o_h_w = o_h_w; this->o_x_w = o_x_w; this->o_x_b = o_x_b;
    this->g_h_w = g_h_w; this->g_x_w = g_x_w; this->g_x_b = g_x_b;
}



PVariable FunctionFullLSTM::forward(vector<PVariable> &inputs, vector<PVariable> &outputs) {

    PVariable x = inputs.at(0);
    PVariable h = inputs.at(1);
    PVariable c = inputs.at(2);
    PVariable c_next = inputs.at(3);

    cuMat ones(1, x->data.cols);
    ones.ones();

    f_hat = f_c_w->data.dot(c->data) + f_h_w->data.dot(h->data) + f_x_w->data.dot(x->data) + f_x_b->data.dot(ones);
    f = f_hat.sigmoid();
    i_hat = i_c_w->data.dot(c->data) + i_h_w->data.dot(h->data) + i_x_w->data.dot(x->data) + i_x_b->data.dot(ones);
    i = i_hat.sigmoid();
    g_hat = g_h_w->data.dot(h->data) + g_x_w->data.dot(x->data) + g_x_b->data.dot(ones);
    g = g_hat.tanh();

    c_next->data = c->data * f + i * g;

    o_hat = o_c_w->data.dot(c_next->data) + o_h_w->data.dot(h->data) + o_x_w->data.dot(x->data) + o_x_b->data.dot(ones);
    o = o_hat.sigmoid();

    PVariable h_next = variable_construct_for_function(this, f_x_w->data.rows, x->data.cols);


    h_next->data = c_next->data.tanh() * o;

    return h_next;
}

void FunctionFullLSTM::backward(cuMat &delta_h, vector<PVariable> &inputs, vector<PVariable> &outputs) {

    PVariable x = inputs.at(0);
    PVariable h = inputs.at(1);
    PVariable c = inputs.at(2);
    PVariable c_next = inputs.at(3);
    PVariable f_for_grad = inputs.at(4);
    PVariable f_next_for_grad = inputs.at(5);
    PVariable i_for_grad = inputs.at(6);
    PVariable i_next_for_grad = inputs.at(7);
    PVariable o_for_grad = inputs.at(8);
    PVariable o_next_for_grad = inputs.at(9);
    PVariable g_for_grad = inputs.at(10);
    PVariable g_next_for_grad = inputs.at(11);

    cuMat ones(1, x->data.cols);
    ones.ones();

    cuMat delta_o = delta_h *  c_next->data.tanh() * o_hat.sigmoid_d();
    c_next->grad += delta_h * o * c_next->data.tanh_d() + o_c_w->data.transpose().dot(delta_o);

    cuMat delta_i = c_next->grad * g * i_hat.sigmoid_d();
    cuMat delta_f = c_next->grad * c->data * f_hat.sigmoid_d();
    cuMat delta_g = c_next->grad * i * g_hat.tanh_d();


    i_for_grad->grad = delta_i;
    f_for_grad->grad = delta_f;
    o_for_grad->grad = delta_o;
    g_for_grad->grad = delta_g;


    c->grad = c_next->grad * f + i_c_w->data.transpose().dot(delta_i) + f_c_w->data.transpose().dot(delta_f);


    o_c_w->grad += delta_o.dot(c_next->grad.transpose());
    i_c_w->grad += i_next_for_grad->grad.dot(c_next->grad.transpose());
    f_c_w->grad += f_next_for_grad->grad.dot(c_next->grad.transpose());


    x->grad += g_x_w->data.transpose().dot(delta_g)
               + i_x_w->data.transpose().dot(delta_i)
               + f_x_w->data.transpose().dot(delta_f)
               + o_x_w->data.transpose().dot(delta_o);


    h->grad += g_h_w->data.transpose().dot(delta_g)
               + i_h_w->data.transpose().dot(delta_i)
               + f_h_w->data.transpose().dot(delta_f)
               + o_h_w->data.transpose().dot(delta_o);


    g_x_w->grad += delta_g.dot(x->data.transpose());
    g_h_w->grad += g_next_for_grad->grad.dot(h->data.transpose());
    i_x_w->grad += delta_i.dot(x->data.transpose());
    i_h_w->grad += i_next_for_grad->grad.dot(h->data.transpose());
    f_x_w->grad += delta_f.dot(x->data.transpose());
    f_h_w->grad += f_next_for_grad->grad.dot(h->data.transpose());
    o_x_w->grad += delta_o.dot(x->data.transpose());
    o_h_w->grad += o_next_for_grad->grad.dot(h->data.transpose());


    g_x_b->grad += delta_g.dot(ones.transpose());
    i_x_b->grad += delta_i.dot(ones.transpose());
    f_x_b->grad += delta_f.dot(ones.transpose());
    o_x_b->grad += delta_o.dot(ones.transpose());
}



FunctionGRU::FunctionGRU(Variable *w_r, Variable *u_r, Variable *b_r,
                         Variable *w_z, Variable *u_z, Variable *b_z,
                         Variable *w_g, Variable *u_g, Variable *b_g){
    this->w_r = w_r;
    this->u_r = u_r;
    this->b_r = b_r;
    this->w_z = w_z;
    this->u_z = u_z;
    this->b_z = b_z;
    this->w_g = w_g;
    this->u_g = u_g;
    this->b_g = b_g;

    name = "FunctionGRU";
}

PVariable FunctionGRU::forward(vector<PVariable> &inputs, vector<PVariable> &outputs) {
    PVariable x = inputs[0];
    PVariable h = inputs[1];

    cuMat ones(w_z->data.rows, x->data.cols);
    ones.ones();
    cuMat ones_b(1, x->data.cols);
    ones_b.ones();


    r_hat = w_r->data.dot(h->data) + u_r->data.dot(x->data) + b_r->data.dot(ones_b);
    r = r_hat.sigmoid();
    z_hat = w_z->data.dot(h->data) + u_z->data.dot(x->data) + b_z->data.dot(ones_b);
    z = z_hat.sigmoid();
    g_hat = w_g->data.dot(h->data * r) + u_g->data.dot(x->data) + b_g->data.dot(ones_b);
    g = g_hat.tanh();

    PVariable h_new = variable_construct_for_function(this, w_r->data.rows, x->data.cols);


    h_new->data = h->data * (ones - z) + z * g;

    return h_new;
}

void FunctionGRU::backward(cuMat &delta_h, vector<PVariable> &inputs, vector<PVariable> &outputs) {
    PVariable x = inputs[0];
    PVariable h = inputs[1];

    cuMat zeros(w_z->data.rows, x->data.cols);
    //ones.ones();
    cuMat ones_b(1, x->data.cols);
    ones_b.ones();

    cuMat delta4 = (zeros - z) * delta_h;
    cuMat delta5 = delta_h * h->data;
    cuMat delta6 = zeros - delta5;
    cuMat delta7 = delta_h * g;
    cuMat delta8 = delta_h * z;

    cuMat delta9 = delta6 + delta7;

    cuMat delta10 = delta8 * g_hat.tanh_d();
    cuMat delta11 = delta9 * z_hat.sigmoid_d();


    cuMat delta12 = u_g->data.transpose().dot(delta10);
    cuMat delta13 = w_g->data.transpose().dot(delta10);
    cuMat delta14 = u_z->data.transpose().dot(delta11);
    cuMat delta15 = w_z->data.transpose().dot(delta11);

    cuMat delta16 = delta13 * h->data;
    cuMat delta17 = delta13 * r;
    cuMat delta18 = delta16 * r_hat.sigmoid_d();
    cuMat delta19 = delta17 + delta4;
    cuMat delta20 = u_r->data.transpose().dot(delta18);
    cuMat delta21 = w_r->data.transpose().dot(delta18);
    cuMat delta22 = delta21 + delta15;
    h->grad += delta19 + delta22;
    x->grad += delta12 + delta14 + delta20;

    w_r->grad += delta18.dot(h->data.transpose());
    u_r->grad += delta18.dot(x->data.transpose());
    w_z->grad += delta11.dot(h->data.transpose());
    u_z->grad += delta11.dot(x->data.transpose());

    w_g->grad += delta10.dot((h->data * r).transpose());
    u_g->grad += delta10.dot(x->data.transpose());

    b_r->grad += delta18.dot(ones_b.transpose());
    b_z->grad += delta11.dot(ones_b.transpose());
    b_g->grad += delta10.dot(ones_b.transpose());

}


FunctionBatchNorm::FunctionBatchNorm(int element_size, int channel_num, Variable *gamma, Variable *beta, Variable *x_mean, Variable *x_var) {
    this->gamma = gamma;
    this->beta = beta;

    this->x_mean = x_mean;
    this->x_var = x_var;

    this->element_size = element_size;
    this->channel_num = channel_num;

    xhat.resize(channel_num);
    rmu.resize(channel_num);
    xmu.resize(channel_num);
    ivar.resize(channel_num);
    sqrtvar.resize(channel_num);
    var.resize(channel_num);

}


PVariable FunctionBatchNorm::forward(vector<PVariable> &inputs, vector<PVariable> &outputs) {

    PVariable x_org = inputs[0];

    PVariable r = variable_construct_for_function(this, x_org->data.rows, x_org->data.cols);

    for(int i=0; i<channel_num; i++) {

        int idx = i*element_size;
        cuMat x_data = x_org->data.sliceRows(idx, element_size);

        int N = x_data.cols;
        int D = x_data.rows;

        cuMat ones(D, N);
        ones.ones();


        //step 1
        if (is_train) rmu[i] = 1.0 / N * x_data.batch_sum();
        else{
            rmu[i] = cuMat(element_size, 1);
            rmu[i].memSetDevice(this->x_mean->data.mDevice + idx);
        }
        cuMat mu = rmu[i].vec_to_mat(N);

        //step 2
        xmu[i] = x_data - mu;

        //step 3
        cuMat sq = xmu[i] * xmu[i];

        //step 4
        if (is_train) var[i] = 1.0 / N * sq.batch_sum();
        else {
            var[i] = cuMat(element_size, 1);
            var[i].memSetDevice(x_var->data.mDevice + idx);
            var[i] = ((float) N) / (((float) N) - 1.0) * var[i];
        } //use unbiased variance

        //step 5
        sqrtvar[i] = var[i].sqrt();

        //step 6
        ivar[i] = sqrtvar[i].inverse();
        cuMat tmp = ivar[i].vec_to_mat(N);

        //step 7
        xhat[i] = xmu[i] * tmp;

        //step 8
        cuMat gamma_tmp(element_size, 1);
        gamma_tmp.memSetDevice(gamma->data.mDevice + idx);
        cuMat gammax = xhat[i].mat_vec_mul(gamma_tmp, 0);

        //step 9
        cuMat beta_tmp(element_size, 1);
        beta_tmp.memSetDevice(beta->data.mDevice + idx);
        cuMat r_c = gammax + ones.mat_vec_mul(beta_tmp, 0);
        r->data.joinRows(r_c, idx, element_size);

    }

    return r;
}

void FunctionBatchNorm::backward(cuMat &dout_org, vector<PVariable> &inputs, vector<PVariable> &outputs) {

    PVariable x = inputs[0];

    for(int i=0; i<channel_num; i++) {

        int idx = i*element_size;
        cuMat dout = dout_org.sliceRows(idx, element_size);

        int N = dout.cols;
        int D = dout.rows;


        //step 9
        beta->grad.memSetDeviceRow(dout.batch_sum().mDevice, i);
        cuMat dgammax = dout;

        //step 8
        cuMat tmp = dgammax * xhat[i];
        gamma->grad.memSetDeviceRow(tmp.batch_sum().mDevice, i);
        cuMat gamma_tmp(element_size, 1);
        gamma_tmp.memSetDevice(gamma->data.mDevice + idx);
        cuMat dxhat = dgammax.mat_vec_mul(gamma_tmp, 0);

        //step 7
        tmp = dxhat * xmu[i];
        cuMat divar = tmp.batch_sum();
        cuMat dxmu1 = dxhat.mat_vec_mul(ivar[i], 0);

        //step 6
        //tmp = sqrtvar.inverse_d();
        tmp = -1.0 * sqrtvar[i].inverse() * sqrtvar[i].inverse();
        cuMat dsqrtvar = tmp * divar;

        //step 5
        cuMat dvar = var[i].sqrt_d() * dsqrtvar;

        //step 4
        cuMat dsq = 1.0 / N * dvar.vec_to_mat(N);

        //step 3
        cuMat dxmu2 = 2.0 * xmu[i] * dsq;

        //step 2
        cuMat dx1 = dxmu1 + dxmu2;
        cuMat dmu = -1.0 * dx1.batch_sum();

        //step 1
        cuMat dx2 = 1.0 / N * dmu.vec_to_mat(N);

        //step0
        cuMat dx3 = dx1 + dx2;
        x->grad.joinRows(dx3, idx, element_size);
    }
}


FunctionConv2D::FunctionConv2D(Variable *w, Variable *b, int batch_num, int channel_num, int w_size, int h_size, int filter_size, int filter_num, int stride, int padding){

    this->batch_num = batch_num;
    this->channel_num = channel_num;
    this->w_size = w_size;
    this->h_size = h_size;
    this->filter_size = filter_size;
    this->filter_num = filter_num;
    this->stride = stride;
    this->padding = padding;

    this->w = w;
    this->b = b;

    this->name = "FunctionConv2D";

    /**
    * Each dimension h and w of the output images is computed as followed:
    * outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride
    */
    this->outputDim_w = 1 + (w_size + (padding+padding) - filter_size) / stride;
    this->outputDim_h = 1 + (h_size + (padding+padding) - filter_size) / stride;

    ones = new Variable(this->outputDim_w * this->outputDim_h, 1);
    ones->ones();
}

FunctionConv2D::~FunctionConv2D(){
    delete ones;
}

cuMat FunctionConv2D::forward_one(cuMat &data){

    int output_dim_w, output_dim_h;

    cuMat stacked = data.im2col(w_size, h_size, channel_num, filter_size, filter_size, stride, stride, padding, padding, padding, padding, output_dim_w, output_dim_h);

    cols.push_back(stacked);

    cuMat r = stacked.dot(w->data.transpose()) + ones->data.dot(b->data.transpose());

    return r;
}


cuMat FunctionConv2D::backward_one(cuMat &col, cuMat &p_grad) {

    cuMat p_grad_t = p_grad.transpose();

    w->grad += p_grad_t.dot(col);

    b->grad += p_grad_t.dot(ones->data);

    cuMat dcol = p_grad.dot(w->data);

    cuMat dx = dcol.col2im(w_size, h_size, channel_num, filter_size, filter_size, stride, stride, padding, padding, padding, padding);

    return dx;
}



PVariable FunctionConv2D::forward(vector<PVariable> &inputs, vector<PVariable> &outputs){

    PVariable x = inputs[0];

    PVariable r = PVariable(new Variable(this, filter_num * outputDim_w * outputDim_h, batch_num));


    for(int i=0; i<batch_num; i++) {
        int data_index = i*(channel_num * w_size * h_size);
        float *one_m = x->data.mDevice + data_index;
        cuMat one_m_dev(w_size * h_size, channel_num);
        one_m_dev.memSetDevice(one_m);
        cuMat r_array = forward_one(one_m_dev);
        r->data.memSetDeviceCol(r_array.mDevice, i);
    }
    return r;
}

void FunctionConv2D::backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs) {

    PVariable x = inputs[0];

    cuMat dx(channel_num * w_size * h_size, batch_num);

    for(int i=0; i<batch_num; i++) {

        int data_index = i*(filter_num * outputDim_w * outputDim_h);

        float *p_grad_one = p_grad.mDevice + data_index;

        cuMat p_grad_one_dev(outputDim_w * outputDim_h, filter_num);
        p_grad_one_dev.memSetDevice(p_grad_one);

        cuMat r_array = backward_one(cols[i], p_grad_one_dev);

        dx.memSetDeviceCol(r_array.mDevice, i);
    }

    x->grad += dx;
}


FunctionPooling::FunctionPooling(int width, int height, int depth, int windowWidth, int windowHeight, int stride, int padding){

    name = "FunctionPooling";

    this->width = width;
    this->height = height;
    this->depth = depth;
    this->windowWidth = windowWidth;
    this->windowHeight = windowHeight;
    this->stride = stride;
    this->padding = padding;
}

PVariable FunctionPooling::forward(vector<PVariable> &inputs, vector<PVariable> &outputs){
    PVariable x = inputs[0];

    int batch_num = x->data.cols;

    //* Pooling size as followed:
    //* outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;

    int pooled_w = 1 + (width + (padding+padding) - windowWidth)/stride;
    int pooled_h = 1 + (height + (padding+padding) - windowHeight)/stride;

    PVariable r = PVariable(new Variable(this, depth * pooled_w * pooled_h, batch_num));

    r->data = x->data.pooling(batch_num, width, height, depth, windowWidth, windowHeight, stride, stride, padding, padding, padding, padding);
    return r;
}

void FunctionPooling::backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs){

    PVariable x = inputs[0];

    int batch_num = x->data.cols;

    int pooled_w = 1 + (width + (padding+padding) - windowWidth)/stride;
    int pooled_h = 1 + (height + (padding+padding) - windowHeight)/stride;

    x->grad = x->data.pooling_backward(batch_num, p_grad.mDevice, width, height, depth, windowWidth, windowHeight, stride, stride, padding, padding, padding, padding);
}
