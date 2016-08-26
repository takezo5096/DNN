/*
 * function.cpp
 *
 *  Created on: 2015/12/25
 *      Author: takeshi.fujita
 */

#include <list>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include <math.h>
#include <boost/pool/object_pool.hpp>

#include "function.h"

using namespace std;

int func_id = 0;

int count_pool = 0;

/*
boost::object_pool<Variable> obj_pool;


void
obj_destroy(Variable *ptr)
{
    if (count_pool > 200){
        obj_pool.destroy(ptr);
        count_pool--;
    }
}
*/

map<Variable *, bool> obj_pool2;

Variable *obj_construct(Function *f, int rows, int cols){
    for(auto itr = obj_pool2.begin(); itr != obj_pool2.end(); ++itr) {
        if (!itr->second){
            Variable *v = (Variable *)itr->first;
            if (v->creator == f && v->data.rows == rows && v->data.cols == cols){
                obj_pool2[v] = true;
                return v;
            }
        }
    }

    Variable *r = new Variable(f, rows, cols);
    obj_pool2[r] = true;
    //cout << "obj_construct obj_pool2.size():" << obj_pool2.size() << endl;

    return r;
}
void obj_destroy(Variable *ptr){
    //cout << "obj_destroy count_pool:" << count_pool << endl;
    obj_pool2[ptr] = false;
    //cout << "obj_destroy obj_pool2.size():" << obj_pool2.size() << endl;
    if (obj_pool2.size() > 2000){
        obj_pool2.erase(ptr);
        delete ptr;
    }
}


// Function class //////////////////////////////////////////////////////////////
Function::Function(){
    this->id = func_id;
    func_id++;

}
Function::~Function(){
    init();

    for(FunctionParam *p : paramsStack){
        if (p != NULL) delete p;
    }
}
void Function::init() {

    for (FunctionParam *p : paramsStack) {
        for (PVariable v : p->inputs) {
            //v->zeros();
            if (v->data.rows != 0) v->data *= 0;
            v->grad *= 0;

            if (v->data_sparse.rows != 0) v->data_sparse.zeros();
        }
        for (PVariable v : p->outputs) {
            //v->zeros();
            if (v->data.rows != 0) v->data *= 0;
            v->grad *= 0;

            if (v->data_sparse.rows != 0) v->data_sparse.zeros();
        }
    }
}


void Function::createParams(vector<PVariable > &inputs, vector<Function *> &funcs){

    bool found = false;
    for(PVariable v : inputs){
        for (Function *f : v->functions_history){
            bool hit = false;
            for(Function *vf : funcs){
                if (f == vf){
                    hit = true;
                    break;
                }
            }
            if (!hit) {
                funcs.push_back(f);
            }
        }
    }
    for (Function *f : funcs){
        if (f == this){
            found = true;
            break;
        }
    }
    if (!found) funcs.push_back(this);


    //if (found || paramsStack.size() == 0){
        FunctionParam *p = new FunctionParam();
        //for(PVariable v : inputs){
        //    p->inputs.push_back(v);
        //}
        p->inputs = inputs;
        paramsStack.push_back(p);

        paramsStackNums.push_back(paramsStack.size()-1);
        //cout << "Function::createParams:" << (paramsStack.size()-1) << endl;
/*
    }
    else{
        FunctionParam *p = paramsStack.back();
        p->inputs = inputs;
        //for(PVariable v : inputs){
        //    p->inputs.push_back(v);
        //}

    }
*/

}

PVariable Function::forward(vector<PVariable> &inputs, vector<PVariable > &outputs){return NULL;}
void Function::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){}


PVariable Function::forward(PVariable v){

    vector<PVariable > inputs;
    inputs.push_back(v);

    vector<Function *> funcs;
    Function::createParams(inputs, funcs);

    FunctionParam *p = paramsStack.back();
    PVariable r = forward(p->inputs, p->outputs);


    if (!paramsStack.empty()){
        r->functions_history.clear();
        for(Function *f : funcs) r->functions_history.push_back(f);
    }
    return r;
}


PVariable Function::forward(PVariable v1, PVariable v2){


    vector<PVariable > inputs;
    inputs.push_back(v1);
    inputs.push_back(v2);

    vector<Function *> funcs;
    Function::createParams(inputs, funcs);

    FunctionParam *p = paramsStack.back();
    PVariable r = forward(p->inputs, p->outputs);
   if (!paramsStack.empty()){
        r->functions_history.clear();
        for(Function *f : funcs) r->functions_history.push_back(f);
    }
    return r;
}
void Function::backward(cuMat &p_grad){

    if (paramsStackNums.size() == 0) return;

    int backwordNums = paramsStackNums.back();


    //cout << "Function::backward 1 backwordNums:" << backwordNums << " paramsStackNums.size():" << paramsStackNums.size() << " paramsStack.size:" << paramsStack.size() << endl;
    FunctionParam *p = paramsStack.at(backwordNums);
    //cout << "Function::backward 2" << endl;


    //clip gradient
/*
    float l2 = p_grad.l2();
    float threshold = 5.0;
    float rate = threshold/l2;
    if (rate < 1) p_grad.mul(rate, p_grad);
*/

    backward(p_grad, p->inputs, p->outputs);
    //cout << "Function::backward 3" << endl;
}

bool Function::popParamStack(){
    if (paramsStackNums.size() == 0) return false;

    //if (paramsStack.size() > 1){
    int backwordNums = paramsStackNums.back();
    if (backwordNums == 0){
        paramsStackNums.pop_back();
        return true;
    }
    else{
        paramsStackNums[paramsStackNums.size()-1] = backwordNums -1;
        return false;
    }
    //}
}

/*
void Function::clearParamStack(bool isPop){

    //int backwordNums = paramsStackNums.back();
    //cout << "Function::popParamStack backwordNums:" << backwordNums  << " paramsStackNums.size():" << paramsStackNums.size() << endl;
    if (isPop){
        //cout << "########1 Function::popParamStack id:" << this->id << " paramsStackNums.size():" << paramsStackNums.size() << endl;
        //cout << "########2 Function::popParamStack id:" << this->id << " paramsStack.size():" << paramsStack.size() << endl;
        FunctionParam *p  = paramsStack.back();
        if (p != NULL){
            delete p; p = NULL;
        }
        //paramsStack.pop_back();
        //cout << "aaa" << endl;
    }
}
*/


void Function::reset_state(){}



FunctionPlus::FunctionPlus() : Function() {
    name = "FunctionPlus";
}



PVariable FunctionPlus::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    //PVariable r;
    //if (outputs.empty()){

    //r = PVariable(Function::obj_pool.construct(this, v1->data.rows, v1->data.cols), Function::obj_destroy);
        //r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
        PVariable r = PVariable(obj_construct(this, v1->data.rows, v1->data.cols), obj_destroy); //custom

        outputs.push_back(r);
    //}
    //else r = outputs.back();

    v1->data.plus(v2->data, r->data);

    return r;
}
void FunctionPlus::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    //cout << "****************************** FunctionPlus::backward id:" << this->id << endl;
    //cout << p_grad << endl;

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);
    //v1->grad += p_grad*1.0;
    //v2->grad += p_grad*1.0;
    p_grad.mul_plus(1.0, v1->grad);
    p_grad.mul_plus(1.0, v2->grad);
}

FunctionMinus::FunctionMinus() : Function() {
    name = "FunctionMinus";
}
PVariable FunctionMinus::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    //PVariable r;
    //if (outputs.empty()){
        //r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
        PVariable r = PVariable(obj_construct(this, v1->data.rows, v1->data.cols), obj_destroy); //custom
        outputs.push_back(r);
    //}
    //else r = outputs.back();

    v1->data.minus(v2->data, r->data);

    return r;

}
void FunctionMinus::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);
    //v1->grad += p_grad*1.0;
    //v2->grad += p_grad*(-1.0);
    p_grad.mul_plus(1.0, v1->grad);
    p_grad.mul_plus(-1.0, v2->grad);
}


FunctionMul::FunctionMul() : Function() {
    name = "FunctionMul";
}
PVariable FunctionMul::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    //PVariable r;
    //if (outputs.empty()){
        //r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
        PVariable r = PVariable(obj_construct(this, v1->data.rows, v1->data.cols), obj_destroy); //custom

        outputs.push_back(r);
    //}
    //else r = outputs.back();
    v1->data.mul(v2->data, r->data);

    return r;

}
void FunctionMul::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);
    PVariable v2 = inputs.at(1);

    //v1->grad += p_grad * v2->data;
    //v2->grad += p_grad * v1->data;
    p_grad.mul_plus(v2->data, v1->grad, 1.0, 1.0);
    p_grad.mul_plus(v1->data, v2->grad, 1.0, 1.0);
}


FunctionSin::FunctionSin() : Function() { }
PVariable FunctionSin::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);

    PVariable r;
    //if (outputs.empty()){
        r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
        outputs.push_back(r);
    //}
    //else r = outputs.back();

    v1->data.sin(r->data);
    return r;
}
void FunctionSin::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);

    if (rr.get() == NULL) rr = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    v1->data.cos(rr->data);
    v1->grad += p_grad * rr->data;
}

FunctionCos::FunctionCos() : Function() { }
PVariable FunctionCos::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);

    PVariable r;
    //if (outputs.empty()){
        r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
        outputs.push_back(r);
    //}
    //else r = outputs.back();
    v1->data.cos(r->data);
    return r;

}
void FunctionCos::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);

    if (rr.get() == NULL) rr = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
    v1->data.sin(rr->data);
    v1->grad += p_grad * rr->data * (-1.0);
}

FunctionLog::FunctionLog() : Function() {}
PVariable FunctionLog::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    PVariable v1 = inputs.at(0);

    PVariable r;
    //if (outputs.empty()){
        r = PVariable(new Variable(this, v1->data.rows, v1->data.cols));
        outputs.push_back(r);
    //}
    //else r = outputs.back();
    v1->data.log(r->data, 0);
    return r;

}
void FunctionLog::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable v1 = inputs.at(0);

    v1->grad += p_grad * 1.0/v1->data;
}


FunctionGRU::FunctionGRU() : Function() {
    name = "FunctionGRU";
}

FunctionGRU::FunctionGRU(int output_size, int input_size) : Function() {

    name = "FunctionLinear";

    this->output_size = output_size;
    this->input_size = input_size;


    f_sigmoid_r = new FunctionSigmoid();
    f_plus_r = new FunctionPlus();
    wr_x = new FunctionLinear(output_size, input_size, true);
    ur_h = new FunctionLinear(output_size, output_size, true);

    f_sigmoid_z = new FunctionSigmoid();
    f_plus_z = new FunctionPlus();
    wz_x = new FunctionLinear(output_size, input_size, true);
    uz_h = new FunctionLinear(output_size, output_size, true);

    f_mul_h = new FunctionMul();
    f_tanh = new FunctionTanh();
    //f_tanh = new FunctionReLU();
    f_plus_h = new FunctionPlus();

    w_x = new FunctionLinear(output_size, input_size, true);
    u_h = new FunctionLinear(output_size, output_size, true);

    f_plus = new FunctionPlus();
    f_mul1 = new FunctionMul();
    f_mul2 = new FunctionMul();
    f_minus = new FunctionMinus();


}
FunctionGRU::~FunctionGRU(){

    delete f_sigmoid_r;
    delete f_plus_r;
    delete wr_x;
    delete ur_h;

    delete f_sigmoid_z;
    delete f_plus_z;
    delete wz_x;
    delete uz_h;

    delete f_mul_h;
    delete f_tanh;
    delete f_plus_h;

    delete w_x;
    delete u_h;

    delete f_plus;
    delete f_mul1;
    delete f_mul2;
    delete f_minus;

}
PVariable FunctionGRU::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable x = inputs[0];


    if (!s_h){
        s_h = PVariable(new Variable(output_size, x->data.cols));
        //cout << "FunctionGRU::forward reset s_h" << endl;
    }



    PVariable r = f_sigmoid_r->forward(f_plus_r->forward(wr_x->forward(x), ur_h->forward(s_h)));
    PVariable z = f_sigmoid_z->forward(f_plus_z->forward(wz_x->forward(x), uz_h->forward(s_h)));

    PVariable r_h = f_mul_h->forward(r, s_h);

    PVariable h_h = f_tanh->forward(f_plus_h->forward(w_x->forward(x), u_h->forward(r_h)));

    PVariable e2 = PVariable(new Variable(z->data.rows, z->data.cols));
    e2->ones();
    s_h = f_plus->forward(f_mul1->forward(f_minus->forward(e2, z), s_h), f_mul2->forward(z, h_h));

    for(FunctionParam *p : paramsStack) delete p;
    paramsStack.clear();
    paramsStackNums.clear();

    return s_h;
}
void FunctionGRU::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
}

void FunctionGRU::reset_state(){
    s_h.reset();
}



FunctionLSTM::FunctionLSTM() : Function(){
    name = "FunctionLSTM";
}

FunctionLSTM::FunctionLSTM(int output_size, int input_size) : Function(){
    name = "FunctionLSTM";

    this->output_size = output_size;
    this->input_size = input_size;

    x_i = new FunctionLinear(output_size, input_size, true);
    w_i = new FunctionLinear(output_size, output_size, true);
    c_i = new FunctionLinear(output_size, output_size, true);
    f_plus_i = new FunctionPlus();
    f_plus_i2 = new FunctionPlus();
    f_sig_i = new FunctionSigmoid();

    x_f = new FunctionLinear(output_size, input_size, true);
    w_f = new FunctionLinear(output_size, output_size, true);
    c_f = new FunctionLinear(output_size, output_size, true);
    f_plus_f = new FunctionPlus();
    f_plus_f2 = new FunctionPlus();
    f_sig_f = new FunctionSigmoid();

    x_o = new FunctionLinear(output_size, input_size, true);
    w_o = new FunctionLinear(output_size, output_size, true);
    c_o = new FunctionLinear(output_size, output_size, true);
    f_plus_o = new FunctionPlus();
    f_plus_o2 = new FunctionPlus();
    f_sig_o = new FunctionSigmoid();

    x_g = new FunctionLinear(output_size, input_size, true);
    w_g = new FunctionLinear(output_size, output_size, true);
    f_plus_g = new FunctionPlus();
    f_tan_g = new FunctionTanh();

    f_mul1_c = new FunctionMul();
    f_mul2_c = new FunctionMul();
    f_plus_c = new FunctionPlus();

    f_mul_s = new FunctionMul();
    f_tan_s = new FunctionTanh();

}

FunctionLSTM::~FunctionLSTM(){
    delete x_i;
    delete w_i;
    delete c_i;
    delete f_plus_i;
    delete f_plus_i2;
    delete f_sig_i;

    delete x_f;
    delete w_f;
    delete c_f;
    delete f_plus_f;
    delete f_plus_f2;
    delete f_sig_f;

    delete x_o;
    delete w_o;
    delete c_o;
    delete f_plus_o;
    delete f_plus_o2;
    delete f_sig_o;

    delete x_g;
    delete w_g;
    delete f_plus_g;
    delete f_tan_g;

    delete f_mul1_c;
    delete f_mul2_c;
    delete f_plus_c;

    delete f_mul_s;
    delete f_tan_s;

}

PVariable FunctionLSTM::forward(vector<PVariable> &inputs, vector<PVariable> &outputs){

    PVariable x = inputs[0];

    if (!s_h){
        s_h = PVariable(new Variable(output_size, x->data.cols));
        //cout << "FunctionLSTM::forward reset s_h output_size:" << output_size << " x->data.cols:" << x->data.cols  << endl;
    }
    if (!c_h){
        c_h = PVariable(new Variable(output_size, x->data.cols));
        //cout << "FunctionLSTM::forward reset c_h" << endl;
    }

    PVariable i = f_sig_i->forward(f_plus_i2->forward(c_h, f_plus_i->forward(x_i->forward(x), w_i->forward(s_h))));
    PVariable f = f_sig_f->forward(f_plus_f2->forward(c_h, f_plus_f->forward(x_f->forward(x), w_f->forward(s_h))));

    PVariable g = f_tan_g->forward(f_plus_g->forward(x_g->forward(x), w_g->forward(s_h)));

    c_h = f_plus_c->forward(f_mul1_c->forward(c_h, f), f_mul2_c->forward(g, i));

    PVariable o = f_sig_o->forward(f_plus_o2->forward(c_h, f_plus_o->forward(x_o->forward(x), w_o->forward(s_h))));

    s_h = f_mul_s->forward(f_tan_s->forward(c_h), o);

    for(FunctionParam *p : paramsStack) delete p;
    paramsStack.clear();
    paramsStackNums.clear();

    return s_h;
}
void FunctionLSTM::backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs){
}
void FunctionLSTM::reset_state(){
    s_h.reset();
    c_h.reset();
}




FunctionLinear::FunctionLinear() : Function() {
    name = "FunctionLinear";
}
FunctionLinear::FunctionLinear(Variable &w, Variable &b) : Function() {
    this->w  = w;
    this->b = b;
}
FunctionLinear::FunctionLinear(int output_size, int input_size) : Function() {
    name = "FunctionLinear";

    Variable w(output_size, input_size);
    Variable b(output_size, 1);
    this->w = w;
    this->b = b;
    this->w.randoms(0., sqrt((1./(float)input_size)));

}

FunctionLinear::FunctionLinear(int output_size, int input_size, bool no_bias) : Function() {
    name = "FunctionLinear";

    noBias = no_bias;

    Variable w(output_size, input_size);
    this->w = w;
    this->w.randoms(0., sqrt((1./(float)input_size)));

    if (!noBias){
        Variable b(output_size, 1);
        this->b = b;
    }
}

void FunctionLinear::toHostArray(){
    i1.toHostArray();
    w.data.toHostArray();
    w.grad.toHostArray();
    if (!noBias){
        b.data.toHostArray();
        b.grad.toHostArray();
    }
}
void FunctionLinear::fromHostArray(){
    i1.fromHostArray();
    w.data.fromHostArray();
    w.grad.fromHostArray();
    if (!noBias){
        b.data.fromHostArray();
        b.grad.fromHostArray();
    }
}








PVariable FunctionLinear::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable x = inputs.at(0);
    //std::chrono::system_clock::time_point  start, end;
    //start = std::chrono::system_clock::now();

    //PVariable r = PVariable(new Variable(this, w.data.rows, x->data.cols));
    //PVariable r = make_shared<Variable>(this, w.data.rows, x->data.cols);
    //PVariable r = PVariable(obj_pool.construct(this, w.data.rows, x->data.cols), obj_destroy);
    //count_pool++;
    //PVariable r = PVariable(obj_pool.construct(this, w.data.rows, x->data.cols), [&obj_pool](Variable* ptr){obj_pool.destroy(ptr);});
    //PVariable r = PVariable(obj_pool.construct(this, w.data.rows, x->data.cols));
    PVariable r = PVariable(obj_construct(this, w.data.rows, x->data.cols), obj_destroy); //custom
    //end = std::chrono::system_clock::now();
    //int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    //cout << "FunctionLinear::forward1 time:" << elapsed << endl;
    outputs.push_back(r);


    if (i1.cols == 0 || i1.cols != x->data.cols){
        i1 = cuMat(1, x->data.cols);
        i1.ones();
    }

    //start = std::chrono::system_clock::now();

    if (!noBias) b.data.dot(i1, r->data);
    w.data.dot_plus(x->data, r->data);
    //r->data = w->data.dot(x->data) + b->data.dot(i1);

    //end = std::chrono::system_clock::now();
    //elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    //cout << "FunctionLinear::forward2 time:" << elapsed << endl;

    return r;
}
void FunctionLinear::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    //std::chrono::system_clock::time_point  start, end;
    //start = std::chrono::system_clock::now();

    PVariable x = inputs.at(0);

    if (x->isGetGrad) w.data.transpose_dot_plus(p_grad, x->grad);
    //x->grad += w->data.transpose().dot(p_grad);

    p_grad.dot_transpose_plus(x->data, w.grad);
    //w->grad += p_grad.dot(x->data.transpose());
    //w->grad.mul(1.0/((float)x->grad.cols), w->grad); //normalize by batch_size

    //cout << "FunctionLinear::backward" << endl;
    //cout << p_grad;
    //cout << "FunctionLinear::backward " << w->grad.sum() << endl;

    if (!noBias) p_grad.dot_transpose_plus(i1, b.grad);
    //b->grad += p_grad.dot(i1.transpose());
    //b->grad.mul(1.0/((float)x->grad.cols), b->grad); //normalize by batch_size

    //end = std::chrono::system_clock::now();
    //int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    //cout << "FunctionLinear::backward time:" << elapsed << endl;
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

    //cout << "FunctionEmbed::forward" << endl;
    //exit(1);

    PVariable x = inputs.at(0);

    int x_cols = 0;
    if (!x->isSparse) x_cols = x->data.cols;
    else x_cols = x->data_sparse.rows;

    PVariable r = PVariable(obj_construct(this, w.data.rows, x_cols), obj_destroy); //custom
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
/*
        cuMat wt = w.data.transpose();
        cuMat rt(r->data.cols, r->data.rows);
        x->data_sparse.s_d_dot(wt, rt);
        cuMat rtmp = r->data;
        rt.transpose(rtmp);
        r->data.plus(rtmp, r->data);
*/
    }
    //cout << "FunctionEmbed::forward end" << endl;


    return r;
}
void FunctionEmbed::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){


    PVariable x = inputs.at(0);

    if (x->isGetGrad) w.data.transpose_dot_plus(p_grad, x->grad);

    if (!x->isSparse){
        p_grad.dot_transpose_plus(x->data, w.grad);
    }
    else{
        //cout << "p_grad.rows:" << p_grad.rows << " p_grad.cols:" << p_grad.cols << endl;
        //cout << "data_sparse.rows:" << x->data_sparse.rows << " data_sparse.cols:" << x->data_sparse.cols << endl;

        cuMat xd = x->data_sparse.toDense();
        p_grad.dot_plus(xd, w.grad);

        /*
        cuMat xd = x->data_sparse.toDense();
        cuMat xdt =xd.transpose();
        p_grad.dot_transpose_plus(xdt, w.grad);
        */
    }

    if (!noBias) p_grad.dot_transpose_plus(i1, b.grad);

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

    //this->inputs = inputs;
    PVariable x = inputs.at(0);


    //PVariable r;
    //if (outputs.empty()){
        //r = PVariable(new Variable(this, x->data.rows, x->data.cols));
        PVariable r = PVariable(obj_construct(this, x->data.rows, x->data.cols), obj_destroy); //custom

        outputs.push_back(r);
    //}
    //else r = outputs.back();
    /*
        }
    if (r == NULL || r->data.cols != x->data.cols){
        if (r != NULL) delete r;
        r = new Variable(this, x->data.rows, x->data.cols);
    }
    */
    x->data.relu(r->data);

    return r;
}
void FunctionReLU::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);


    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.relu_d(rr->data);

    rr->data.mul_plus(p_grad, x->grad, 1.0, 1.0);

    //rr->data.mul(p_grad, rr->data);
    //x->grad.plus(rr->data, x->grad);
}

FunctionSigmoid::FunctionSigmoid() : Function() {
    name = "FunctionSigmoid";
}
PVariable FunctionSigmoid::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    //this->inputs = inputs;
    PVariable x = inputs.at(0);

    //PVariable r;
    //if (outputs.empty()){
        //r = PVariable(new Variable(this, x->data.rows, x->data.cols));
        PVariable r = PVariable(obj_construct(this, x->data.rows, x->data.cols), obj_destroy); //custom

        outputs.push_back(r);
   //}
    //else r = outputs.back();
    /*
    if (r == NULL || r->data.cols != x->data.cols){
        if (r != NULL) delete r;
        r = new Variable(this, x->data.rows, x->data.cols);
    }
    */
    x->data.sigmoid(r->data);

    return r;
}
void FunctionSigmoid::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);

    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }
    x->data.sigmoid_d(rr->data);

    rr->data.mul_plus(p_grad, x->grad, 1.0, 1.0);

    //rr->data.mul(p_grad, rr->data);
    //x->grad.plus(rr->data, x->grad);
}

FunctionTanh::FunctionTanh() : Function() {
    name = "FunctionTanh";
}
PVariable FunctionTanh::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    //this->inputs = inputs;
    PVariable x = inputs.at(0);

    //PVariable r;
    //if (outputs.empty()){
        //r = PVariable(new Variable(this, x->data.rows, x->data.cols));
        PVariable r = PVariable(obj_construct(this, x->data.rows, x->data.cols), obj_destroy); //custom

        outputs.push_back(r);
    //}
    //else r = outputs.back();
    /*
    if (r == NULL || r->data.cols != x->data.cols){
        if (r != NULL) delete r;
        r = new Variable(this, x->data.rows, x->data.cols);
    }
    */
    x->data.tanh(r->data);

    return r;
}
void FunctionTanh::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);

    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }
    x->data.tanh_d(rr->data);

    rr->data.mul_plus(p_grad, x->grad, 1.0, 1.0);

    //rr->data.mul(p_grad, rr->data);
    //x->grad.plus(rr->data, x->grad);
}

FunctionSoftmax::FunctionSoftmax() : Function() {
    name = "FunctionSoftmax";
}
PVariable FunctionSoftmax::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    //this->inputs = inputs;
    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);

    //cuMat y(x->data.rows, x->data.cols);
    PVariable r;
    //if (outputs.empty()){
        r = PVariable(new Variable(x->data.rows, x->data.cols));
        outputs.push_back(r);
    //}
    //else r = outputs.back();

    x->data.softmax(r->data);

    return r;
}

FunctionSoftmaxCrossEntropy::FunctionSoftmaxCrossEntropy() : Function() {
    name = "FunctionSoftmaxCrossEntropy";
    loss = cuMat(1,1);
    loss.ones();

}
PVariable FunctionSoftmaxCrossEntropy::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    //this->inputs = inputs;
    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);

    //cuMat y(x->data.rows, x->data.cols);
    if (rr3.get() == NULL || rr3->data.cols != x->data.cols){
        rr3 = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.softmax(rr3->data);



    if (rr.get() == NULL || rr->data.cols != rr3->data.cols){
        rr = PVariable(new Variable(rr3->data));
    }
    else rr->data = rr3->data;



    //if (seed == NULL) {
    //    seed = new cuMat(x->data.rows, x->data.cols);
    //    seed->ones();
   //}
    //rr3->data.plus(1e-8, *seed, rr3->data);
    //rr3->data.log(rr3->data, 1e-8);
    //rr3->data.mul(t->data, rr3->data);
    //rr3->data.mul(-1.0, rr3->data);
    rr3->data.softmax_cross_entropy(t->data, rr3->data);
    float sum = rr3->data.sum();

    sum /= rr3->data.cols;

    //PVariable r;
    //if (outputs.empty()){
        PVariable r = PVariable(new Variable(this, loss.rows, loss.cols));
        //PVariable r = PVariable(obj_construct(this, loss.rows, loss.cols), obj_destroy); //custom
        outputs.push_back(r);
    //}
    //else r = outputs.back();

    //if (r == NULL) r = new Variable(this, loss.rows, loss.cols);

    loss.mul(sum, r->data);

    return r;
}
void FunctionSoftmaxCrossEntropy::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){

    //cout << "FunctionSoftmaxCrossEntropy::backward p_grad" << endl;
    //cout << p_grad << endl;

    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);
    PVariable y = rr;

    if (rr2.get() == NULL || rr2->data.cols != y->data.cols){
        rr2 = PVariable(new Variable(y->data));
    }
    //else rr2->data = y->data;

    //rr2->data.minus(t->data, rr2->data);
    y->data.minus(t->data, rr2->data);
    float batch_size = rr2->data.cols;
    rr2->data.mul(1.0/batch_size, rr2->data);
    x->grad.plus(rr2->data, x->grad);
    //x->grad.plus(rr2->data/rr2->data.cols, x->grad);
    //x->grad += y->data - t->data;
}


FunctionMeanSquaredError::FunctionMeanSquaredError() : Function() {
    name = "FunctionMeanSquaredError";
    loss = cuMat(1,1);
    loss.ones();
}

PVariable FunctionMeanSquaredError::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    //this->inputs = inputs;
    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);

    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data));
    }

    x->data.minus(t->data, rr->data);

    rr->data.mul(rr->data, rr->data);

    float sum = rr->data.sum();
    sum /= (2*rr->data.cols);

    PVariable r;
    r = PVariable(new Variable(this, loss.rows, loss.cols));
   outputs.push_back(r);

    //if (r == NULL) r = new Variable(this, loss.rows, loss.cols);
    loss.mul(sum, r->data);

    return r;
}
void FunctionMeanSquaredError::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);
    PVariable t = inputs.at(1);
    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data));
    }
    //else rr->data = x->data;

    x->data.minus(t->data, rr->data);
    float batch_size = rr->data.cols;
    rr->data.mul(1.0/batch_size, rr->data);
    x->grad.plus(rr->data, x->grad);
    //x->grad.plus(rr->data/rr->data.cols, x->grad);
    //x->grad += x->data - t->data;
}



FunctionDropout::FunctionDropout(float p) : Function() {
    name = "FunctionDropout";
    this->p = p;
}
PVariable FunctionDropout::forward(vector<PVariable > &inputs, vector<PVariable > &outputs){

    //this->inputs = inputs;
    PVariable x = inputs.at(0);

    //PVariable r;
    //if (outputs.empty()){
        //r = PVariable(new Variable(this, x->data.rows, x->data.cols));
        PVariable r = PVariable(obj_construct(this, x->data.rows, x->data.cols), obj_destroy); //custom

        outputs.push_back(r);
    //}
    //else r = outputs.back();
    /*
    if (r == NULL || r->data.cols != x->data.cols) {
        if (r != NULL) delete r;
        r = new Variable(this, x->data.rows, x->data.cols);
    }
    */



    if (rr.get() == NULL || rr->data.cols != x->data.cols){
        rr = PVariable(new Variable(x->data.rows, x->data.cols));
    }

    x->data.dropout(r->data, rr->data, p);
    //outputs.push_back(rr);
    return r;
}
void FunctionDropout::backward(cuMat &p_grad, vector<PVariable > &inputs, vector<PVariable > &outputs){
    PVariable x = inputs.at(0);
    PVariable idx = rr;

    if (rr2.get() == NULL || rr2->data.cols != x->grad.cols){
        rr2 = PVariable(new Variable(x->grad.rows, x->grad.cols));
    }

    idx->data.mul_plus(p_grad, x->grad, 1.0, 1.0);
    //idx->data.mul(p_grad, rr2->data);
    //x->grad.plus(rr2->data, x->grad);

    //x->grad += idx->data * p_grad;
}
