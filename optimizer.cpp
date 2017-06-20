/*
 * Optimizer.c
 *
 */

#include <thread>

#include "optimizer.h"
#include "variable.h"


Optimizer::Optimizer(Model *model, float learning_rate) {

    this->model = model;

    lr = learning_rate;

}
Optimizer::Optimizer(Model *model, float learning_rate, float clip_grad_threshold) {

    this->model = model;

    lr = learning_rate;

    this->clip_grad_threshold = clip_grad_threshold;

}

Optimizer::~Optimizer() {

    delOpts();

}


void Optimizer::delOpts(){
    for (int i = 0; i < opts.size(); i++) {
            OptimizerParams *p = opts.at(i);
            delete p;
        }
    opts.clear();
}

OptimizerParams *Optimizer::createOptimizerParams(Variable *v){
    cout << "createOptimizerParams base" << endl;
}


void Optimizer::init() {

    epoch = 1;
    updateParams = model->getUpdateParams();

    delOpts();

    for (int i = 0; i < updateParams.size(); i++) {
        UpdateParams *up = updateParams.at(i);
        for (int j = 0; j < up->params.size(); j++) {
            Variable *v = up->params.at(j);

            opts.push_back(createOptimizerParams(v));

        }
    }
}


void Optimizer::update_param(Variable *w, OptimizerParams &opp) {
    cout << "update_param" << endl;
}


void Optimizer::zero_grads() {

    for (int i = 0; i < updateParams.size(); i++) {
        UpdateParams *up = updateParams.at(i);
        for(int j=0; j < up->params.size(); j++){
            Variable *v = up->params.at(j);
            v->grad *= 0.0;

        }
    }
}

void Optimizer::clip_grad(Variable *v){
    if (clip_grad_threshold > 0.0){

        v->grad.element_wise_clip(v->grad, clip_grad_threshold);
    }
}

void Optimizer::update() {

    int k = 0;

    for (int i = 0; i < updateParams.size(); i++) {
        UpdateParams *up = updateParams.at(i);
        for(int j=0; j < up->params.size(); j++){
            Variable *v = up->params.at(j);

            clip_grad(v);
            update_param(v, *opts.at(k));
            k++;
        }
    }
    epoch++;

    zero_grads();
}

