/*
 * Optimizer.c
 *
 *  Created on: 2016/01/25
 *      Author: takeshi.fujita
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
            //OptimizerParams *op = createOptimizerParams(v);
            //if (v == NULL) op->no_params = true;
            //opts.push_back(op);
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
        /*
        float sq = v->grad.l2();
        float rate = clip_grad_threshold/sq;
        if (rate < 1.){
            v->grad.mul(rate, v->grad);
        }
         */

        v->grad.element_wise_clip(v->grad, clip_grad_threshold);
    }
}

void Optimizer::update() {

    int k = 0;

    for (int i = 0; i < updateParams.size(); i++) {
        UpdateParams *up = updateParams.at(i);
        for(int j=0; j < up->params.size(); j++){
            Variable *v = up->params.at(j);

            /*
            if (opts.at(k)->no_params){
                opts.at(k)->init(v->data.rows, v->data.cols);
                opts.at(k)->no_params = false;
            }
            */
            //cout << v->grad;

            //changed v->grad to v->grad means (v->grad / batch_size )
            //v->grad = 1./v->grad.cols * v->grad;


            clip_grad(v);
            update_param(v, *opts.at(k));
            k++;
        }
    }
    epoch++;



/*
    //updating weight params with threads
    for (int i = 0; i < updateParams.size(); i++) {
        UpdateParams *up = updateParams.at(i);
        for(int j=0; j < up->params.size(); j++){
            Variable *v = up->params.at(j);
            //v->grad.mul(1.0/100.0, v->grad);
            thread *t = new std::thread(&Optimizer::update_param, this,
            std::ref(*v), std::ref(*opts.at(k))
            k++;
        }
    }
    epoch++;


    for (int i = 0; i < ts.size(); i++) {
        ts.at(i)->join();
    }

    for (int i = 0; i < ts.size(); i++) {
        delete ts.at(i);
    }
    ts.clear();
*/

    zero_grads();
}

