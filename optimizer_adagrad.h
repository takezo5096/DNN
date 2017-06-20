/*
 * optimizer_adagrad.h
 *
 */

#ifndef OPTIMIZER_ADAGRAD_H_
#define OPTIMIZER_ADAGRAD_H_

#include "model.h"
#include "optimizer.h"

class OptimizerAdagradParams : public OptimizerParams {
public:
    cuMat ndw;
    cuMat g2;


    OptimizerAdagradParams(int output_units, int input_units) {


        ndw = cuMat(output_units, input_units);
        g2 = cuMat(output_units, input_units);
    }
};

class OptimizerAdagrad : public Optimizer {
public:


    OptimizerAdagrad(Model *model, float lr) : Optimizer(model, lr) {
    }
    OptimizerAdagrad(Model *model, float lr, float clip_grad_threshold) : Optimizer(model, lr, clip_grad_threshold) {
    }

    OptimizerParams *createOptimizerParams(Variable *v){
        return new OptimizerAdagradParams(v->data.rows, v->data.cols);
    }



    void update_param(Variable *w, OptimizerParams &opp) {

        OptimizerAdagradParams &op = (OptimizerAdagradParams &)opp;

        op.g2 += w->grad * w->grad;

        cuMat tmp = op.g2.sqrt();
        tmp = w->grad / tmp;

        tmp.mul(-lr, op.ndw);

        w->data.plus(op.ndw, w->data);

    }

};

#endif /* OPTIMIZER_ADAGRAD_H_ */
