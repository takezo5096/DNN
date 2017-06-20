/*
 * optimizer_adam.h
 *
 */

#ifndef OPTIMIZER_SGD_MOMENT_H_
#define OPTIMIZER_SGD_MOMENT_H_

#include "model.h"
#include "optimizer.h"

class OptimizerSGDMomentParams : public OptimizerParams {
public:
    cuMat dw_tmp;
    cuMat ndw;

    cuMat prev_w_grad;

    OptimizerSGDMomentParams(int output_units, int input_units) {

        dw_tmp = cuMat(output_units, input_units);

        ndw = cuMat(output_units, input_units);

        prev_w_grad = cuMat(output_units, input_units);
    }
};

class OptimizerSGDMoment: public Optimizer {
public:

    float mu;

    OptimizerSGDMoment(Model *model, float lr, float mu) : Optimizer(model, lr) {
        this->mu = mu;
    }
    OptimizerSGDMoment(Model *model, float lr, float mu, float clip_grad_threshold) : Optimizer(model, lr, clip_grad_threshold) {
        this->mu = mu;
    }

    OptimizerParams *createOptimizerParams(Variable *v){
        return new OptimizerSGDMomentParams(v->data.rows, v->data.cols);
    }



    void update_param(Variable *w, OptimizerParams &opp) {

        //weight decay
        //w.data.plus_util(0.0001, 1.0, w.grad, w.grad);

        OptimizerSGDMomentParams &op = (OptimizerSGDMomentParams &)opp;

        w->grad.mul(-lr, op.ndw);

        //moment
        op.prev_w_grad.mul(mu, op.dw_tmp);
        op.ndw.plus(op.dw_tmp, op.ndw);

        w->data.plus(op.ndw, w->data);

        op.prev_w_grad = op.ndw;

    }

};

#endif /* OPTIMIZER_SGD_MOMENT_H_ */
