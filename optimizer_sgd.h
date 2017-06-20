/*
 * optimizer_adam.h
 *
 */

#ifndef OPTIMIZER_SGD_H_
#define OPTIMIZER_SGD_H_

#include "model.h"
#include "optimizer.h"

class OptimizerSGDParams : public OptimizerParams {
public:
    cuMat ndw;


    OptimizerSGDParams(int output_units, int input_units) {


        ndw = cuMat(output_units, input_units);

    }
};

class OptimizerSGD : public Optimizer {
public:


    OptimizerSGD(Model *model, float lr) : Optimizer(model, lr) {
    }
    OptimizerSGD(Model *model, float lr, float clip_grad_threshold) : Optimizer(model, lr, clip_grad_threshold) {
    }

    OptimizerParams *createOptimizerParams(Variable *v){
        return new OptimizerSGDParams(v->data.rows, v->data.cols);
    }



    void update_param(Variable *w, OptimizerParams &opp) {

        //weight decay
        //w.data.plus_util(0.0001, 1.0, w.grad, w.grad);

        OptimizerSGDParams &op = (OptimizerSGDParams &)opp;

        w->grad.mul(-lr, op.ndw);

        w->data.plus(op.ndw, w->data);
    }

};

#endif /* OPTIMIZER_SGD_H_ */
