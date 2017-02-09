/*
 * optimizer_adam.h
 *
 *  Created on: 2016/01/27
 *      Author: takeshi.fujita
 */

#ifndef OPTIMIZER_ADAM_H_
#define OPTIMIZER_ADAM_H_

#include "model.h"
#include "optimizer.h"

class OptimizerAdamParams : public OptimizerParams {
public:
    cuMat adam_w_m;
    cuMat adam_w_v;

    cuMat dw_tmp;

    cuMat m_h_t;
    cuMat v_h_t;

    cuMat ndw;


    OptimizerAdamParams(int output_units, int input_units) {
        //init(output_units, input_units);
        adam_w_m = cuMat(output_units, input_units);
        adam_w_v = cuMat(output_units, input_units);
        m_h_t = cuMat(output_units, input_units);
        v_h_t = cuMat(output_units, input_units);

        dw_tmp = cuMat(output_units, input_units);

        ndw = cuMat(output_units, input_units);
    }

    /*
    void init(int output_units, int input_units){
        adam_w_m = cuMat(output_units, input_units);
        adam_w_v = cuMat(output_units, input_units);
        m_h_t = cuMat(output_units, input_units);
        v_h_t = cuMat(output_units, input_units);

        dw_tmp = cuMat(output_units, input_units);

        ndw = cuMat(output_units, input_units);
    }*/
};

class OptimizerAdam: public Optimizer {
public:

    float beta1 = 0.9;
    float beta2 = 0.999;

    OptimizerAdam(Model *model, float lr) : Optimizer(model, lr) {
    }
    OptimizerAdam(Model *model, float lr, float clip_grad_threshold) : Optimizer(model, lr, clip_grad_threshold) {
    }


    OptimizerParams *createOptimizerParams(Variable *v){
        //if (v != NULL) return new OptimizerAdamParams(v->data.rows, v->data.cols);
        //else return new OptimizerAdamParams(0, 0);
        return new OptimizerAdamParams(v->data.rows, v->data.cols);
    }


    float lr_f(float alpha, int epoch){
            float fix1 = 1.0 - std::pow(beta1, epoch);
            float fix2 = 1.0 - std::pow(beta2, epoch);
            return alpha * std::sqrt(fix2) / fix1;
    }

    void update_param(Variable *w, OptimizerParams &opp) {

        //w.grad.mul(1.0/batch_size, w.grad);

        //weight decay
        //w.data.plus_util(0.0001, 1.0, w.grad, w.grad);

        OptimizerAdamParams &op = (OptimizerAdamParams &)opp;


/*
        //m_t = beta1 * m_t-1 + (1.0-beta1)dw;
        //v_t = beta2 * v_t-1 + (1.0-beta2)dw^2;
        // m_h_t = m_t / (1.0 - beta1^t)
        // v_h_t = v_t / (1.0 - beta2^t)
        //theta_t = theta_t-1 - alpha * m_h_t / (sqrt(v_h_t) + eqs)
        op.adam_w_m = beta1 * op.adam_w_m + (1-beta1)*w->grad;
        op.adam_w_v = beta2 * op.adam_w_v + (1-beta2)*w->grad*w->grad;
        op.m_h_t = op.adam_w_m / (1.0 - std::pow(beta1, epoch));
        op.v_h_t = op.adam_w_v / (1.0 - std::pow(beta2, epoch));
        op.ndw = lr_f(-lr, epoch) * op.m_h_t / (op.v_h_t.sqrt() + 1e-8);
*/



/*
        //op->adam_w_m += (1.0 - beta1) * (w.grad - op->adam_w_m);
        //op->adam_w_v += (1.0 - beta2) * (w.grad*w.grad - op->adam_w_v);
        //op->ndw = lr(alpha, epoch) * op->adam_w_m / (op->adam_w_v.sqrt() + 1e-8);
        w->grad.plus_util(1.0-beta1, -(1.0-beta1), op.adam_w_m, op.dw_tmp);
        op.adam_w_m.plus(op.dw_tmp, op.adam_w_m);
        w->grad.mul(w->grad, op.dw_tmp);
        op.dw_tmp.plus_util(1.0-beta2, -(1.0-beta2), op.adam_w_v, op.dw_tmp);
        op.adam_w_v.plus(op.dw_tmp, op.adam_w_v);

        //op.adam_w_m.adam(op.adam_w_v, op.ndw, lr(-alpha, epoch), 1e-8);
        op.adam_w_v.sqrt(op.dw_tmp, 0.0);
        op.dw_tmp.plus(1e-8, op.dw_tmp);
        op.adam_w_m.div(op.dw_tmp, op.dw_tmp);
        op.dw_tmp.mul(lr_f(-lr, epoch), op.ndw);
*/
        op.adam_w_m.adam2(op.adam_w_v, w->grad, op.ndw, beta1, beta2, lr_f(-lr, epoch), 1e-8);
        w->data.plus(op.ndw, w->data);
    }

};

#endif /* OPTIMIZER_ADAM_H_ */
