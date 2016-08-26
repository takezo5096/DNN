/*
 * autoencoder.h
 *
 *  Created on: 2016/01/18
 *      Author: takeshi.fujita
 */

#ifndef AUTOENCODER_H_
#define AUTOENCODER_H_

#include "variable.h"
#include "function.h"
#include "model.h"
#include "optimizer_adam.h"
#include "optimizer_sgd_moment.h"

class AutoEncoder {
public:
    Function *linear1;
    Function *linear2;

    PVariable z1;
    PVariable z2;

    Variable *w2, *b2;

    Function *f_relu1;
    Function *f_drop1;
    Function *f_mean_squared_error;
    Model model;
    //OptimizerAdam *optimizer;
    Optimizer *optimizer;

    PVariable nx;
    PVariable drx;



    AutoEncoder(FunctionLinear *linear1,
            const float dropout_p, Optimizer *optimizer) {


        this->linear1 = linear1;

        linear1->w.randoms(0, 1.0 / sqrt(((float) linear1->w.data.cols)));
        linear1->b.zeros();

        w2 = new Variable(linear1->w.data.cols, linear1->w.data.rows);
        b2 = new Variable(linear1->w.data.cols, 1);
        w2->randoms(0, 1.0 / sqrt(((float) linear1->w.data.rows)));
        linear2 = new FunctionLinear(*w2, *b2);

        f_relu1 = new FunctionReLU();
        f_drop1 = new FunctionDropout(dropout_p);
        f_mean_squared_error = new FunctionMeanSquaredError();

        optimizer->model->putF("l1", linear1);
        optimizer->model->putF("l2", linear2);
        optimizer->init();

        this->optimizer = optimizer;
    }

    ~AutoEncoder() {
        delete linear2;
        delete f_relu1;
        delete f_drop1;
        delete f_mean_squared_error;
        //delete optimizer;
        //delete nx;
        //delete drx;
        delete w2;
        delete b2;
    }

    float train(PVariable x) {



        if (nx == NULL) {
            nx = PVariable(new Variable(x->data.rows, x->data.cols));
            drx = PVariable(new Variable(x->data.rows, x->data.cols));
            drx->randoms(0, 0.01);
        }

        //denoising auto-encoder
        x->data.plus(drx->data, nx->data);
        //nx->data = x->data;

        z1 = f_drop1->forward(f_relu1->forward(linear1->forward(nx)));
        z2 = linear2->forward(z1);

        PVariable loss = f_mean_squared_error->forward(z2, x);

        loss->backward();

        float loss_val = loss->data(0, 0);
        //cout << " loss:" << loss_val << endl;

        // update -------------------------------------------
        optimizer->update();

        loss->zero_grads();

        return loss_val;
    }
};

#endif /* AUTOENCODER_H_ */
