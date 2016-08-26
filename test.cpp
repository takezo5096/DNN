#include <list>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

#include "function.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "batchdata.h"
#include "iris.h"
#include "mnist.h"
#include "autoencoder.h"
#include "optimizer_adam.h"
#include "optimizer_sgd_moment.h"
#include "word_embed.h"


using namespace std;

MallocCounter mallocCounter;


void asMatrix(PVariable x1, float *X){
    x1->data.memSetHost(X);
}


int main(){

/*
    WordEmbed we;
    we.add("今日は良い天気です");
    we.add("今日は悪い天気です");
    we.add("明日は良い気候です");
    we.add("明日は天気");
    vector<vector<float>> vs = we.getOneHotVectors(we.getIdSamles()[3], 5);
    for(int i=0; i<vs.size(); i++){
        vector<float> v = vs.at(i);
        for(int j=0; j<v.size(); j++){
            cout << v.at(j) << " ";
        }
        cout << endl;
    }


    Variable xx1(1,1);
    Variable xx2(1,1);
    xx1.data.fill(2);
    xx2.data.fill(5);

    FunctionPlus *f_plus = new FunctionPlus();
    FunctionMinus *f_minus = new FunctionMinus();
    FunctionMul *f_mul = new FunctionMul();
    FunctionSin *f_sin = new FunctionSin();
    FunctionLog *f_log = new FunctionLog();

    Variable *f_log_r = f_log->forward(&xx1);
    Variable *f_mul_r = f_mul->forward(&xx1, &xx2);
    Variable *f_plus_r = f_plus->forward(f_log_r, f_mul_r);
    Variable *f_sin_r = f_sin->forward(&xx2);
    Variable *f_minus_r = f_minus->forward(f_plus_r, f_sin_r);


    f_minus_r->backward();


    cout << "r.data" << endl;
    cout << f_minus_r->data;
    cout << "xx1.grad" << endl;
    cout << xx1.grad;
    cout << "xx2.grad" << endl;
    cout << xx2.grad;
    f_minus_r->zero_grads();
*/

    //int epochNums = 50;
    //int epochAENums = 100;
    //int totalSampleSize = 150;
    //int batchSize = 5;
    //int i_size = 4;
    //int n_size = 10;
    //int o_size = 3;
    //float learning_rate = 0.001;
    //float dropout_p = 0.5;

    int epochNums = 20;
    int epochAENums = 20;
    int totalSampleSize = 60000;
    int totalTestSize = 10000;

    int batchSize = 100;
    int i_size = 784;
    int n_size = 1024;
    int o_size = 10;
    float learning_rate = 0.0001;
    float ae_learning_rate = 0.0001;
    float dropout_p = 0.3;
    float ae_dropout_p = 0.3;

    cout << "init dataset..." << endl;
    vector<vector<float>> train_data, test_data;
    vector<float> label_data, label_test_data;
    //Iris iris;
    //train_data =  iris.getTrainData();
    //label_data = iris.getLabelData();
    //test_data =  iris.getTrainData();
    //label_test_data = iris.getLabelData();
    Mnist mnist, mnist_test;
    train_data = mnist.readTrainingFile("train-images-idx3-ubyte");
    label_data = mnist.readLabelFile("train-labels-idx1-ubyte");
    test_data = mnist_test.readTrainingFile("t10k-images-idx3-ubyte");
    label_test_data = mnist_test.readLabelFile("t10k-labels-idx1-ubyte");

    Dataset *dataset = new Dataset();
    dataset->standrize(&train_data);
    vector<BatchData *> bds;
    for(int i=0; i<totalSampleSize/batchSize; i++){
        BatchData *bdata = new BatchData(i_size, o_size, batchSize);
        dataset->createMiniBatch(train_data, label_data, bdata->getX(), bdata->getD(), batchSize, o_size, i);
        bds.push_back(bdata);
    }
    dataset->standrize(&test_data);
    vector<BatchData *> bds_test;
    for(int i=0; i<totalTestSize/batchSize; i++){
        BatchData *bdata = new BatchData(i_size, o_size, batchSize);
        dataset->createMiniBatch(test_data, label_test_data, bdata->getX(), bdata->getD(), batchSize, o_size, i);
        bds_test.push_back(bdata);
    }



    cout << "create model..." << endl;
    Variable w1(n_size, i_size); Variable b1(n_size, 1);
    Variable w2(n_size, n_size); Variable b2(n_size, 1);
    Variable w3(o_size, n_size); Variable b3(o_size, 1);
    w1.randoms(0., sqrt((1./(float)i_size)));
    w2.randoms(0., sqrt(1./((float)n_size)));
    w3.randoms(0., sqrt(1./((float)n_size)));

    Function *f1 = new FunctionLinear(w1, b1);
    Function *f_relu1 = new FunctionReLU();
    Function *f_drop1 = new FunctionDropout(dropout_p);
    Function *f2 = new FunctionLinear(w2, b2);
    Function *f_relu2 = new FunctionReLU();
    Function *f_drop2 = new FunctionDropout(dropout_p);
    Function *f3 = new FunctionLinear(w3, b3);
    Function *f_softmax_cross_entoropy = new FunctionSoftmaxCrossEntropy();
    Function *f_softmax = new FunctionSoftmax();


    std::chrono::system_clock::time_point  start, end;


/*
    cout << "start training autoencoder1..." << endl;
    Model ae1Model;
    OptimizerAdam ae1Optimizer(&ae1Model, ae_learning_rate);
    //OptimizerSGDMoment ae1Optimizer(&ae1Model, ae_learning_rate, 0.7);
    AutoEncoder *f_autoencoder1 = new AutoEncoder((FunctionLinear *)f1,
            ae_dropout_p, &ae1Optimizer);
    for(int k=0; k<epochAENums; k++){

        start = std::chrono::system_clock::now();

        std::random_shuffle(bds.begin(), bds.end());

        float sum_loss = 0;

        for(int i=0; i<totalSampleSize/batchSize; i++){
            // create mini-batch =========================
            float *X = bds.at(i)->getX();
            asMatrix(x1, X);

            float loss = f_autoencoder1->train(&x1);
            sum_loss += loss*batchSize;
        }
        end = std::chrono::system_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        float loss_mean = sum_loss/((float)totalSampleSize);
        cout << "epoch:" << k+1 << " loss:" << loss_mean << " time:" << elapsed << "ms" << endl;
    }
    delete f_autoencoder1;

    cout << "start training autoencoder2..." << endl;
    Model ae2Model;
    OptimizerAdam ae2Optimizer(&ae2Model, ae_learning_rate);
    //OptimizerSGDMoment ae2Optimizer(&ae2Model, ae_learning_rate, 0.7);
    AutoEncoder *f_autoencoder2 = new AutoEncoder((FunctionLinear *)f2,
            ae_dropout_p, &ae2Optimizer);
    for(int k=0; k<epochAENums; k++){

        start = std::chrono::system_clock::now();

        std::random_shuffle(bds.begin(), bds.end());

        float sum_loss = 0;

        for(int i=0; i<totalSampleSize/batchSize; i++){
            // create mini-batch =========================
            float *X = bds.at(i)->getX();
            asMatrix(x1, X);

            float loss = f_autoencoder2->train(f1->forward(&x1));
            sum_loss += loss*batchSize;
        }
        end = std::chrono::system_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        float loss_mean = sum_loss/((float)totalSampleSize);
        cout << "epoch:" << k+1 << " loss:" << loss_mean << " time:" << elapsed << "ms" << endl;
    }
    delete f_autoencoder2;
*/


    Model model;
    model.putF("f1", f1);
    model.putF("f2", f2);
    model.putF("f3", f3);


    OptimizerAdam optimizer(&model, learning_rate);
    //OptimizerSGDMoment optimizer(&model, learning_rate, 0.7);
    optimizer.init();

    cout << "start training fine tuning..." << endl;
    for(int k=0; k<epochNums; k++){

        start = std::chrono::system_clock::now();

        std::random_shuffle(bds.begin(), bds.end());

        float sum_loss = 0.0;

        for(int i=0; i<totalSampleSize/batchSize; i++){
            PVariable x1(new Variable(i_size, batchSize));
            PVariable d(new Variable(o_size, batchSize));

            // create mini-batch =========================
            float *X = bds.at(i)->getX();
            float *D = bds.at(i)->getD();
            asMatrix(x1, X);
            asMatrix(d, D);

            //cout << "forward" << endl;
            // forward ------------------------------------------

            PVariable h1 = f_drop1->forward(f_relu1->forward(f1->forward(x1)));
            PVariable h2 = f_drop2->forward(f_relu2->forward(f2->forward(h1)));

            PVariable h3 = f3->forward(h2);
            PVariable loss = f_softmax_cross_entoropy->forward(h3, d);
            //cout << "backward" << endl;
            // backward -----------------------------------------
            loss->backward();

            //cout << "loss" << endl;
            // loss ---------------------------------------------
            float loss_val = loss->val()*batchSize;
            sum_loss += loss_val;

            //cout << "update" << endl;
            // update -------------------------------------------
            optimizer.update();

            //cout << "zero grads" << endl;
            // zero grads
            loss->zero_grads();

            loss->unchain();

            //cout << "loop end" << endl;
        }
        end = std::chrono::system_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        float loss_mean = sum_loss/((float)totalSampleSize);
        cout << "epoch:" << k+1 << " loss:" << loss_mean << " time:" << elapsed << "ms" << endl;
        //if (loss_mean < 0.05) break;
    }

    cout << "saving model..." << endl;
    model.save("mlp_test.model");



    cout << "loading model..." << endl;
    Model model_train;
    model_train.load("mlp_test.model");
    Function *nf1 = model_train.f("f1");
    Function *nf2 = model_train.f("f2");
    Function *nf3 = model_train.f("f3");

    cout << "start predict..." << endl;
    float accurecy = 0;
    int predict_epoch = totalTestSize/batchSize;
    for(int i=0; i<predict_epoch; i++){

        std::random_shuffle(bds_test.begin(), bds_test.end());

        PVariable x1(new Variable(i_size, batchSize));
        PVariable d(new Variable(o_size, batchSize));

        // create mini-batch =========================
        float *X = bds_test.at(i)->getX();
        float *D = bds_test.at(i)->getD();
        asMatrix(x1, X);
        asMatrix(d, D);

        nf1->forward(x1);

        // forward ------------------------------------------
        PVariable h1 = f_relu1->forward(nf1->forward(x1));
        PVariable h2 = f_relu2->forward(nf2->forward(h1));

        PVariable h3 = nf3->forward(h2);

        PVariable y = f_softmax->forward(h3, d);


        int maxIdx_z3[batchSize];
        y->data.maxRowIndex(maxIdx_z3);

        int maxIdx_d[batchSize];
        d->data.maxRowIndex(maxIdx_d);

        int hit = 0;
        for(int i=0; i<batchSize; i++){
            if (maxIdx_d[i] == maxIdx_z3[i]) hit++;
        }
        accurecy += ((float)hit) / ((float) batchSize);

    }
    cout << "accurecy: " << accurecy/((float)predict_epoch)*100 << "%" << endl;


}
