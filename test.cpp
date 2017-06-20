#include <list>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "png.h"


#include "graph.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "batchdata.h"
#include "iris.h"
#include "mnist.h"
#include "optimizer_adam.h"
#include "optimizer_sgd_moment.h"
#include "optimizer_adagrad.h"
#include "word_embed.h"
#include "cifar10.h"

using namespace std;

MallocCounter mallocCounter;


void asMatrix(PVariable x1, float *X){
    x1->data.memSetHost(X);
}


float getAccurecy(Graph *g_softmax, PVariable h, PVariable d, int batchSize){
    PVariable y = ((Softmax *)g_softmax)->forward(h);

    int maxIdx_z3[batchSize];
    y->data.maxRowIndex(maxIdx_z3);

    int maxIdx_d[batchSize];
    d->data.maxRowIndex(maxIdx_d);

    int hit = 0;
    for(int i=0; i<batchSize; i++){
        if (maxIdx_d[i] == maxIdx_z3[i]) hit++;
    }
    float accurecy = ((float)hit) / ((float) batchSize);
    return accurecy;
}


PVariable forward_one_step(Model &model, PVariable x1, bool is_train) {

    ((Dropout *)model.G("dropout4"))->isTrain(is_train);

    PVariable h1 = model.G("g_relu1")->forward(model.G("g_conv2d1")->forward(x1));
    PVariable h2 = model.G("g_relu2")->forward(model.G("g_conv2d2")->forward(h1));
    PVariable p1 = model.G("g_pooling1")->forward(h2);


    PVariable h3 = model.G("g_relu3")->forward(model.G("g_conv2d3")->forward(p1));
    PVariable h4 = model.G("g_relu4")->forward(model.G("g_conv2d4")->forward(h3));
    PVariable p2 = model.G("g_pooling2")->forward(h4);


    PVariable h5 = model.G("g_relu5")->forward(model.G("g_conv2d5")->forward(p2));
    PVariable h6 = model.G("g_relu6")->forward(model.G("g_conv2d6")->forward(h5));
    PVariable p3 = model.G("g_pooling3")->forward(h6);

    PVariable g1;
    g1 = model.G("dropout4")->forward(model.G("g_relu7")->forward(model.G("g1")->forward(p3)));
    PVariable g3 = model.G("g3")->forward(g1);

    return g3;
}


float test_accurecy(Model &model, vector<BatchData *> &bds_test, int i_size, int o_size, int totalTestSize, int batchSize, float *sum_loss){

    float accurecy = 0.0;

    int predict_epoch = totalTestSize/batchSize;
    for(int i=0; i<predict_epoch; i++){

        PVariable x(new Variable(i_size, batchSize, false));
        PVariable d(new Variable(o_size, batchSize, false));

        // create mini-batch =========================
        float *X = bds_test.at(i)->getX();
        float *D = bds_test.at(i)->getD();
        asMatrix(x, X);
        asMatrix(d, D);

        PVariable h = forward_one_step(model, x, false);


        PVariable loss = model.G("g_softmax_cross_entoropy")->forward(h, d);
        float l = loss->val();
        *sum_loss += l;

        accurecy += getAccurecy(model.G("g_softmax"), h, d, batchSize);

        model.zero_grads();
        model.unchain();
    }

    *sum_loss /= ((float)predict_epoch);
    return accurecy / ((float)predict_epoch);
}


int main(){

    Model model;

    int epochNums = 40;
    int totalSampleSize = 50000;
    int totalTestSize = 10000;

    int batchSize = 100;
    int i_size = 1024*3;
    int n_size = 512;
    int n_size2 = 512;
    int o_size = 10;
    float learning_rate = 0.001;

    int disp_num = 10;


    cout << "init dataset..." << endl;
    vector<vector<float>> train_data, test_data;
    vector<float> label_data, label_test_data;


    CIFAR10 cifar10, cifar10_test;
    cifar10.readFile("./cifar-10-batches-bin/data_batch_1.bin");
    cifar10.readFile("./cifar-10-batches-bin/data_batch_2.bin");
    cifar10.readFile("./cifar-10-batches-bin/data_batch_3.bin");
    cifar10.readFile("./cifar-10-batches-bin/data_batch_4.bin");
    cifar10.readFile("./cifar-10-batches-bin/data_batch_5.bin");
    train_data = cifar10.getDatas();
    label_data = cifar10.getLabels();
    totalSampleSize = train_data.size();

    cifar10_test.readFile("./cifar-10-batches-bin/test_batch.bin");
    test_data = cifar10_test.getDatas();
    label_test_data = cifar10_test.getLabels();
    totalTestSize = test_data.size();
    cout << "totalSampleSize:" << totalSampleSize << " totalTestSize:" << totalTestSize << endl;

    Dataset *dataset = new Dataset();

    cout << "create BatchData for training" << endl;
    dataset->normalize(&train_data, 255.0);
    vector<BatchData *> bds;
    for(int i=0; i<totalSampleSize/batchSize; i++){
        BatchData *bdata = new BatchData(i_size, o_size, batchSize);
        dataset->createMiniBatch(train_data, label_data, bdata->getX(), bdata->getD(), batchSize, o_size, i);
        bds.push_back(bdata);
    }
    cout << "create BatchData for test" << endl;
    dataset->normalize(&test_data, 255.0);
    vector<BatchData *> bds_test;
    for(int i=0; i<totalTestSize/batchSize; i++){
        BatchData *bdata = new BatchData(i_size, o_size, batchSize);
        dataset->createMiniBatch(test_data, label_test_data, bdata->getX(), bdata->getD(), batchSize, o_size, i);
        bds_test.push_back(bdata);
    }




    std::chrono::system_clock::time_point  start, end;

    //Prepare MODEL
    cout << "create model..." << endl;
    model.putG("g1", new Linear(n_size, 4 * 4 * 32));

    model.putG("g3", new Linear(o_size, n_size2));

    model.putG("dropout4", new Dropout(0.5));

    model.putG("g_relu1", new ReLU());
    model.putG("g_relu2", new ReLU());
    model.putG("g_relu3", new ReLU());
    model.putG("g_relu4", new ReLU());
    model.putG("g_relu5", new ReLU());
    model.putG("g_relu6", new ReLU());
    model.putG("g_relu7", new ReLU());


    model.putG("g_softmax_cross_entoropy", new SoftmaxCrossEntropy());
    model.putG("g_softmax", new Softmax());


    // outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride
    model.putG("g_conv2d1", new Conv2D(batchSize, 3, 32, 32, 3, 32, 1, 1));
    model.putG("g_conv2d2", new Conv2D(batchSize, 32, 32, 32, 3, 32, 1, 1));
    model.putG("g_conv2d3", new Conv2D(batchSize, 32, 16, 16, 3, 32, 1, 1));
    model.putG("g_conv2d4", new Conv2D(batchSize, 32, 16, 16, 3, 32, 1, 1));
    model.putG("g_conv2d5", new Conv2D(batchSize, 32, 8, 8, 3, 32, 1, 1));
    model.putG("g_conv2d6", new Conv2D(batchSize, 32, 8, 8, 3, 32, 1, 1));

    // Pooling(int width, int height, int depth, int windowWidth, int windowHeight)
    model.putG("g_pooling1", new Pooling(32, 32, 32, 2, 2, 2, 0));
    model.putG("g_pooling2", new Pooling(16, 16, 32, 2, 2, 2, 0));
    model.putG("g_pooling3", new Pooling(8, 8, 32, 2, 2, 2, 0));


    // Prepare optimizer
    OptimizerAdam optimizer(&model, learning_rate);
    optimizer.init();


    cout << "start training ..." << endl;
    for(int k=0; k<epochNums; k++){

        start = std::chrono::system_clock::now();

        std::random_shuffle(bds.begin(), bds.end());

        float sum_loss = 0.0;
        float sum_loss_tmp = 0.0;
        float accurecy = 0.0;
        float accurecy_tmp = 0.0;


        for(int i=0; i<totalSampleSize/batchSize; i++){

            PVariable x(new Variable(i_size, batchSize, false));
            PVariable d(new Variable(o_size, batchSize, false));

            // create mini-batch =========================
            float *X = bds.at(i)->getX();
            float *D = bds.at(i)->getD();
            asMatrix(x, X);
            asMatrix(d, D);

            PVariable h = forward_one_step(model, x, true);

            PVariable loss = model.G("g_softmax_cross_entoropy")->forward(h, d);

            float l = loss->val();
            sum_loss += l;
            sum_loss_tmp += l;

            loss->backward();

            optimizer.update();

            float ac = getAccurecy(model.G("g_softmax"), h, d, batchSize);
            accurecy += ac;
            accurecy_tmp += ac;


            if ((i+1) % disp_num == 0){
                cout << (i+1) << " loss:" << sum_loss_tmp/((float)disp_num) << " accurecy:" << accurecy_tmp/((float)disp_num)*100 << "%" << endl;
                accurecy_tmp = 0.0;
                sum_loss_tmp = 0.0;
            }

            model.unchain();
            model.zero_grads();
        }


        end = std::chrono::system_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
        float loss_mean = sum_loss/((float)totalSampleSize/batchSize);
        float accurecy_mean = accurecy/((float)totalSampleSize/batchSize);
        cout << "epoch:" << k+1 << " loss:" << loss_mean << " accurecy:"  << accurecy_mean*100 << "% time:" << elapsed << "s" << endl;

        float test_loss = 0.0;
        float test_acc = test_accurecy(model, bds_test, i_size, o_size, totalTestSize, batchSize, &test_loss);
        cout << "test loss:" << test_loss << " accurecy:" << test_acc*100 << "%" << endl;
        start = std::chrono::system_clock::now();

    }

    cout << "saving model..." << endl;
    model.save("cnn_test.model");


/*
    cout << "loading model..." << endl;
    Model model_train;
    model_train.load("cnn_test.model");
    cout << "loaded" << endl;

    float test_loss = 0.0;
    float test_acc = test_accurecy(model_train, bds_test, i_size, o_size, totalTestSize, batchSize, &test_loss);
    cout << "test loss:" << test_loss << " accurecy:" << test_acc*100 << "%" << endl;
*/
}

