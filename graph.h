#ifndef GRAPH_H
#define GRAPH_H

#include <random>
#include <vector>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

#include "function.h"


using namespace std;

class Graph {
public:

    vector<PFunction > funcs_chain;

    Graph();
    virtual ~Graph();

    virtual vector<Variable *> getParams();

    virtual void toHostArray();
    virtual void fromHostArray();

    void init();
    void remove_chain();

    virtual PVariable forward(PVariable input);
    virtual PVariable forward(PVariable x, PVariable t);

    virtual void zero_grads();
    virtual void reset_state();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {
    }
};



class Linear : public Graph {

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & w;
        ar & b;
        ar & noBias;
        ar & isTranpose;
    }


public:

    Variable *w, *b;
    bool noBias = false;
    bool isTranpose = false;


    Linear();

    Linear(int output_size, int input_size, bool no_bias = false);

    Linear(Variable *w, bool isTranspose = false);

    ~Linear();

    vector<Variable *> getParams();

    PVariable forward(PVariable v);

    PVariable forward(PVariable x, PVariable t);

    void zero_grads();

    void toHostArray();
    void fromHostArray();
};

class SparseLinear : public Graph {

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & w;
        ar & b;
        ar & noBias;
        ar & beta;
        ar & p;
        ar & ph;
    }


public:

    Variable *w, *b;
    bool noBias = false;

    float g;
    float beta;
    float p;
    Variable *ph = NULL;

    SparseLinear();

    SparseLinear(int output_size, int input_size, bool no_bias = false, float g = 0.9, float beta = 0.1, float p = 0.05);

    ~SparseLinear();

    vector<Variable *> getParams();

    PVariable forward(PVariable v);

    PVariable forward(PVariable x, PVariable t);

    void zero_grads();

    void toHostArray();
    void fromHostArray();
};


class Sigmoid : public Graph {
public:

    Sigmoid();

    PVariable forward(PVariable v);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};


class ReLU : public Graph {
public:

    ReLU();

    PVariable forward(PVariable v);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};

class PReLU : public Graph {
public:
    Variable *a = NULL;

    PReLU();
    PReLU(int rows, int cols);
    ~PReLU();

    vector<Variable *> getParams();

    PVariable forward(PVariable v);

    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & a;
    }

};


class Tanh : public Graph {
public:

    Tanh();

    PVariable forward(PVariable v);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};


class Sqrt : public Graph {
public:

    Sqrt();

    PVariable forward(PVariable v);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};

class Inverse : public Graph {
public:

    Inverse();

    PVariable forward(PVariable v);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};




class Dropout : public Graph {
public:

    float dropout_rate = 0.0;
    bool is_train = true;

    Dropout();

    Dropout(float dropout_rate);

    PVariable forward(PVariable v);

    void isTrain(bool is_train);

        private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & dropout_rate;
    }

};

class SoftmaxCrossEntropy : public Graph {
public:

    SoftmaxCrossEntropy();

    PVariable forward(PVariable v1, PVariable v2);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};

class Softmax : public Graph {
public:
    Softmax();

    PVariable forward(PVariable v1);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};

class Plus : public Graph {
public:
    Plus();

    PVariable forward(PVariable v1, PVariable v2);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }
};


class MeanSquaredError : public Graph {
public:
    MeanSquaredError();

    PVariable forward(PVariable v1, PVariable v2);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};

class Identity : public Graph {
public:
    Identity();
    PVariable forward(PVariable v1);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
    }

};

class LSTM : public Graph {
public:


    int id = 0;
    int last_opt = 0;
    bool is_last_backward = false;

    int input_size = 0;
    int output_size = 0;


    Variable *x_w, *x_b;
    Variable *h_w, *h_b;

    PVariable c;
    PVariable c_next;

    PVariable h;


    LSTM();
    LSTM(int output_size, int input_size);

    ~LSTM();

    PVariable forward(PVariable x);


    void reset_state();
    void unchain();
    void zero_grads();


    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & x_w;
        ar & x_b;
        ar & h_w;
        ar & h_b;
        ar & input_size;
        ar & output_size;
    }

};


class FullLSTM : public Graph {
public:


    int id = 0;
    int last_opt = 0;
    bool is_last_backward = false;

    int input_size = 0;
    int output_size = 0;


    Variable *f_c_w, *f_h_w, *f_x_w, *f_x_b;
    Variable *i_c_w, *i_h_w, *i_x_w, *i_x_b;
    Variable *o_c_w, *o_h_w, *o_x_w, *o_x_b;
    Variable *g_h_w, *g_x_w, *g_x_b;


    PVariable c;
    PVariable c_next;

    PVariable f;
    PVariable f_next;

    PVariable i;
    PVariable i_next;

    PVariable o;
    PVariable o_next;

    PVariable g;
    PVariable g_next;


    PVariable h;


    FullLSTM();
    FullLSTM(int output_size, int input_size);

    ~FullLSTM();

    PVariable forward(PVariable x);


    void reset_state();
    void zero_grads();


    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);

        ar & input_size;
        ar & output_size;

        ar & f_c_w;
        ar & f_h_w;
        ar & f_x_w;
        ar & f_x_b;

        ar & i_c_w;
        ar & i_h_w;
        ar & i_x_w;
        ar & i_x_b;

        ar & o_c_w;
        ar & o_h_w;
        ar & o_x_w;
        ar & o_x_b;

        ar & g_h_w;
        ar & g_x_w;
        ar & g_x_b;
    }

};

class FullLSTM2 : public Graph {
public:


    int id = 0;
    int last_opt = 0;
    bool is_last_backward = false;

    int input_size = 0;
    int output_size = 0;

    bool batch_norm = false;
    float batch_norm_gamma = 1.0;
    float batch_norm_beta = 0.0;

    bool is_first = true;

    Variable *f_c_w, *f_h_w, *f_x_w, *f_x_b;
    Variable *i_c_w, *i_h_w, *i_x_w, *i_x_b;
    Variable *o_c_w, *o_h_w, *o_x_w, *o_x_b;
    Variable *g_h_w, *g_x_w, *g_x_b;

    Variable *x_mean_f, *x_var_f;
    Variable *x_mean_i, *x_var_i;
    Variable *x_mean_g, *x_var_g;
    Variable *x_mean_o, *x_var_o;

    Variable *gamma_f, *beta_f;
    Variable *gamma_i, *beta_i;
    Variable *gamma_g, *beta_g;
    Variable *gamma_o, *beta_o;


    bool is_train = true;
    float lambda = 0.9;

    PVariable c;
    PVariable h;


    FullLSTM2();
    FullLSTM2(int output_size, int input_size);

    ~FullLSTM2();

    vector<Variable *> getParams();

    PVariable forward(PVariable x);

    void set_train_status(bool status);


    void reset_state();
    void zero_grads();


    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);

        ar & input_size;
        ar & output_size;

        ar & batch_norm;
        ar & batch_norm_gamma;
        ar & batch_norm_beta;

        ar & f_c_w;
        ar & f_h_w;
        ar & f_x_w;
        ar & f_x_b;

        ar & i_c_w;
        ar & i_h_w;
        ar & i_x_w;
        ar & i_x_b;

        ar & o_c_w;
        ar & o_h_w;
        ar & o_x_w;
        ar & o_x_b;

        ar & g_h_w;
        ar & g_x_w;
        ar & g_x_b;


        ar & x_mean_f;
        ar & x_mean_i;
        ar & x_mean_g;
        ar & x_mean_o;
        ar & x_var_f;
        ar & x_var_i;
        ar & x_var_g;
        ar & x_var_o;

        ar & gamma_f;
        ar & beta_f;
        ar & gamma_i;
        ar & beta_i;
        ar & gamma_g;
        ar & beta_g;
        ar & gamma_o;
        ar & beta_o;


    }

};



class GRU : public Graph {
public:


    int id = 0;
    int last_opt = 0;
    bool is_last_backward = false;

    int input_size = 0;
    int output_size = 0;


    Variable *w_r, *u_r, *b_r;
    Variable *w_z, *u_z, *b_z;
    Variable *w_g, *u_g, *b_g;


    PVariable h;

    PVariable ones;

    GRU();
    GRU(int output_size, int input_size);

    ~GRU();

    vector<Variable *> getParams();

    PVariable forward(PVariable x);


    void reset_state();
    void zero_grads();


    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & w_r;
        ar & u_r;
        ar & b_r;
        ar & w_z;
        ar & u_z;
        ar & b_z;
        ar & w_g;
        ar & u_g;
        ar & b_g;

        ar & input_size;
        ar & output_size;
    }

};


class BatchNorm : public Graph {
public:

    Variable *gamma = NULL;
    Variable *beta = NULL;

    Variable *x_mean = NULL;
    Variable *x_var = NULL;

    float lambda = 0.999;
    bool is_train = true;
    bool is_first = true;

    int element_size, channel_num;

    BatchNorm();
    BatchNorm(int element_size, int channel_num, float decay);
    ~BatchNorm();

    vector<Variable *> getParams();

    PVariable forward(PVariable x);

    void setTrainStatus(bool status);

    void zero_grads();

    void toHostArray();
    void fromHostArray();


private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & gamma;
        ar & beta;
        ar & x_mean;
        ar & x_var;
        ar & element_size;
        ar & channel_num;
    }

};

class Conv2D : public Graph {
public:


    int batch_num, channel_num, w_size, h_size, filter_size, filter_num, stride, padding;


    Variable *w = NULL;
    Variable *b = NULL;

    vector<Variable *> getParams();


    Conv2D();
    Conv2D(int batch_num, int channel_num, int w_size, int h_size, int filter_size, int filter_num, int stride, int padding);
    ~Conv2D();

    PVariable forward(PVariable x);

    void zero_grads();

    void toHostArray();
    void fromHostArray();


private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);
        ar & w;
        ar & b;

        ar & batch_num;
        ar & channel_num;
        ar & w_size;
        ar & h_size;
        ar & filter_size;
        ar & filter_num;
            ar & stride;
            ar & padding;
    }

};


class Pooling : public Graph {
public:

    int width, height, depth, windowWidth, windowHeight, stride, padding;


    Pooling();
    Pooling(int width, int height, int depth, int windowWidth, int windowHeight, int stride, int padding);

    PVariable forward(PVariable x);



        private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Graph>(*this);

        ar & width;
        ar & height;
        ar & depth;
        ar & windowWidth;
        ar & windowHeight;
            ar & stride;
            ar & padding;
    }

};


#endif //GRAPH_H
