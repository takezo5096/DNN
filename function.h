#include <list>
#include <random>
#include <vector>
#include <map>


#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

using namespace std;

#ifndef _FUNCTION_
#define _FUNCTION_

#include "variable.h"



extern map<Variable *, bool> obj_pool2;
extern int count_function;
extern int count_variable;

class Function {
public:


    vector<PVariable> inputs;
    vector<PVariable> outputs;


    int id = -1;
    string name;
    string custom_name;
    int inner_count = 0;

    Function();
    virtual ~Function();


    virtual PVariable forward(PVariable input);
    virtual PVariable forward(PVariable x, PVariable t);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3, PVariable input4);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3, PVariable input4,
                              PVariable input5, PVariable input6, PVariable input7, PVariable input8,
                              PVariable input9, PVariable input10, PVariable input11, PVariable input12
    );

    virtual void backward(cuMat &p_grad);

    virtual PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    virtual void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);


    void init();

    void clip_grad(Variable *v);

    virtual void reset_state();

        private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {
    }

};


 class FunctionPlus : public Function {
 public:
 FunctionPlus() ;
 PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
 void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
 };
 class FunctionMinus : public Function {
 public:
 FunctionMinus() ;
 PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
 void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
 };
 class FunctionMul : public Function {
 public:
 FunctionMul() ;
 PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
 void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
 };
 class FunctionSin : public Function {
     public:
         PVariable rr = NULL;
         FunctionSin() ;
         PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
         void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
 };
 class FunctionCos : public Function {
      public:
          PVariable rr = NULL;
          FunctionCos() ;
          PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
          void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
  };
 class FunctionLog : public Function {
     public:
         FunctionLog() ;
         PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
         void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
 };

class FunctionSqrt : public Function {
public:
    FunctionSqrt() ;
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionInverse : public Function {
public:
    FunctionInverse() ;
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};



class FunctionLinear: public Function {

public:
    Variable *w;
    Variable *b;
    cuMat i1;

    bool noBias = false;
    bool isTranspose = false;

    FunctionLinear();
    FunctionLinear(Variable *w, Variable *b, bool isTranspose =false);
    FunctionLinear(Variable *w, bool isTranspose = false);
    FunctionLinear(int output_size, int input_size);
    FunctionLinear(int output_size, int input_size, bool no_bias);
    //~FunctionLinear();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Function>(*this);
        ar & w;
        ar & b;
        ar & i1;
        ar & noBias;
        ar & isTranspose;
    }

};


class FunctionSparseLinear : public Function {
public:
    Variable *w;
    Variable *b;
    cuMat i1;

    bool noBias = false;

    float beta;
    float p;
    Variable *ph;

    FunctionSparseLinear();
    FunctionSparseLinear(Variable *w, Variable *b, float beta, float p, Variable *ph);
    FunctionSparseLinear(Variable *w, float beta, float p, Variable *ph);

    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
    void toHostArray();
    void fromHostArray();
    
private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Function>(*this);
        ar & w;
        ar & b;
        ar & i1;
        ar & noBias;
        ar & beta;
        ar & p;
        ar & ph;
    }

};


class FunctionEmbed: public Function {

public:
    Variable w;
    Variable b;
    cuMat i1;

    cuMat wt;
    cuMat rt;
    cuMat rtmp;

    bool noBias = false;

    FunctionEmbed();
    FunctionEmbed(int output_size, int input_size, bool no_bias);
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
    void toHostArray();
    void fromHostArray();

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Function>(*this);
        ar & w;
        ar & b;
        ar & i1;
        ar & noBias;
    }

};







class FunctionReLU: public Function {
public:
    PVariable rr = NULL;
    FunctionReLU();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionPReLU: public Function {
public:
    Variable *a;
    PVariable xd = NULL;
    PVariable ad = NULL;
    FunctionPReLU();
    FunctionPReLU(Variable *);
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};


class FunctionSigmoid: public Function {
public:
    PVariable rr = NULL;
    FunctionSigmoid();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};
class FunctionTanh: public Function {
public:
    PVariable rr = NULL;
    FunctionTanh();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};



 class FunctionSoftmax : public Function {
 public:
 FunctionSoftmax() ;
 PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionSoftmaxCrossEntropy: public Function {
public:
    PVariable rr = NULL;
    PVariable rr2 = NULL;
    PVariable rr3 = NULL;
    cuMat loss;
    cuMat *seed = NULL;

    FunctionSoftmaxCrossEntropy();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};
class FunctionMeanSquaredError: public Function {
public:
    PVariable rr = NULL;
    cuMat loss;

    FunctionMeanSquaredError();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

class FunctionDropout: public Function {
public:
    PVariable rr = NULL;
    PVariable rr2 = NULL;
    float p = 0.0;

    FunctionDropout(float p);
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};


class FunctionIdentity: public Function {
public:
    FunctionIdentity();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};



class FunctionLSTM: public Function {
public:

    cuMat i, f, g, o;

    FunctionLSTM();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);

    void backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs);

    void splitMat(int offset, cuMat &target, cuMat &i, cuMat &f, cuMat &g, cuMat &o);
    void jonMat(int offset, cuMat &target, cuMat &i, cuMat &f, cuMat &g, cuMat &o);

};

class FunctionFullLSTM: public Function {
public:

    Variable *f_c_w,  *f_h_w,  *f_x_w,  *f_x_b,
    *i_c_w,  *i_h_w,  *i_x_w,  *i_x_b,
    *o_c_w,  *o_h_w,  *o_x_w,  *o_x_b,
    *g_h_w,  *g_x_w,  *g_x_b;

    cuMat f, i, g, o;
    cuMat f_hat, i_hat, g_hat, o_hat;

    FunctionFullLSTM(Variable *f_c_w, Variable *f_h_w, Variable *f_x_w, Variable *f_x_b,
            Variable *i_c_w, Variable *i_h_w, Variable *i_x_w, Variable *i_x_b,
            Variable *o_c_w, Variable *o_h_w, Variable *o_x_w, Variable *o_x_b,
            Variable *g_h_w, Variable *g_x_w, Variable *g_x_b);


    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);

    void backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs);

};


class FunctionGRU: public Function {
public:

    Variable *w_r, *u_r, *b_r,
            *w_z, *u_z, *b_z,
            *w_g, *u_g, *b_g;

    cuMat r, z, g;
    cuMat r_hat, z_hat, g_hat;

    FunctionGRU(Variable *w_r, Variable *u_r, Variable *b_r, Variable *w_z, Variable *u_z, Variable *b_z, Variable *w_g, Variable *u_g, Variable *b_g);

    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);

    void backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs);

};


class FunctionBatchNorm: public Function {

public:
    Variable *gamma, *beta;

    vector<cuMat> xhat, rmu, xmu, ivar, sqrtvar, var;

    bool is_train = true;
    Variable *x_mean, *x_var;

    int element_size, channel_num;


public:
    FunctionBatchNorm(int element_size, int channel_num, Variable *gamma, Variable *beta, Variable *x_mean, Variable *x_var);

    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);

    void backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs);

};

class FunctionConv2D: public Function {

public:
    Variable *w, *b;

    int batch_num, channel_num, w_size, h_size, filter_size, filter_num,  stride, padding;

    int outputDim_w, outputDim_h;

    vector<cuMat> cols;

    Variable *ones;

    FunctionConv2D(Variable *w, Variable *b, int batch_num, int channel_num, int w_size, int h_size, int filter_size, int filter_num,  int stride, int padding);

    ~FunctionConv2D();

    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);

    void backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs);


    cuMat forward_one(cuMat &data);
    cuMat backward_one(cuMat &data, cuMat &p_grad);

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Function>(*this);
        ar & ones;
    }

};


class FunctionPooling: public Function {

public:

    int width, height, depth, windowWidth, windowHeight,  stride, padding;

    FunctionPooling(int width, int height, int depth, int windowWidth, int windowHeight, int stride, int padding);

    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);

    void backward(cuMat &gh, vector<PVariable> &inputs, vector<PVariable> &outputs);

};



using PFunction = shared_ptr<Function>;

#endif

