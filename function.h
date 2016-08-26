#include <list>
#include <random>
//#include <stack>
#include <vector>
#include <map>


#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

using namespace std;

#ifndef _FUNCTION_
#define _FUNCTION_

#include "variable.h"


class FunctionParam {
public:
    vector<PVariable> inputs;
    vector<PVariable> outputs;
};


class Function {
public:

    vector<FunctionParam *> paramsStack;

    vector<int> paramsStackNums;


    int id;
    string name;


    Function();
    virtual ~Function();


    void createParams(vector<PVariable> &inputs, vector<Function *> &funcs);

    virtual PVariable forward(PVariable input);
    virtual PVariable forward(PVariable x, PVariable t);
    virtual void backward(cuMat &p_grad);

    virtual PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    virtual void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);

    virtual bool popParamStack();
    //virtual void clearParamStack(bool isPop);

    void init();

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



class FunctionLinear: public Function {

public:
    Variable w;
    Variable b;
    cuMat i1;

    bool noBias = false;

    FunctionLinear();
    FunctionLinear(Variable &w, Variable &b);
    FunctionLinear(int output_size, int input_size);
    FunctionLinear(int output_size, int input_size, bool no_bias);
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



// Gated Recurrent Unit
// http://arxiv.org/pdf/1406.1078v3.pdf
class FunctionGRU: public Function {
public:

    //PVariable e1;
    PVariable s_h;
    int input_size=0;
    int output_size=0;

    Function *f_sigmoid_r;
    Function *f_plus_r;
    Function *wr_x;
    Function *ur_h;

    Function *f_sigmoid_z;
    Function *f_plus_z;
    Function *wz_x;
    Function *uz_h;

    Function *f_mul_h;
    Function *f_tanh;
    Function *f_plus_h;

    Function *w_x;
    Function *u_h;

    Function *f_plus;
    Function *f_mul1;
    Function *f_mul2;
    Function *f_minus;




    FunctionGRU();
    FunctionGRU(int output_size, int input_size);
    ~FunctionGRU();
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
    void reset_state();
private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Function>(*this);
        ar & input_size;
        ar & output_size;

        ar & f_sigmoid_r;
        ar & f_plus_r;
        ar & wr_x;
        ar & ur_h;

        ar & f_sigmoid_z;
        ar & f_plus_z;
        ar & wz_x;
        ar & uz_h;

        ar & f_mul_h;
        ar & f_tanh;
        ar & f_plus_h;

        ar & w_x;
        ar & u_h;

        ar & f_plus;
        ar & f_mul1;
        ar & f_mul2;
        ar & f_minus;
    }

};

// Long Short Term Memory
// http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
class FunctionLSTM: public Function {
public:
    int input_size=0;
    int output_size=0;

    PVariable s_h;
    PVariable c_h;

    Function *x_i;
    Function *w_i;
    Function *c_i;
    Function *f_plus_i;
    Function *f_plus_i2;
    Function *f_sig_i;

    Function *x_f;
    Function *w_f;
    Function *c_f;
    Function *f_plus_f;
    Function *f_plus_f2;
    Function *f_sig_f;

    Function *x_o;
    Function *w_o;
    Function *c_o;
    Function *f_plus_o;
    Function *f_plus_o2;
    Function *f_sig_o;

    Function *x_g;
    Function *w_g;
    Function *f_plus_g;
    Function *f_tan_g;

    Function *f_mul1_c;
    Function *f_mul2_c;
    Function *f_plus_c;

    Function *f_mul_s;
    Function *f_tan_s;




    FunctionLSTM();
    ~FunctionLSTM();
    FunctionLSTM(int output_size, int input_size);
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
    void reset_state();
private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {

        ar & boost::serialization::base_object<Function>(*this);
        ar & x_i;
        ar & w_i;
        ar & f_plus_i;
        ar & f_sig_i;

        ar & x_f;
        ar & w_f;
        ar & f_plus_f;
        ar & f_sig_f;

        ar & x_o;
        ar & w_o;
        ar & f_plus_o;
        ar & f_sig_o;

        ar & x_g;
        ar & w_g;
        ar & f_plus_g;
        ar & f_tan_g;

        ar & f_mul1_c;
        ar & f_mul2_c;
        ar & f_plus_c;

        ar & f_mul_s;
        ar & f_tan_s;

    }

};



class FunctionReLU: public Function {
public:
    PVariable rr = NULL;
    FunctionReLU();
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
    float p = 0;

    FunctionDropout(float p);
    PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    void backward(cuMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);
};

#endif

