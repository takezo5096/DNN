#include "graph.h"

using namespace std;


Graph::Graph(){
}

Graph::~Graph() {
    init();
}

void Graph::init(){
    remove_chain();
}

void Graph::remove_chain(){

    funcs_chain.clear();
}

vector<Variable *> Graph::getParams() {
    vector < Variable * > params;

    return params;
}

PVariable Graph::forward(PVariable input) {

}
PVariable Graph::forward(PVariable x, PVariable t) {

}

void Graph::zero_grads() {}

void Graph::reset_state() {}


void Graph::toHostArray(){}
void Graph::fromHostArray(){}



Linear::Linear() : Graph() {

}
Linear::Linear(int output_size, int input_size, bool no_bias) : Graph() {

        noBias = no_bias;

        this->w = new Variable(output_size, input_size);
        this->w->randoms(0., sqrt(2.0/((float)input_size)));

        if (!noBias){
            this->b = new Variable(output_size, 1);
        }

    }

Linear::Linear(Variable *w, bool isTranspose) : Graph() {
    this->w = w;
    this->isTranpose = isTranspose;

    int output_size = w->data.rows;
    if (isTranspose){
        output_size = w->data.cols;
    }
    this->b = new Variable(output_size, 1);
}

Linear::~Linear(){
    if (this->w != NULL) delete this->w;
    if (this->b != NULL) delete this->b;
}


vector<Variable *> Linear::getParams(){
    vector<Variable *> params;
    params.push_back(w);
    if (!noBias) params.push_back(b);

    return params;
}

PVariable Linear::forward(PVariable v){

    Function *f;
    if (noBias)
        f = new FunctionLinear(w, isTranpose);
    else
        f = new FunctionLinear(w, b, isTranpose);

    PFunction pf(f);

    funcs_chain.push_back(pf);

    return pf->forward(v);
}

void Linear::zero_grads() {
    w->zero_grad();
    if (!noBias) b->zero_grad();
}

void Linear::toHostArray(){
    w->data.toHostArray();
    if (!noBias) b->data.toHostArray();
}
void Linear::fromHostArray(){
    w->data.fromHostArray();
    if (!noBias) b->data.fromHostArray();

}

PVariable Linear::forward(PVariable x, PVariable t){}


// SparseLinear ------------------------------------------
SparseLinear::SparseLinear() : Graph() {

}
SparseLinear::SparseLinear(int output_size, int input_size, bool no_bias, float g, float beta, float p) : Graph() {

    noBias = no_bias;

    this->w = new Variable(output_size, input_size);
    this->w->randoms(0., sqrt(1.0/((float)input_size)));

    if (!noBias){
        this->b = new Variable(output_size, 1);
    }

    this->g = g;
    this->beta = beta;
    this->p = p;


}

SparseLinear::~SparseLinear(){
    if (this->w != NULL) delete this->w;
    if (this->b != NULL) delete this->b;
}


vector<Variable *> SparseLinear::getParams(){
    vector<Variable *> params;
    params.push_back(w);
    if (!noBias) params.push_back(b);

    return params;
}

PVariable SparseLinear::forward(PVariable v){

    if (this->ph == NULL){
        this->ph = new Variable(this->w->data.rows, v->data.cols);
    }

    Function *f;
    if (noBias)
        f = new FunctionSparseLinear(w, beta, p, ph);
    else
        f = new FunctionSparseLinear(w, b, beta, p, ph);
    PFunction pf(f);
    funcs_chain.push_back(pf);

    PVariable r = pf->forward(v);

    cuMat np = 1.0/v->data.rows * r->data.batch_sum().vec_to_mat(v->data.cols);

    ph->data = g * ph->data + (1.0-g) * np;

    return r;
}

void SparseLinear::zero_grads() {
    w->zero_grad();
    if (!noBias) b->zero_grad();
}

void SparseLinear::toHostArray(){
    w->data.toHostArray();
    if (!noBias) b->data.toHostArray();
}
void SparseLinear::fromHostArray(){
    w->data.fromHostArray();
    if (!noBias) b->data.fromHostArray();

}

PVariable SparseLinear::forward(PVariable x, PVariable t){}


Sigmoid::Sigmoid() : Graph() {

}
PVariable Sigmoid::forward(PVariable v){
    Function *f = new FunctionSigmoid();
    PFunction pf(f);
    funcs_chain.push_back(pf);
    return pf->forward(v);
}


ReLU::ReLU() : Graph() {

}
PVariable ReLU::forward(PVariable v){
        Function *f = new FunctionReLU();
        PFunction pf(f);
        funcs_chain.push_back(pf);
        return pf->forward(v);
}

PReLU::PReLU() : Graph() {

}
PReLU::PReLU(int rows, int cols) {
    a = new Variable(rows, cols);
    // init weight using 0.25
    // https://arxiv.org/pdf/1502.01852.pdf
    a->data.fill(0.25);

}
PVariable PReLU::forward(PVariable v){

    Function *f = new FunctionPReLU(this->a);
    PFunction pf(f);
    funcs_chain.push_back(pf);
    return pf->forward(v);
}

PReLU::~PReLU(){
    if (a != NULL) delete a;
}

vector<Variable *> PReLU::getParams(){
    vector<Variable *> params;
    params.push_back(a);
    return params;
}

void PReLU::toHostArray(){
    a->data.toHostArray();
}
void PReLU::fromHostArray(){
    a->data.fromHostArray();
}




Tanh::Tanh() : Graph() {

}
PVariable Tanh::forward(PVariable v){
    Function *f = new FunctionTanh();
    PFunction pf(f);
    funcs_chain.push_back(pf);
    return pf->forward(v);
}


Sqrt::Sqrt() : Graph() {

}
PVariable Sqrt::forward(PVariable v){
    Function *f = new FunctionSqrt();
    PFunction pf(f);
    funcs_chain.push_back(pf);
    return pf->forward(v);
}

Inverse::Inverse() : Graph() {

}
PVariable Inverse::forward(PVariable v){
    Function *f = new FunctionInverse();
    PFunction pf(f);
    funcs_chain.push_back(pf);
    return pf->forward(v);
}




Dropout::Dropout() : Graph() {
}

Dropout::Dropout(float dropout_rate) : Graph() {
    this->dropout_rate = dropout_rate;
}
PVariable Dropout::forward(PVariable v){
    if (this->is_train) {
        Function *f = new FunctionDropout(dropout_rate);
        PFunction pf(f);
        funcs_chain.push_back(pf);
        return pf->forward(v);
    }
    else{
        return v;
    }
}

void Dropout::isTrain(bool is_train){
    this->is_train = is_train;
}




SoftmaxCrossEntropy::SoftmaxCrossEntropy() : Graph() {

}

PVariable SoftmaxCrossEntropy::forward(PVariable v1, PVariable v2) {

        Function *f = new FunctionSoftmaxCrossEntropy();
        PFunction pf(f);
        funcs_chain.push_back(pf);

        return pf->forward(v1, v2);
}


Softmax::Softmax() : Graph(){

}
PVariable Softmax::forward(PVariable v1) {

        Function *f = new FunctionSoftmax();
        PFunction pf(f);
        funcs_chain.push_back(pf);

        return pf->forward(v1);
}


MeanSquaredError::MeanSquaredError() : Graph() {

}

PVariable MeanSquaredError::forward(PVariable v1, PVariable v2) {

    Function *f = new FunctionMeanSquaredError();
    PFunction pf(f);
    funcs_chain.push_back(pf);

    return pf->forward(v1, v2);
}


Plus::Plus() {

}
PVariable Plus::forward(PVariable v1, PVariable v2) {
    Function *f = new FunctionPlus();
    PFunction pf(f);
    funcs_chain.push_back(pf);

    return pf->forward(v1, v2);
}


Identity::Identity() : Graph() {

}
PVariable Identity::forward(PVariable v1) {
    Function *f = new FunctionIdentity();
    PFunction pf(f);
    funcs_chain.push_back(pf);

    return pf->forward(v1);
}


LSTM::LSTM() : Graph() {

}
LSTM::LSTM(int output_size, int input_size) {

    this->output_size = output_size;
    this->input_size = input_size;


    x_w = new Variable(output_size*4, input_size);
    x_b = new Variable(output_size*4, 1);
    x_w->randoms(0., sqrt((1./(float)input_size)));

    h_w = new Variable(output_size*4, output_size);
    h_b = new Variable(output_size*4, 1);
    h_w->randoms(0., sqrt((1./(float)output_size)));


}
LSTM::~LSTM() {
    delete x_w; delete x_b;
    delete h_w; delete h_b;
}


PVariable LSTM::forward(PVariable x) {

    // prepare functions -------------------------
    Function *f_x = new FunctionLinear(x_w, x_b);
    PFunction p_f_x(f_x);
    funcs_chain.push_back(p_f_x);

    Function *f_h = new FunctionLinear(h_w, h_b);
    PFunction p_f_h(f_h);
    funcs_chain.push_back(p_f_h);

    Function *f_plus = new FunctionPlus();
    PFunction p_f_plus(f_plus);
    funcs_chain.push_back(p_f_plus);


    Function *f_lstm = new FunctionLSTM();
    PFunction p_f_lstm(f_lstm);
    funcs_chain.push_back(p_f_lstm);
    //--------------------------------------------


    if (c.get() == NULL || c->data.rows == 0 || c->data.cols != x->data.cols) {
        c = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }
    if (c_next.get() == NULL || c_next->data.rows ==0 || c_next->data.cols != x->data.cols) {
        c_next = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);

    }

    if (h.get() == NULL || h->data.rows == 0 || h->data.cols != x->data.cols) {
        h = PVariable(variable_construct(output_size *4, x->data.cols), variable_destroy);

        h->opt = id;

        last_opt = id;

        h->last_opt = &last_opt;
        h->is_last_backward = &is_last_backward;
        id++;
    }



    PVariable h_d = f_plus->forward(f_x->forward(x), f_h->forward(h));

    h = f_lstm->forward(h_d, c, c_next);

    c = c_next;

    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;

    id++;
    return h;


}

void LSTM::reset_state(){

    x_w->grad *= 0;
    x_b->grad *= 0;
    h_w->grad *= 0;
    h_b->grad *= 0;

    c->zeros();
    c->unchain();
    c_next->zeros();
    c_next->unchain();

    h->zeros();
    h->unchain();

    last_opt = 0;
    is_last_backward = false;

    id = 0;


    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;
    id++;
}

void LSTM::unchain() {
}

void LSTM::toHostArray(){
    x_w->data.toHostArray();
    x_b->data.toHostArray();
    h_w->data.toHostArray();
    h_b->data.toHostArray();

}

void LSTM::zero_grads() {
    x_w->zero_grad();
    x_b->zero_grad();
    h_w->zero_grad();
    h_b->zero_grad();

    c->zero_grad();
    c_next->zero_grad();

    h->zero_grad();
}
void LSTM::fromHostArray(){

    x_w->data.fromHostArray();
    x_b->data.fromHostArray();
    h_w->data.fromHostArray();
    h_b->data.fromHostArray();


}


FullLSTM::FullLSTM() : Graph() {

}
FullLSTM::FullLSTM(int output_size, int input_size) {

    this->output_size = output_size;
    this->input_size = input_size;


    //------------------------------------------------------
    f_c_w = new Variable(output_size, output_size);
    f_h_w = new Variable(output_size, output_size);
    f_x_w = new Variable(output_size, input_size);
    f_x_b = new Variable(output_size, 1);

    f_c_w->randoms(0., sqrt((1./(float)output_size)));
    f_h_w->randoms(0., sqrt((1./(float)output_size)));
    f_x_w->randoms(0., sqrt((1./(float)input_size)));
    // initialize forget gate bias to 1
    // for more detail, http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    f_x_b->data.ones();

    //------------------------------------------------------
    i_c_w = new Variable(output_size, output_size);
    i_h_w = new Variable(output_size, output_size);
    i_x_w = new Variable(output_size, input_size);
    i_x_b = new Variable(output_size, 1);

    i_c_w->randoms(0., sqrt((1./(float)input_size)));
    i_h_w->randoms(0., sqrt((1./(float)input_size)));
    i_x_w->randoms(0., sqrt((1./(float)input_size)));

    //------------------------------------------------------
    o_c_w = new Variable(output_size, output_size);
    o_h_w = new Variable(output_size, output_size);
    o_x_w = new Variable(output_size, input_size);
    o_x_b = new Variable(output_size, 1);

    o_c_w->randoms(0., sqrt((1./(float)output_size)));
    o_h_w->randoms(0., sqrt((1./(float)output_size)));
    o_x_w->randoms(0., sqrt((1./(float)input_size)));

    //------------------------------------------------------
    g_h_w = new Variable(output_size, output_size);
    g_x_w = new Variable(output_size, input_size);
    g_x_b = new Variable(output_size, 1);

    g_h_w->randoms(0., sqrt((1./(float)output_size)));
    g_x_w->randoms(0., sqrt((1./(float)input_size)));
}
FullLSTM::~FullLSTM() {
    delete f_c_w; delete f_h_w; delete f_x_w; delete f_x_b;
    delete i_c_w; delete i_h_w; delete i_x_w; delete i_x_b;
    delete o_c_w; delete o_h_w; delete o_x_w; delete o_x_b;
    delete g_h_w; delete g_x_w; delete g_x_b;

}

PVariable FullLSTM::forward(PVariable x) {


    // prepare function
    Function *f_lstm = new FunctionFullLSTM(f_c_w, f_h_w, f_x_w, f_x_b,
                                            i_c_w, i_h_w, i_x_w, i_x_b,
                                            o_c_w, o_h_w, o_x_w, o_x_b,
                                            g_h_w, g_x_w, g_x_b
    );
    PFunction p_f_lstm(f_lstm);
    funcs_chain.push_back(p_f_lstm);
    //--------------------------------------------


    if (c.get() == NULL || c->data.rows == 0 || c->data.cols != x->data.cols) {
        c = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }
    if (c_next.get() == NULL || c_next->data.rows ==0 || c_next->data.cols != x->data.cols) {
        c_next = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }

    if (f.get() == NULL || f->data.rows == 0 || f->data.cols != x->data.cols) {
        f = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }
    if (f_next.get() == NULL || f_next->data.rows ==0 || f_next->data.cols != x->data.cols) {
        f_next = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }

    if (i.get() == NULL || i->data.rows == 0 || i->data.cols != x->data.cols) {
        i = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }
    if (i_next.get() == NULL || i_next->data.rows ==0 || i_next->data.cols != x->data.cols) {
        i_next = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }

    if (o.get() == NULL || o->data.rows == 0 || o->data.cols != x->data.cols) {
        o = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }
    if (o_next.get() == NULL || o_next->data.rows ==0 || o_next->data.cols != x->data.cols) {
        o_next = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }

    if (g.get() == NULL || g->data.rows == 0 || g->data.cols != x->data.cols) {
        g = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }
    if (g_next.get() == NULL || g_next->data.rows ==0 || g_next->data.cols != x->data.cols) {
        g_next = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }


    if (h.get() == NULL || h->data.rows == 0 || h->data.cols != x->data.cols) {
        h = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
        h->opt = id;

        last_opt = id;

        h->last_opt = &last_opt;
        h->is_last_backward = &is_last_backward;
        id++;
    }

    h = f_lstm->forward(x, h, c, c_next, f, f_next, i, i_next, o, o_next, g, g_next);

    c = c_next;
    f = f_next;
    i = i_next;
    o = o_next;
    g = g_next;

    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;
    id++;
    return h;
}


void FullLSTM::reset_state(){


    if (f_c_w->grad.mDevice != NULL) f_c_w->grad *= 0;
    if (f_h_w->grad.mDevice != NULL) f_h_w->grad *= 0;
    if (f_x_w->grad.mDevice != NULL) f_x_w->grad *= 0;
    if (f_x_b->grad.mDevice != NULL) f_x_b->grad *= 0;

    if (i_c_w->grad.mDevice != NULL) i_c_w->grad *= 0;
    if (i_h_w->grad.mDevice != NULL) i_h_w->grad *= 0;
    if (i_x_w->grad.mDevice != NULL) i_x_w->grad *= 0;
    if (i_x_b->grad.mDevice != NULL) i_x_b->grad *= 0;

    if (o_c_w->grad.mDevice != NULL) o_c_w->grad *= 0;
    if (o_h_w->grad.mDevice != NULL) o_h_w->grad *= 0;
    if (o_x_w->grad.mDevice != NULL) o_x_w->grad *= 0;
    if (o_x_b->grad.mDevice != NULL) o_x_b->grad *= 0;

    if (g_h_w->grad.mDevice != NULL) g_h_w->grad *= 0;
    if (g_x_w->grad.mDevice != NULL) g_x_w->grad *= 0;
    if (g_x_b->grad.mDevice != NULL) g_x_b->grad *= 0;


    c->zeros();
    c->unchain();
    c_next->zeros();
    c_next->unchain();

    f->zeros();
    f->unchain();
    f_next->zeros();
    f_next->unchain();

    i->zeros();
    i->unchain();
    i_next->zeros();
    i_next->unchain();

    o->zeros();
    o->unchain();
    o_next->zeros();
    o_next->unchain();

    g->zeros();
    g->unchain();
    g_next->zeros();
    g_next->unchain();

    h->zeros();
    h->unchain();

    last_opt = 0;
    is_last_backward = false;

    id = 0;

    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;
    id++;

}

void FullLSTM::zero_grads() {
    f_c_w->zero_grad();
    f_h_w->zero_grad();
    f_x_w->zero_grad();
    f_x_b->zero_grad();

    i_c_w->zero_grad();
    i_h_w->zero_grad();
    i_x_w->zero_grad();
    i_x_b->zero_grad();

    o_c_w->zero_grad();
    o_h_w->zero_grad();
    o_x_w->zero_grad();
    o_x_b->zero_grad();

    g_h_w->zero_grad();
    g_x_w->zero_grad();
    g_x_b->zero_grad();

    c->zero_grad();
    c_next->zero_grad();

    f->zero_grad();
    f_next->zero_grad();

    i->zero_grad();
    i_next->zero_grad();

    o->zero_grad();
    o_next->zero_grad();

    g->zero_grad();
    g_next->zero_grad();

    h->zero_grad();

}


void FullLSTM::toHostArray(){


    f_c_w->data.toHostArray();
    f_h_w->data.toHostArray();
    f_x_w->data.toHostArray();
    f_x_b->data.toHostArray();

    i_c_w->data.toHostArray();
    i_h_w->data.toHostArray();
    i_x_w->data.toHostArray();
    i_x_b->data.toHostArray();

    o_c_w->data.toHostArray();
    o_h_w->data.toHostArray();
    o_x_w->data.toHostArray();
    o_x_b->data.toHostArray();

    g_h_w->data.toHostArray();
    g_x_w->data.toHostArray();
    g_x_b->data.toHostArray();

}
void FullLSTM::fromHostArray(){

    f_c_w->data.fromHostArray();
    f_h_w->data.fromHostArray();
    f_x_w->data.fromHostArray();
    f_x_b->data.fromHostArray();

    i_c_w->data.fromHostArray();
    i_h_w->data.fromHostArray();
    i_x_w->data.fromHostArray();
    i_x_b->data.fromHostArray();

    o_c_w->data.fromHostArray();
    o_h_w->data.fromHostArray();
    o_x_w->data.fromHostArray();
    o_x_b->data.fromHostArray();

    g_h_w->data.fromHostArray();
    g_x_w->data.fromHostArray();
    g_x_b->data.fromHostArray();

}



FullLSTM2::FullLSTM2() : Graph() {

}
FullLSTM2::FullLSTM2(int output_size, int input_size) {

    this->output_size = output_size;
    this->input_size = input_size;


    //------------------------------------------------------
    f_c_w = new Variable(output_size, output_size);
    f_h_w = new Variable(output_size, output_size);
    f_x_w = new Variable(output_size, input_size);
    f_x_b = new Variable(output_size, 1);

    f_c_w->randoms(0., sqrt((1./(float)output_size)));
    f_h_w->randoms(0., sqrt((1./(float)output_size)));
    f_x_w->randoms(0., sqrt((1./(float)input_size)));
    // initialize forget gate bias to 1
    // for more detail, http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    f_x_b->data.ones();

    //------------------------------------------------------
    i_c_w = new Variable(output_size, output_size);
    i_h_w = new Variable(output_size, output_size);
    i_x_w = new Variable(output_size, input_size);
    i_x_b = new Variable(output_size, 1);

    i_c_w->randoms(0., sqrt((1./(float)input_size)));
    i_h_w->randoms(0., sqrt((1./(float)input_size)));
    i_x_w->randoms(0., sqrt((1./(float)input_size)));

    //------------------------------------------------------
    o_c_w = new Variable(output_size, output_size);
    o_h_w = new Variable(output_size, output_size);
    o_x_w = new Variable(output_size, input_size);
    o_x_b = new Variable(output_size, 1);

    o_c_w->randoms(0., sqrt((1./(float)output_size)));
    o_h_w->randoms(0., sqrt((1./(float)output_size)));
    o_x_w->randoms(0., sqrt((1./(float)input_size)));

    //------------------------------------------------------
    g_h_w = new Variable(output_size, output_size);
    g_x_w = new Variable(output_size, input_size);
    g_x_b = new Variable(output_size, 1);

    g_h_w->randoms(0., sqrt((1./(float)output_size)));
    g_x_w->randoms(0., sqrt((1./(float)input_size)));


    //------------------------------------------------------
    gamma_f = new Variable(output_size, 1);
    gamma_i = new Variable(output_size, 1);
    gamma_g = new Variable(output_size, 1);
    gamma_o = new Variable(output_size, 1);
    beta_f = new Variable(output_size, 1);
    beta_i = new Variable(output_size, 1);
    beta_g = new Variable(output_size, 1);
    beta_o = new Variable(output_size, 1);

    gamma_f->randoms(0., sqrt((1./(float)output_size)));
    gamma_i->randoms(0., sqrt((1./(float)output_size)));
    gamma_g->randoms(0., sqrt((1./(float)output_size)));
    gamma_o->randoms(0., sqrt((1./(float)output_size)));


    x_mean_f = new Variable(output_size, 1);
    x_mean_i = new Variable(output_size, 1);
    x_mean_g = new Variable(output_size, 1);
    x_mean_o = new Variable(output_size, 1);
    x_var_f = new Variable(output_size, 1);
    x_var_i = new Variable(output_size, 1);
    x_var_g = new Variable(output_size, 1);
    x_var_o = new Variable(output_size, 1);

}

FullLSTM2::~FullLSTM2() {
    delete f_c_w; delete f_h_w; delete f_x_w; delete f_x_b;
    delete i_c_w; delete i_h_w; delete i_x_w; delete i_x_b;
    delete o_c_w; delete o_h_w; delete o_x_w; delete o_x_b;
    delete g_h_w; delete g_x_w; delete g_x_b;

    delete gamma_f;
    delete gamma_i;
    delete gamma_g;
    delete gamma_o;
    delete beta_f;
    delete beta_i;
    delete beta_g;
    delete beta_o;

    delete x_mean_f;
    delete x_mean_i;
    delete x_mean_g;
    delete x_mean_o;
    delete x_var_f;
    delete x_var_i;
    delete x_var_g;
    delete x_var_o;


}


vector<Variable *> FullLSTM2::getParams(){
    vector<Variable *> params;

    params.push_back(f_c_w);
    params.push_back(f_h_w);
    params.push_back(f_x_w);
    params.push_back(f_x_b);

    params.push_back(i_c_w);
    params.push_back(i_h_w);
    params.push_back(i_x_w);
    params.push_back(i_x_b);

    params.push_back(o_c_w);
    params.push_back(o_h_w);
    params.push_back(o_x_w);
    params.push_back(o_x_b);

    params.push_back(g_h_w);
    params.push_back(g_x_w);
    params.push_back(g_x_b);

    params.push_back(gamma_f);
    params.push_back(gamma_i);
    params.push_back(gamma_g);
    params.push_back(gamma_o);
    params.push_back(beta_f);
    params.push_back(beta_i);
    params.push_back(beta_g);
    params.push_back(beta_o);

    params.push_back(x_mean_f);
    params.push_back(x_mean_i);
    params.push_back(x_mean_g);
    params.push_back(x_mean_o);
    params.push_back(x_var_f);
    params.push_back(x_var_i);
    params.push_back(x_var_g);
    params.push_back(x_var_o);

    return params;
}


void FullLSTM2::set_train_status(bool status){
    is_train = status;
}

PVariable FullLSTM2::forward(PVariable x) {


    // prepare function
    PFunction p_f_x(new FunctionLinear(f_x_w, f_x_b));
    funcs_chain.push_back(p_f_x);
    PFunction p_f_h(new FunctionLinear(f_h_w));
    funcs_chain.push_back(p_f_h);
    PFunction p_f_c(new FunctionLinear(f_c_w));
    funcs_chain.push_back(p_f_c);
    PFunction p_f_sig(new FunctionSigmoid());
    funcs_chain.push_back(p_f_sig);
    PFunction p_f_sum1(new FunctionPlus());
    funcs_chain.push_back(p_f_sum1);
    PFunction p_f_sum2(new FunctionPlus());
    funcs_chain.push_back(p_f_sum2);

    PFunction p_i_x(new FunctionLinear(i_x_w, i_x_b));
    funcs_chain.push_back(p_i_x);
    PFunction p_i_h(new FunctionLinear(i_h_w));
    funcs_chain.push_back(p_i_h);
    PFunction p_i_c(new FunctionLinear(i_c_w));
    funcs_chain.push_back(p_i_c);
    PFunction p_i_sig(new FunctionSigmoid());
    funcs_chain.push_back(p_i_sig);
    PFunction p_i_sum1(new FunctionPlus());
    funcs_chain.push_back(p_i_sum1);
    PFunction p_i_sum2(new FunctionPlus());
    funcs_chain.push_back(p_i_sum2);

    PFunction p_g_x(new FunctionLinear(g_x_w, g_x_b));
    funcs_chain.push_back(p_g_x);
    PFunction p_g_h(new FunctionLinear(g_h_w));
    funcs_chain.push_back(p_g_h);
    PFunction p_g_tanh(new FunctionTanh());
    funcs_chain.push_back(p_g_tanh);
    PFunction p_g_sum(new FunctionPlus());
    funcs_chain.push_back(p_g_sum);

    PFunction p_c_mul1(new FunctionMul());
    funcs_chain.push_back(p_c_mul1);
    PFunction p_c_mul2(new FunctionMul());
    funcs_chain.push_back(p_c_mul2);
    PFunction p_c_plus(new FunctionPlus());
    funcs_chain.push_back(p_c_plus);

    PFunction p_o_x(new FunctionLinear(o_x_w, o_x_b));
    funcs_chain.push_back(p_o_x);
    PFunction p_o_h(new FunctionLinear(o_h_w));
    funcs_chain.push_back(p_o_h);
    PFunction p_o_c(new FunctionLinear(o_c_w));
    funcs_chain.push_back(p_o_c);
    PFunction p_o_sig(new FunctionSigmoid());
    funcs_chain.push_back(p_o_sig);
    PFunction p_o_sum1(new FunctionPlus());
    funcs_chain.push_back(p_o_sum1);
    PFunction p_o_sum2(new FunctionPlus());
    funcs_chain.push_back(p_o_sum2);

    PFunction p_h_tanh(new FunctionTanh());
    funcs_chain.push_back(p_h_tanh);
    PFunction p_h_mul(new FunctionMul());
    funcs_chain.push_back(p_h_mul);


    //--------------------------------------------


    if (c.get() == NULL || c->data.rows == 0 || c->data.cols != x->data.cols) {
        c = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
    }


    if (h.get() == NULL || h->data.rows == 0 || h->data.cols != x->data.cols) {
        h = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);
        h->opt = id;

        last_opt = id;

        h->last_opt = &last_opt;
        h->is_last_backward = &is_last_backward;
        id++;
    }

    PVariable f_x, i_x, g_x, o_x;

    if (batch_norm){
        // Batch Normalization for RNNs
        // 1. https://arxiv.org/pdf/1510.01378.pdf
        // 2. https://arxiv.org/pdf/1603.09025v4.pdf
        // we chose algo of 2.


        FunctionBatchNorm *f_batch_norm  = new FunctionBatchNorm(input_size, 1, gamma_f, beta_f, x_mean_f, x_var_f);
        FunctionBatchNorm *i_batch_norm  = new FunctionBatchNorm(input_size, 1, gamma_i, beta_i, x_mean_i, x_var_i);
        FunctionBatchNorm *g_batch_norm  = new FunctionBatchNorm(input_size, 1, gamma_g, beta_g, x_mean_g, x_var_g);
        FunctionBatchNorm *o_batch_norm  = new FunctionBatchNorm(input_size, 1, gamma_o, beta_o, x_mean_o, x_var_o);

        f_batch_norm->is_train = this->is_train;
        i_batch_norm->is_train = this->is_train;
        g_batch_norm->is_train = this->is_train;
        o_batch_norm->is_train = this->is_train;

        PFunction p_f_batch_norm(f_batch_norm);
        PFunction p_i_batch_norm(i_batch_norm);
        PFunction p_g_batch_norm(g_batch_norm);
        PFunction p_o_batch_norm(o_batch_norm);
        funcs_chain.push_back(p_f_batch_norm);
        funcs_chain.push_back(p_i_batch_norm);
        funcs_chain.push_back(p_g_batch_norm);
        funcs_chain.push_back(p_o_batch_norm);


        f_x = p_f_batch_norm->forward(p_f_x->forward(x));
        i_x = p_i_batch_norm->forward(p_i_x->forward(x));
        g_x = p_g_batch_norm->forward(p_g_x->forward(x));
        o_x = p_o_batch_norm->forward(p_o_x->forward(x));


        if (this->is_train) {
            float lam = lambda;
            if (is_first) {
                lam = 0.0;
                is_first = false;
            }
            x_mean_f->data = lam * x_mean_f->data + (1.0 - lam) * f_batch_norm->rmu[0];
            x_mean_i->data = lam * x_mean_i->data + (1.0 - lam) * i_batch_norm->rmu[0];
            x_mean_g->data = lam * x_mean_g->data + (1.0 - lam) * g_batch_norm->rmu[0];
            x_mean_o->data = lam * x_mean_o->data + (1.0 - lam) * o_batch_norm->rmu[0];

            x_var_f->data = lam * x_var_f->data + (1.0 - lam) * f_batch_norm->var[0];
            x_var_i->data = lam * x_var_i->data + (1.0 - lam) * i_batch_norm->var[0];
            x_var_g->data = lam * x_var_g->data + (1.0 - lam) * g_batch_norm->var[0];
            x_var_o->data = lam * x_var_o->data + (1.0 - lam) * o_batch_norm->var[0];
        }
    }
    else{
        f_x = p_f_x->forward(x);
        i_x = p_i_x->forward(x);
        g_x = p_g_x->forward(x);
        o_x = p_o_x->forward(x);
    }

    PVariable f_sum = p_f_sum1->forward(p_f_sum2->forward(f_x, p_f_h->forward(h)), p_f_c->forward(c));
    PVariable f = p_f_sig->forward(f_sum);

    PVariable i_sum = p_i_sum1->forward(p_i_sum2->forward(i_x, p_i_h->forward(h)), p_i_c->forward(c));
    PVariable i = p_i_sig->forward(i_sum);

    PVariable g_sum = p_g_sum->forward(g_x, p_g_h->forward(h));
    PVariable g = p_g_tanh->forward(g_sum);


    c = p_c_plus->forward(
            p_c_mul1->forward(i, g),
            p_c_mul2->forward(f, c)
    );



    PVariable o_sum = p_o_sum1->forward(p_o_sum2->forward(o_x, p_o_h->forward(h)), p_o_c->forward(c));
    PVariable o = p_o_sig->forward(o_sum);


    h = p_h_mul->forward(o, p_h_tanh->forward(c));



    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;
    id++;

    return h;
}


void FullLSTM2::reset_state(){

    if (f_c_w->grad.mDevice != NULL) f_c_w->grad *= 0;
    if (f_h_w->grad.mDevice != NULL) f_h_w->grad *= 0;
    if (f_x_w->grad.mDevice != NULL) f_x_w->grad *= 0;
    if (f_x_b->grad.mDevice != NULL) f_x_b->grad *= 0;

    if (i_c_w->grad.mDevice != NULL) i_c_w->grad *= 0;
    if (i_h_w->grad.mDevice != NULL) i_h_w->grad *= 0;
    if (i_x_w->grad.mDevice != NULL) i_x_w->grad *= 0;
    if (i_x_b->grad.mDevice != NULL) i_x_b->grad *= 0;

    if (o_c_w->grad.mDevice != NULL) o_c_w->grad *= 0;
    if (o_h_w->grad.mDevice != NULL) o_h_w->grad *= 0;
    if (o_x_w->grad.mDevice != NULL) o_x_w->grad *= 0;
    if (o_x_b->grad.mDevice != NULL) o_x_b->grad *= 0;

    if (g_h_w->grad.mDevice != NULL) g_h_w->grad *= 0;
    if (g_x_w->grad.mDevice != NULL) g_x_w->grad *= 0;
    if (g_x_b->grad.mDevice != NULL) g_x_b->grad *= 0;


    if (gamma_f->grad.mDevice != NULL) gamma_f->grad *= 0;
    if (gamma_i->grad.mDevice != NULL) gamma_i->grad *= 0;
    if (gamma_g->grad.mDevice != NULL) gamma_g->grad *= 0;
    if (gamma_o->grad.mDevice != NULL) gamma_o->grad *= 0;
    if (beta_f->grad.mDevice != NULL) beta_f->grad *= 0;
    if (beta_i->grad.mDevice != NULL) beta_i->grad *= 0;
    if (beta_g->grad.mDevice != NULL) beta_g->grad *= 0;
    if (beta_o->grad.mDevice != NULL) beta_o->grad *= 0;

    c->zeros();
    c->unchain();

    h->zeros();
    h->unchain();

    last_opt = 0;
    is_last_backward = false;

    id = 0;

    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;
    id++;
}

void FullLSTM2::zero_grads() {
    f_c_w->zero_grad();
    f_h_w->zero_grad();
    f_x_w->zero_grad();
    f_x_b->zero_grad();

    i_c_w->zero_grad();
    i_h_w->zero_grad();
    i_x_w->zero_grad();
    i_x_b->zero_grad();

    o_c_w->zero_grad();
    o_h_w->zero_grad();
    o_x_w->zero_grad();
    o_x_b->zero_grad();

    g_h_w->zero_grad();
    g_x_w->zero_grad();
    g_x_b->zero_grad();

    gamma_f->zero_grad();
    gamma_i->zero_grad();
    gamma_g->zero_grad();
    gamma_o->zero_grad();
    beta_f->zero_grad();
    beta_i->zero_grad();
    beta_g->zero_grad();
    beta_o->zero_grad();


    c->zero_grad();
    h->zero_grad();

}


void FullLSTM2::toHostArray(){


    f_c_w->data.toHostArray();
    f_h_w->data.toHostArray();
    f_x_w->data.toHostArray();
    f_x_b->data.toHostArray();

    i_c_w->data.toHostArray();
    i_h_w->data.toHostArray();
    i_x_w->data.toHostArray();
    i_x_b->data.toHostArray();

    o_c_w->data.toHostArray();
    o_h_w->data.toHostArray();
    o_x_w->data.toHostArray();
    o_x_b->data.toHostArray();

    g_h_w->data.toHostArray();
    g_x_w->data.toHostArray();
    g_x_b->data.toHostArray();

    x_mean_f->data.toHostArray();
    x_mean_i->data.toHostArray();
    x_mean_g->data.toHostArray();
    x_mean_o->data.toHostArray();
    x_var_f->data.toHostArray();
    x_var_i->data.toHostArray();
    x_var_g->data.toHostArray();
    x_var_o->data.toHostArray();

    gamma_f->data.toHostArray();
    beta_f->data.toHostArray();
    gamma_i->data.toHostArray();
    beta_i->data.toHostArray();
    gamma_g->data.toHostArray();
    beta_g->data.toHostArray();
    gamma_o->data.toHostArray();
    beta_o->data.toHostArray();


}
void FullLSTM2::fromHostArray(){

    f_c_w->data.fromHostArray();
    f_h_w->data.fromHostArray();
    f_x_w->data.fromHostArray();
    f_x_b->data.fromHostArray();

    i_c_w->data.fromHostArray();
    i_h_w->data.fromHostArray();
    i_x_w->data.fromHostArray();
    i_x_b->data.fromHostArray();

    o_c_w->data.fromHostArray();
    o_h_w->data.fromHostArray();
    o_x_w->data.fromHostArray();
    o_x_b->data.fromHostArray();

    g_h_w->data.fromHostArray();
    g_x_w->data.fromHostArray();
    g_x_b->data.fromHostArray();

    x_mean_f->data.fromHostArray();
    x_mean_i->data.fromHostArray();
    x_mean_g->data.fromHostArray();
    x_mean_o->data.fromHostArray();
    x_var_f->data.fromHostArray();
    x_var_i->data.fromHostArray();
    x_var_g->data.fromHostArray();
    x_var_o->data.fromHostArray();

    gamma_f->data.fromHostArray();
    beta_f->data.fromHostArray();
    gamma_i->data.fromHostArray();
    beta_i->data.fromHostArray();
    gamma_g->data.fromHostArray();
    beta_g->data.fromHostArray();
    gamma_o->data.fromHostArray();
    beta_o->data.fromHostArray();

}







GRU::GRU() : Graph() {

}
GRU::GRU(int output_size, int input_size) {

    this->output_size = output_size;
    this->input_size = input_size;


    u_r = new Variable(output_size, output_size);
    w_r = new Variable(output_size, input_size);
    b_r = new Variable(output_size, 1);

    u_z = new Variable(output_size, output_size);
    w_z = new Variable(output_size, input_size);
    b_z = new Variable(output_size, 1);

    u_g = new Variable(output_size, output_size);
    w_g = new Variable(output_size, input_size);
    b_g = new Variable(output_size, 1);

    u_r->randoms(0., sqrt((1./(float)output_size)));
    w_r->randoms(0., sqrt((1./(float)input_size)));
    u_z->randoms(0., sqrt((1./(float)output_size)));
    w_z->randoms(0., sqrt((1./(float)input_size)));
    u_g->randoms(0., sqrt((1./(float)output_size)));
    w_g->randoms(0., sqrt((1./(float)input_size)));


}
GRU::~GRU() {
    delete w_r; delete u_r, delete b_r;
    delete w_z; delete u_z, delete b_z;
    delete w_g; delete u_g, delete b_g;
}

vector<Variable *> GRU::getParams() {
    vector < Variable * > params;

    params.push_back(w_r);
    params.push_back(u_r);
    params.push_back(b_r);
    params.push_back(w_z);
    params.push_back(u_z);
    params.push_back(b_z);
    params.push_back(w_g);
    params.push_back(u_g);
    params.push_back(b_g);

    return params;
}

PVariable GRU::forward(PVariable x) {
    // prepare function
    PFunction p_f_w_r_linear(new FunctionLinear(w_r));
    funcs_chain.push_back(p_f_w_r_linear);
    PFunction p_f_u_r_linear(new FunctionLinear(u_r, b_r));
    funcs_chain.push_back(p_f_u_r_linear);
    PFunction p_f_r_plus(new FunctionPlus());
    funcs_chain.push_back(p_f_r_plus);
    PFunction p_f_r_sig(new FunctionSigmoid());
    funcs_chain.push_back(p_f_r_sig);

    PFunction p_f_w_z_linear(new FunctionLinear(w_z));
    funcs_chain.push_back(p_f_w_z_linear);
    PFunction p_f_u_z_linear(new FunctionLinear(u_z, b_z));
    funcs_chain.push_back(p_f_u_z_linear);
    PFunction p_f_z_plus(new FunctionPlus());
    funcs_chain.push_back(p_f_z_plus);
    PFunction p_f_z_sig(new FunctionSigmoid());
    funcs_chain.push_back(p_f_z_sig);

    PFunction p_f_w_g_linear(new FunctionLinear(w_g));
    funcs_chain.push_back(p_f_w_g_linear);
    PFunction p_f_u_g_linear(new FunctionLinear(u_g, b_g));
    funcs_chain.push_back(p_f_u_g_linear);
    PFunction p_f_g_plus(new FunctionPlus());
    funcs_chain.push_back(p_f_g_plus);
    PFunction p_f_g_tanh(new FunctionTanh());
    funcs_chain.push_back(p_f_g_tanh);
    PFunction p_f_g_mul(new FunctionMul());
    funcs_chain.push_back(p_f_g_mul);

    PFunction p_f_minus(new FunctionMinus());
    funcs_chain.push_back(p_f_minus);
    PFunction p_f_mul1(new FunctionMul());
    funcs_chain.push_back(p_f_mul1);
    PFunction p_f_mul2(new FunctionMul());
    funcs_chain.push_back(p_f_mul2);
    PFunction p_f_plus(new FunctionPlus());
    funcs_chain.push_back(p_f_plus);

    //--------------------------------------------


    if (h.get() == NULL || h->data.rows == 0 || h->data.cols != x->data.cols) {
        h = PVariable(variable_construct(output_size, x->data.cols), variable_destroy);

        h->opt = id;

        last_opt = id;

        h->last_opt = &last_opt;
        h->is_last_backward = &is_last_backward;
        id++;

    }
    if (ones.get() == NULL || ones->data.rows == 0 || ones->data.cols != x->data.cols){
        ones = PVariable(new Variable(output_size, x->data.cols, false));
        ones->ones();
    }


    PVariable r = p_f_r_sig->forward(
            p_f_r_plus->forward(
                    p_f_w_r_linear->forward(x), p_f_u_r_linear->forward(h)
            )
    );

    PVariable z = p_f_z_sig->forward(
            p_f_z_plus->forward(
                    p_f_w_z_linear->forward(x), p_f_u_z_linear->forward(h)
            )
    );

    PVariable g = p_f_g_tanh->forward(
            p_f_g_plus->forward(
                    p_f_w_g_linear->forward(x), p_f_u_g_linear->forward(p_f_g_mul->forward(r, h))
            )
    );

    h = p_f_plus->forward(
      p_f_mul1->forward(p_f_minus->forward(ones, z), h), p_f_mul2->forward(z, g)
    );


    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;
    id++;

    return h;
}


void GRU::reset_state(){

    w_r->zero_grad();
    u_r->zero_grad();
    b_r->zero_grad();
    w_z->zero_grad();
    u_z->zero_grad();
    b_z->zero_grad();
    w_g->zero_grad();
    u_g->zero_grad();
    b_g->zero_grad();
    ones->zero_grad();

    h->zeros();
    h->unchain();

    last_opt = 0;
    is_last_backward = false;

    id = 0;

    h->opt = id;

    last_opt = id;

    h->last_opt = &last_opt;
    h->is_last_backward = &is_last_backward;

    id++;

}

void GRU::zero_grads() {
    w_r->zero_grad();
    u_r->zero_grad();
    b_r->zero_grad();
    w_z->zero_grad();
    u_z->zero_grad();
    b_z->zero_grad();
    w_g->zero_grad();
    u_g->zero_grad();
    b_g->zero_grad();
    ones->zero_grad();

    h->zero_grad();
}

void GRU::toHostArray(){
    w_r->data.toHostArray();
    u_r->data.toHostArray();
    b_r->data.toHostArray();
    w_z->data.toHostArray();
    u_z->data.toHostArray();
    b_z->data.toHostArray();
    w_g->data.toHostArray();
    u_g->data.toHostArray();
    b_g->data.toHostArray();

}
void GRU::fromHostArray(){

    w_r->data.fromHostArray();
    u_r->data.fromHostArray();
    b_r->data.fromHostArray();
    w_z->data.fromHostArray();
    u_z->data.fromHostArray();
    b_z->data.fromHostArray();
    w_g->data.fromHostArray();
    u_g->data.fromHostArray();
    b_g->data.fromHostArray();
}



BatchNorm::BatchNorm() : Graph() {
}

BatchNorm::BatchNorm(int element_size, int channel_num, float decay) {
    x_mean = new Variable(element_size, channel_num, false);
    x_var = new Variable(element_size, channel_num, false);

    gamma = new Variable(element_size, channel_num);
    beta = new Variable(element_size, channel_num);

    gamma->ones();

    lambda = decay;

    this->element_size = element_size;
    this->channel_num = channel_num;
}
BatchNorm::~BatchNorm() {
    if (x_mean != NULL) delete x_mean;
    if (x_var != NULL)delete x_var;

    if (gamma != NULL) delete gamma;
    if (beta != NULL) delete beta;
}

vector<Variable *> BatchNorm::getParams(){
    vector<Variable *> params;
    params.push_back(gamma);
    params.push_back(beta);

    return params;
}

PVariable BatchNorm::forward(PVariable x) {

    // prepare function
    FunctionBatchNorm *f = new FunctionBatchNorm(element_size, channel_num, gamma, beta, x_mean, x_var);
    PFunction p_batch_norm(f);
    funcs_chain.push_back(p_batch_norm);


    PVariable x_h;

    f->is_train = is_train;
    x_h = p_batch_norm->forward(x);

    if (is_train) {

        float lam = lambda;
        if (is_first){
            lam = 0.0;
            is_first = false;
        }

        cuMat current_mean(element_size, channel_num);
        cuMat current_var(element_size, channel_num);

        for(int i=0; i<channel_num; i++){
            current_mean.memSetDeviceCol(f->rmu[i].mDevice, i);
            current_var.memSetDeviceCol(f->var[i].mDevice, i);
        }

        x_mean->data = lam * x_mean->data + (1.0-lam) * current_mean;
        x_var->data = lam * x_var->data  + (1.0-lam) * current_var;
    }


    return x_h;
}

void BatchNorm::setTrainStatus(bool status) {
    is_train = status;
}

void BatchNorm::toHostArray() {
    gamma->data.toHostArray();
    beta->data.toHostArray();
}

void BatchNorm::fromHostArray() {
    gamma->data.fromHostArray();
    beta->data.fromHostArray();
}

void BatchNorm::zero_grads() {
    gamma->zero_grad();
    beta->zero_grad();
}



Conv2D::Conv2D() : Graph() {

}

Conv2D::Conv2D(int batch_num, int channel_num, int w_size, int h_size, int filter_size, int filter_num, int stride, int padding) {

    this->batch_num = batch_num;
    this->channel_num = channel_num;
    this->w_size = w_size;
    this->h_size = h_size;
    this->filter_size = filter_size;
    this->filter_num = filter_num;
    this->stride = stride;
    this->padding = padding;


    w = new Variable(filter_num, filter_size * filter_size * channel_num);
    //He-Normal
    //https://arxiv.org/pdf/1502.01852.pdf
    w->randoms(0., sqrt(2.0/((float)filter_size*filter_size * channel_num)));

    b = new Variable(filter_num, 1);
}
Conv2D::~Conv2D() {
    delete w;
    delete b;
}

vector<Variable *> Conv2D::getParams(){
    vector<Variable *> params;
    params.push_back(w);
    params.push_back(b);

    return params;
}


PVariable Conv2D::forward(PVariable x) {

    // prepare function
    FunctionConv2D *f = new FunctionConv2D(w, b, batch_num, channel_num, w_size, h_size, filter_size, filter_num,  stride, padding);
    PFunction p_conv2d(f);
    funcs_chain.push_back(p_conv2d);


    return p_conv2d->forward(x);

}

void Conv2D::zero_grads() {
    w->zero_grad();
    b->zero_grad();
}


void Conv2D::toHostArray() {
    w->data.toHostArray();
    b->data.toHostArray();
}

void Conv2D::fromHostArray() {
    w->data.fromHostArray();
    b->data.fromHostArray();
}


Pooling::Pooling() : Graph() {
}
Pooling::Pooling(int width, int height, int depth, int windowWidth, int windowHeight, int stride, int padding){
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->windowWidth = windowWidth;
    this->windowHeight = windowHeight;
    this->stride = stride;
    this->padding = padding;

}

PVariable Pooling::forward(PVariable x){
    FunctionPooling *f = new FunctionPooling(width, height, depth, windowWidth, windowHeight,  stride, padding);

    PFunction p_pooling(f);
    funcs_chain.push_back(p_pooling);


    return p_pooling->forward(x);
}

