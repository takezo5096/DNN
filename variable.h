#include <list>
#include <random>
#include <memory>
#include <boost/intrusive_ptr.hpp>
//#include <boost/pool/object_pool.hpp>

#include "cuMat.h"
#include "cuMatSparse.h"

using namespace std;

#ifndef _VARIABLE_
#define _VARIABLE_


//#include "function.h"
class Function;

//class Variable;
//extern boost::object_pool<Variable> obj_pool;

class Variable {

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {
        ar & id;
        ar & data;
        ar & grad;
        ar & seed;
        ar & isGetGrad;
    }

public:
    //using PVariable = shared_ptr<Variable>;

    int id = -1;

    Function *creator = NULL;

    vector<Function *> functions_history;

    cuMat data;
    cuMatSparse data_sparse;
    cuMat grad;
    cuMat seed;

    bool isGetGrad = true;

    bool isSparse = false;

    //int refc = 0;

    Variable();

    Variable(const Variable &a);

    Variable(int rows, int cols);

    Variable(int rows, int cols, bool is_get_grad);

    Variable(Function *f, int rows, int cols);

    Variable(cuMat &input);

    Variable(Function *f, cuMat &input);

    Variable(vector<float> &ids, int nums);

    ~Variable();

    void creatorSet(Function *f);

    Variable &operator=(const Variable &a);

    Variable sin();
    Variable log();

    void backward();
    void backward(Variable *v);

    void zero_grads();
    void zero_grads(Variable *v);

    void unchain();
    void unchain(Variable *v);

    void ones();
    void zeros();
    void randoms(float m, float a);

    float val();


/*
    friend void
    intrusive_ptr_add_ref(Variable *p_obj)
    {
            p_obj->refc++;
    }

    friend void
    intrusive_ptr_release(Variable *p_obj)
    {
            p_obj->refc--;
            if (p_obj->refc <= 0) {
                //delete p_obj;
                obj_pool.destroy(p_obj);
            }
    }
*/
};


using PVariable = shared_ptr<Variable>;
//using PVariable = boost::intrusive_ptr<Variable>;



/*
Variable operator+(const Variable &v1, const Variable &v2);
Variable operator-(const Variable &v1, const Variable &v2);

Variable operator*(const Variable &v1, const Variable &v2);
*/

#endif
