/*
 * function_set.h
 *
 *  Created on: 2016/01/06
 *      Author: takeshi.fujita
 */


#ifndef FUNCTION_SET_H_
#define FUNCTION_SET_H_

#include <typeinfo>
#include <vector>
#include <map>

#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "function.h"
#include "variable.h"

using namespace std;

class UpdateParams {
public:
    vector<Variable *> params;


    void add(Variable *v){
        params.push_back(v);
    }
};

class Model {
private:

    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {
        ar & funcs;
    }


public:
    map<string, Function *> funcs;
    vector<UpdateParams *> updateParams;

    ~Model(){
        for (int i=0; i<updateParams.size(); i++){
            delete updateParams.at(i);
        }
    }

    void putF(string name, Function *f){
        funcs[name] = f;
    }
    Function *f(string name){
        return funcs.at(name);
    }

    vector<UpdateParams *> &getUpdateParams(){
        for(auto fs : funcs){
            Function *f = fs.second;

            const type_info& id = typeid(*f);

            if (typeid(FunctionLinear) == id){
                UpdateParams *p = new UpdateParams();
                Variable *w = &((FunctionLinear *)f)->w;
                p->add(w);
                if (!((FunctionLinear *)f)->noBias){
                    Variable *b = &((FunctionLinear *)f)->b;
                    p->add(b);
                }
                updateParams.push_back(p);
            }
            else if (typeid(FunctionEmbed) == id) {
                UpdateParams *p = new UpdateParams();
                Variable *w = &((FunctionEmbed *) f)->w;
                p->add(w);
                if (!((FunctionEmbed *) f)->noBias) {
                    Variable *b = &((FunctionEmbed *) f)->b;
                    p->add(b);
                }
                updateParams.push_back(p);
            }
            else if (typeid(FunctionGRU) == id){
                FunctionGRU *fg =  (FunctionGRU *)f;

                FunctionLinear *u_h = (FunctionLinear *)fg->u_h;
                UpdateParams *p1 = new UpdateParams();
                p1->add(&u_h->w); if (!u_h->noBias) p1->add(&u_h->b);
                updateParams.push_back(p1);

                FunctionLinear *ur_h = (FunctionLinear *)fg->ur_h;
                UpdateParams *p2 = new UpdateParams();
                p2->add(&ur_h->w); if (!ur_h->noBias) p2->add(&ur_h->b);
                updateParams.push_back(p2);

                FunctionLinear *uz_h = (FunctionLinear *)fg->uz_h;
                UpdateParams *p3 = new UpdateParams();
                p3->add(&uz_h->w); if (!uz_h->noBias) p3->add(&uz_h->b);
                updateParams.push_back(p3);

                FunctionLinear *w_x = (FunctionLinear *)fg->w_x;
                UpdateParams *p4 = new UpdateParams();
                p4->add(&w_x->w); if (!w_x->noBias) p4->add(&w_x->b);
                updateParams.push_back(p4);

                FunctionLinear *wr_x = (FunctionLinear *)fg->wr_x;
                UpdateParams *p5 = new UpdateParams();
                p5->add(&wr_x->w); if (!wr_x->noBias) p5->add(&wr_x->b);
                updateParams.push_back(p4);

                FunctionLinear *wz_x = (FunctionLinear *)fg->wz_x;
                UpdateParams *p6 = new UpdateParams();
                p6->add(&wz_x->w); if (!wz_x->noBias) p6->add(&wz_x->b);
                updateParams.push_back(p6);

            }
            else if (typeid(FunctionLSTM) == id){
                FunctionLSTM *fl =  (FunctionLSTM *)f;

                FunctionLinear *x_i = (FunctionLinear *)fl->x_i;
                UpdateParams *p1 = new UpdateParams();
                p1->add(&x_i->w); if (!x_i->noBias) p1->add(&x_i->b);
                updateParams.push_back(p1);
                FunctionLinear *w_i = (FunctionLinear *)fl->w_i;
                UpdateParams *p2 = new UpdateParams();
                p2->add(&w_i->w); if (!w_i->noBias) p2->add(&w_i->b);
                updateParams.push_back(p2);

                FunctionLinear *x_f = (FunctionLinear *)fl->x_f;
                UpdateParams *p3 = new UpdateParams();
                p3->add(&x_f->w); if (!x_f->noBias) p3->add(&x_f->b);
                updateParams.push_back(p3);
                FunctionLinear *w_f = (FunctionLinear *)fl->w_f;
                UpdateParams *p4 = new UpdateParams();
                p4->add(&w_f->w); if (!w_f->noBias) p4->add(&w_f->b);
                updateParams.push_back(p4);

                FunctionLinear *x_o = (FunctionLinear *)fl->x_o;
                UpdateParams *p5 = new UpdateParams();
                p5->add(&x_o->w); if (!x_o->noBias) p5->add(&x_o->b);
                updateParams.push_back(p5);
                FunctionLinear *w_o = (FunctionLinear *)fl->w_o;
                UpdateParams *p6 = new UpdateParams();
                p6->add(&w_o->w); if (!w_o->noBias) p6->add(&w_o->b);
                updateParams.push_back(p6);

                FunctionLinear *x_g = (FunctionLinear *)fl->x_g;
                UpdateParams *p7 = new UpdateParams();
                p7->add(&x_g->w); if (!x_g->noBias) p7->add(&x_g->b);
                updateParams.push_back(p7);
                FunctionLinear *w_g = (FunctionLinear *)fl->w_g;
                UpdateParams *p8 = new UpdateParams();
                p8->add(&w_g->w); if (!w_g->noBias) p8->add(&w_g->b);
                updateParams.push_back(p8);

            }
        }
        return updateParams;
    }


    void save(string path){
        for(auto fs : funcs){
            Function *f = fs.second;

            const type_info& id = typeid(*f);

            if (typeid(FunctionLinear) == id){
                ((FunctionLinear *)f)->toHostArray();
            }
            else if (typeid(FunctionEmbed) == id){
                ((FunctionEmbed *)f)->toHostArray();
            }
            else if (typeid(FunctionGRU) == id){
                FunctionGRU *fg =  (FunctionGRU *)f;
                FunctionLinear *u_h = (FunctionLinear *)fg->u_h;
                FunctionLinear *ur_h = (FunctionLinear *)fg->ur_h;
                FunctionLinear *uz_h = (FunctionLinear *)fg->uz_h;
                FunctionLinear *w_x = (FunctionLinear *)fg->w_x;
                FunctionLinear *wr_x = (FunctionLinear *)fg->wr_x;
                FunctionLinear *wz_x = (FunctionLinear *)fg->wz_x;
                u_h->toHostArray();
                ur_h->toHostArray();
                uz_h->toHostArray();
                w_x->toHostArray();
                wr_x->toHostArray();
                wz_x->toHostArray();
            }
            else if (typeid(FunctionLSTM) == id){
                 FunctionLSTM *fl =  (FunctionLSTM *)f;

                FunctionLinear *x_i = (FunctionLinear *)fl->x_i;
                FunctionLinear *w_i = (FunctionLinear *)fl->w_i;

                FunctionLinear *x_f = (FunctionLinear *)fl->x_f;
                FunctionLinear *w_f = (FunctionLinear *)fl->w_f;

                FunctionLinear *x_o = (FunctionLinear *)fl->x_o;
                FunctionLinear *w_o = (FunctionLinear *)fl->w_o;

                FunctionLinear *x_g = (FunctionLinear *)fl->x_g;
                FunctionLinear *w_g = (FunctionLinear *)fl->w_g;
                x_i->toHostArray();
                w_i->toHostArray();
                x_f->toHostArray();
                w_f->toHostArray();
                x_o->toHostArray();
                w_o->toHostArray();
                x_g->toHostArray();
                w_g->toHostArray();
            }
        }

        std::ofstream ofs(path);
        boost::archive::binary_oarchive oa(ofs);
        oa.register_type<FunctionLinear>(); // add if you define new function
        oa.register_type<FunctionGRU>(); // add if you define new function
        oa.register_type<FunctionLSTM>(); // add if you define new function
        oa.register_type<FunctionEmbed>(); // add if you define new function

        oa << *this;

        ofs.close();
    }
    void load(string path){
        std::ifstream ifs(path);
        boost::archive::binary_iarchive ia(ifs);
        ia.register_type<FunctionLinear>(); // add if you define new function
        ia.register_type<FunctionGRU>(); // add if you define new function
        ia.register_type<FunctionLSTM>(); // add if you define new function
        ia.register_type<FunctionEmbed>(); // add if you define new function

        ia >> *this;

        ifs.close();

        for(auto fs : funcs){
            Function *f = fs.second;

            const type_info& id = typeid(*f);

            if (typeid(FunctionLinear) == id){
                ((FunctionLinear *)f)->fromHostArray();
            }
            else if (typeid(FunctionEmbed) == id){
                ((FunctionEmbed *)f)->fromHostArray();
            }
            else if (typeid(FunctionGRU) == id) {
                FunctionGRU *fg = (FunctionGRU *) f;
                FunctionLinear *u_h = (FunctionLinear *) fg->u_h;
                FunctionLinear *ur_h = (FunctionLinear *) fg->ur_h;
                FunctionLinear *uz_h = (FunctionLinear *) fg->uz_h;
                FunctionLinear *w_x = (FunctionLinear *) fg->w_x;
                FunctionLinear *wr_x = (FunctionLinear *) fg->wr_x;
                FunctionLinear *wz_x = (FunctionLinear *) fg->wz_x;
                u_h->fromHostArray();
                ur_h->fromHostArray();
                uz_h->fromHostArray();
                w_x->fromHostArray();
                wr_x->fromHostArray();
                wz_x->fromHostArray();
            }
            else if (typeid(FunctionLSTM) == id){
                 FunctionLSTM *fl =  (FunctionLSTM *)f;

                FunctionLinear *x_i = (FunctionLinear *)fl->x_i;
                FunctionLinear *w_i = (FunctionLinear *)fl->w_i;

                FunctionLinear *x_f = (FunctionLinear *)fl->x_f;
                FunctionLinear *w_f = (FunctionLinear *)fl->w_f;

                FunctionLinear *x_o = (FunctionLinear *)fl->x_o;
                FunctionLinear *w_o = (FunctionLinear *)fl->w_o;

                FunctionLinear *x_g = (FunctionLinear *)fl->x_g;
                FunctionLinear *w_g = (FunctionLinear *)fl->w_g;
                x_i->fromHostArray();
                w_i->fromHostArray();
                x_f->fromHostArray();
                w_f->fromHostArray();
                x_o->fromHostArray();
                w_o->fromHostArray();
                x_g->fromHostArray();
                w_g->fromHostArray();
            }
        }

        getUpdateParams();
    }

};

#endif /* FUNCTION_SET_H_ */
