#ifndef NN_HPP
#define NN_HPP
#include "gradwizard.hpp"

class Neuron : public enable_shared_from_this<Neuron>
{
    vector<shared_ptr<node>> weights ;
    shared_ptr<node> bias ;

    public : 
    Neuron(int nin) ;
    shared_ptr<node> operator() (const vector<shared_ptr<node>>& x) ;
    vector<shared_ptr<node>> parameters() const ; 
};

#endif