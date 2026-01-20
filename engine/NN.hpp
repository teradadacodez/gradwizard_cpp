#ifndef NN_HPP
#define NN_HPP
#include "gradwizard.hpp"

class Neuron
{
    vector<shared_ptr<node>> weight ;
    shared_ptr<node> bias ;
    bool is_output_neuron ;

    public : 
    Neuron(int nin, bool is_out) ;
    shared_ptr<node> operator() (const vector<shared_ptr<node>>& x) ;
    vector<shared_ptr<node>> parameters() const ; 
};
class Layer
{
    vector<Neuron> layer ; // no vector<shared_ptr<Neuron>> because layer owns it's neurons !!
    public : 
    Layer(int nin, int nout, bool is_output_layer);
    vector<shared_ptr<node>> operator() (const vector<shared_ptr<node>>& x);
    vector<shared_ptr<node>> parameters() const ;
};
class MLP
{
    vector<Layer> mlp ;
    public :
    MLP(int nin, vector<int> nouts) ;
    vector<shared_ptr<node>> operator() (vector<shared_ptr<node>> x) ;
    vector<shared_ptr<node>> parameters() const ;
};

#endif