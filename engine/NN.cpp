#include "NN.hpp"
#include "rng.hpp"

Neuron::Neuron(int nin)
{
    for(int i {0} ; i<nin ; i++) weights.push_back(make_shared<node>(random_uniform())) ;
    bias = make_shared<node>(random_uniform()) ;
}
shared_ptr<node> Neuron::operator() (const vector<shared_ptr<node>>& x)
{
    auto act {bias} ;
    for(int i {0} ; i<weights.size() ; i++) act = act + weights[i]*x[i] ;
    return act->tanh() ;
}
vector<shared_ptr<node>> Neuron::parameters() const
{
    vector<shared_ptr<node>> params = weights ;
    params.push_back(bias) ;
    return params ;
}

