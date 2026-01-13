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
Layer::Layer(int nin, int nout)
{
    for(int i {0} ; i<nout ; i++) layer.push_back(Neuron(nin)) ;
}
vector<shared_ptr<node>> Layer::operator() (const vector<shared_ptr<node>>& x)
{
    vector<shared_ptr<node>> output ;
    for(auto& neu : this->layer) output.push_back(neu(x));
    return output ;
}
vector<shared_ptr<node>> Layer::parameters() const
{
    vector<shared_ptr<node>> params ;
    for(const auto& i : this->layer)
    {
        for(auto& j : i.parameters()) params.push_back(j) ;
    }
    return params ;
}

