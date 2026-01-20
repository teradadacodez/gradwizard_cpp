#include "NN.hpp"
#include "rng.hpp"

Neuron::Neuron(int nin, bool is_out = false) : is_output_neuron(is_out)
{
    for(int i {0} ; i<nin ; i++) weight.push_back(make_shared<node>(random_uniform())) ;
    bias = make_shared<node>(random_uniform()) ;
}
shared_ptr<node> Neuron::operator() (const vector<shared_ptr<node>>& x)
{
    auto act {bias} ;   
    for(int i {0} ; i<weight.size() ; i++) act = act + weight[i]*x[i] ;
    return ((is_output_neuron) ? act : act->tanh()) ;
}
vector<shared_ptr<node>> Neuron::parameters() const
{
    vector<shared_ptr<node>> params = weight ;
    params.push_back(bias) ;
    return params ;
}
Layer::Layer(int nin, int nout, bool is_output_layer = false)
{
    for(int i {0} ; i<nout ; i++) layer.push_back(Neuron(nin,is_output_layer)) ;
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
MLP::MLP(int nin, vector<int> nouts)
{
    vector<int> sz ;
    sz.push_back(nin) ;
    for(auto i : nouts) sz.push_back(i) ;
    for(int i {0} ; i<nouts.size() ; i++) 
    {
        bool is_last_layer = (i==nouts.size()-1) ;
        mlp.push_back(Layer(sz[i],sz[i+1],is_last_layer)) ;
    }
}
vector<shared_ptr<node>> MLP::operator() (vector<shared_ptr<node>> x) 
{
    for(auto& l : mlp) x = l(x) ;
    return x ;
}
vector<shared_ptr<node>> MLP::parameters() const
{
    vector<shared_ptr<node>> params ;
    for(auto& layer : mlp) for(auto& par : layer.parameters()) params.push_back(par) ;
    return params ;
}
