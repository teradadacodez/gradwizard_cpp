#include "opt_loss.hpp"

optimizer::optimizer(double lr = 0.001) : learning_rate(lr) {}
void optimizer::step(const vector<shared_ptr<node>>& params)
{
    for(auto& p : params) p->add_to_data((-learning_rate)*(p->getgrad())) ;
}
void optimizer::zero_grad(const vector<shared_ptr<node>>& params)
{
    for(auto& p : params) p->add_to_grad(-(p->getgrad())) ;
}
loss_function::loss_function(string type = "mse") : criterion(type)
{
    if(type == "mse")
    {
        loss_fn = [](vector<shared_ptr<node>> pred, vector<shared_ptr<node>> target)
        {
            auto loss = Value(0.0) ;
            for(int i {0} ; i<pred.size() ; i++) loss = loss + (pred[i]-target[i])->power(2) ;
            return loss/Value((double)pred.size()) ;
        };
    }
    else if(type == "rmse")
    {
        loss_fn = [](vector<shared_ptr<node>> pred, vector<shared_ptr<node>> target)
        {
            auto loss = Value(0.0) ;
            for(int i {0} ; i<pred.size() ; i++) loss = loss + (pred[i]-target[i])->power(2) ;
            auto mse =  loss/Value((double)pred.size()) ;
            return mse->power(0.5) ;
        };
    }
    else // "mae"
    {
        loss_fn = [](vector<shared_ptr<node>> pred, vector<shared_ptr<node>> target)
        {
            auto loss = Value(0.0) ;
            for(int i {0} ; i<pred.size() ; i++) loss = loss + ((pred[i]-target[i])->power(2))->power(0.5) ;
            return loss/Value((double)pred.size()) ;
        };
    }
}
shared_ptr<node> loss_function::operator() (vector<shared_ptr<node>> pred, vector<shared_ptr<node>> target)
{
    return loss_fn(pred,target) ;
}
string loss_function::get_criterion()
{
    return criterion ;
}
vector<string> loss_function::list_available_lf()
{
    return {"mse","rmse","mae"} ;
}