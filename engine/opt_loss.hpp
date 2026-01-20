#ifndef OPTLOSS_HPP
#define OPTLOSS_HPP

#include "NN.hpp"

class optimizer
{
    double learning_rate ;
    public :
    optimizer(double lr) ;
    void step(const vector<shared_ptr<node>>& params) ;
    void zero_grad(const vector<shared_ptr<node>>& params) ;
};

class loss_function
{
    string criterion ;
    function<shared_ptr<node>(vector<shared_ptr<node>>,vector<shared_ptr<node>>)> loss_fn ;
    public : 
    loss_function(string type) ;
    shared_ptr<node> operator() (vector<shared_ptr<node>> pred, vector<shared_ptr<node>> target) ;
    string get_criterion() ;
    vector<string> list_available_lf() ;
};

#endif