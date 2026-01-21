#ifndef GRADWIZ_HPP
#define GRADWIZ_HPP

#include <bits/stdc++.h>
using namespace std;

class node : public enable_shared_from_this<node>
{
    double data ;
    double grad ;
    string op ;
    string label ;
    function<void()> _backward ;
    vector<shared_ptr<node>> parents ;
    // dfs
    static void build_dfs(shared_ptr<node> v,
    vector<shared_ptr<node>> &dfs, unordered_set<node*>& vis) ;

    public : 
    // constructor
    node(double d, string l = "", string o = "default") ;
    // getters.
    double getdata() const ;
    double getgrad() const ;
    string getlabel() const ;
    string getop() const ;
    void show() ;
    const vector<shared_ptr<node>>& getparents() const ;

    //adders
    void add_to_data(double delta) ;
    void add_to_grad(double delta) ;

    // operations and gradient calculation
    void backward() ;
    shared_ptr<node> tanh() ;
    shared_ptr<node> power(double) ;
    friend shared_ptr<node> operator+(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator*(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator-(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator/(shared_ptr<node> self, shared_ptr<node> other) ;
} ;

void print_tree(const shared_ptr<node>&,bool annot = false, int depth = 0, unordered_set<node*>* visited = nullptr) ;

// wrapper function to return shared_ptr<node>
shared_ptr<node> Value(double v, string label = "") ;
// template wrapper
template<typename T>
vector<shared_ptr<node>> Value(const vector<T> v)
{
    vector<shared_ptr<node>> ret ;
    for(const auto& i : v) ret.push_back(Value(static_cast<double>(i))) ;
}
#endif