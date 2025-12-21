#include <bits/stdc++.h>
using namespace std;

class node : public enable_shared_from_this<node>
{
    double data ; 
    double grad ;
    string op ;
    function<void()> _backward ;
    vector<shared_ptr<node>> parents ;

    static void build_dfs(shared_ptr<node> v, vector<shared_ptr<node>>& dfs, unordered_set<node*>& vis) ;

    public : 
    node(double d, string o = "") : data(d), grad(0), op(o), _backward([](){}) {}
    double getdata() const {return (double)data ;}
    double getgrad() const {return (double)grad ;}

    friend shared_ptr<node> operator+(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator*(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator-(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator/(shared_ptr<node> self, shared_ptr<node> other) ;

    void backward()
    {
        vector<shared_ptr<node>> dfs ;
        unordered_set<node*> vis ;
        build_dfs(shared_from_this(),dfs,vis) ;
        for(auto& n : dfs) n->grad = 0.0 ;
        grad = 1 ;
        for (auto it = dfs.rbegin() ; it!=dfs.rend() ; it++) (*it)->_backward() ;
    }
    
};

void node::build_dfs(shared_ptr<node> v, vector<shared_ptr<node>>& dfs, unordered_set<node*>& vis)
{
    if(vis.count(v.get())) return ;
    vis.insert(v.get()) ;
    for (auto& p : v->parents) build_dfs(p,dfs,vis) ;
    dfs.push_back(v) ;
}

shared_ptr<node> operator+(shared_ptr<node> self, shared_ptr<node> other)
{
    auto out = make_shared<node>(self->data+other->data,"+");
    out->parents = {self,other} ;
    out->_backward = [self,other,out]()
    {
        self->grad += out->grad ;
        other->grad += out->grad ;
    };
    return out ;
}
shared_ptr<node> operator*(shared_ptr<node> self, shared_ptr<node> other)
{
    auto out = make_shared<node>(self->data*other->data,"*") ;
    out->parents = {self,other} ;
    out->_backward = [self,other,out]()
    {
        self->grad += other->data*out->grad ;
        other->grad += self->data*out->grad ;
    };
    return out ;
}
shared_ptr<node> operator-(shared_ptr<node> self, shared_ptr<node> other)
{
    auto out = make_shared<node>(self->data-other->data,"-") ;
    out->parents = {self,other} ;
    out->_backward = [self,other,out]()
    {
        self->grad += out->grad ;
        other->grad += -out->grad ;
    };
    return out ;
}
shared_ptr<node> operator/(shared_ptr<node> self, shared_ptr<node> other)
{
    auto out = make_shared<node>(self->data/other->data,"/") ;
    out->parents = {self,other} ;
    out->_backward = [self,other,out]()
    {
        self->grad += (1.0/other->data)*out->grad ;
        other->grad += (-self->data/(other->data*other->data))*out->grad ;
    };
    return out ;
}
int main()
{
    auto a {make_shared<node>(2.0)}, b {make_shared<node>(3.0)} ;
    auto c = a*b ;
    auto d = make_shared<node>(5.0) ;
    auto e = d/c ;
    auto f {make_shared<node>(6.0)} ;
    auto g {make_shared<node>(8.0)} ;
    auto h = e + f*g ;
    h->backward() ;
    cout << a->getgrad() << " " << b->getgrad() << " " << c->getgrad() << " " << d->getgrad() << " " ;
    cout << e->getgrad() << " " << f->getgrad() << " " << g->getgrad() << endl;
}