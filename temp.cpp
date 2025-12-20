#include <bits/stdc++.h>
using namespace std;

class node : public enable_shared_from_this<node>
{
    double data ; 
    double grad ;
    string op ;
    function<void()> _backward ;
    vector<shared_ptr<node>> parents ;

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
        grad = 1.0 ;
        _backward() ;
    }
    
};

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
        self->grad += (1/other->data)*out->grad ;
        other->grad += (-self->data/(other->data*other->data))*out->grad ;
    };
    return out ;
}
int main()
{
    auto a {make_shared<node>(2.0)}, b {make_shared<node>(3.0)} ;
    auto c = a/b ;
    c->backward() ;
    cout << a->getgrad() << " " << b->getgrad() << endl;
    cout << -2.0/9.0 << endl;
}