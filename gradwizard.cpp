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
    string label ;
    node(double d, string l = "", string o = "default") : data(d), grad(0), op(o), _backward([](){}), label(l)
    {
        cout << "node constructor" << endl;
    }
    ~node() {cout << "node destructor" << endl;}

    double getdata() const {return (double)data ;}
    double getgrad() const {return (double)grad ;}
    string getop() const {return op ;}
    string getlabel() const {return label ;}

    void show()
    {
        cout << "data : " << getdata() << " ||  grad : " << getgrad() << " || " ;
        cout << "op : " << getop() << " || label : " << getlabel() << endl;
    }

    friend shared_ptr<node> operator+(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator*(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator-(shared_ptr<node> self, shared_ptr<node> other) ;
    friend shared_ptr<node> operator/(shared_ptr<node> self, shared_ptr<node> other) ;

    shared_ptr<node> power(double n)
    {
        double m {n}, val {1} ;
        while(m--) val*=data ;
        auto out = make_shared<node>(val,"","power") ;
        auto self {shared_from_this()} ;
        out->parents = {self} ;
        out->_backward = [self,out,n]()
        {
            self->grad += n*(pow(self->data,n-1))*out->grad ;
        };
        return out ;
    }
    shared_ptr<node> tanh()
    {
        double x {data} ;
        double t {(exp(2*x)-1)/(exp(2*x)+1)} ;
        auto out = make_shared<node>(t,"","tanh()") ;
        auto self {shared_from_this()} ;
        out->parents = {self} ;
        out->_backward = [self,out,t]()
        {
            self->grad += (1-t*t)*out->grad ;
        };
        return out ;
    }

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
    auto out = make_shared<node>(self->data+other->data,"","+");
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
    auto out = make_shared<node>(self->data*other->data,"","*") ;
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
    auto out = make_shared<node>(self->data-other->data,"","-") ;
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
    auto out = make_shared<node>(self->data/other->data,"","/") ;
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
    auto a {make_shared<node>(2.0,"a")}, b {make_shared<node>(3.0,"b")} ;
    auto c = a*b ;
    c->label = "c" ;
    auto d = make_shared<node>(5.0,"d") ;
    auto e = d/c ;
    e->label = "e" ;
    auto f {make_shared<node>(6.0,"f")} ;
    auto g {make_shared<node>(8.0,"g")} ;
    auto h = e + f*g ;
    h->label = "h" ;
    auto i {make_shared<node>(10.0,"i")} ;
    auto j = h/(i->power(2.0)) ;
    j->label = "j" ;
    auto k = j->tanh() ;
    k->backward() ;
    a->show() ;
    b->show() ;
    c->show() ;
    d->show() ;
    e->show() ;
    f->show() ;
    g->show() ;
    h->show() ;
    i->show() ;
    j->show() ;
    k->show() ;
}