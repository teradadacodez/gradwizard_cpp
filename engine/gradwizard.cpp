#include "gradwizard.hpp"

node::node(double d, string l, string o) : data(d), grad(0.0), op(0), _backward([](){}), label(l) {}

double node::getdata() const {return data ;}
double node::getgrad() const {return grad ;}
string node::getlabel() const {return label ;}
string node::getop() const {return op ;}
void node::show()
{
    cout << "data : " << getdata() << " ||  grad : " << getgrad() << " || " ;
    cout << "op : " << getop() << " || label : " << getlabel() << endl;
}

const vector<shared_ptr<node>>& node::getparents() const {return parents ;}

shared_ptr<node> node::power(double n)
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
shared_ptr<node> node::tanh()
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

void node::backward()
{
    vector<shared_ptr<node>> dfs ;
    unordered_set<node*> vis ;
    build_dfs(shared_from_this(),dfs,vis) ;
    for(auto& n : dfs) n->grad = 0.0 ;
    grad = 1 ;
    for (auto it = dfs.rbegin() ; it!=dfs.rend() ; it++) (*it)->_backward() ;
}
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
void print_tree(
    const shared_ptr<node>& v,
    bool annot = false,
    int depth = 0,
    unordered_set<node*>* visited = nullptr
)
{
    bool root_call = false;
    if (!visited)
    {
        visited = new unordered_set<node*>();
        root_call = true;
    }

    // indentation
    for (int i = 0; i < depth; ++i)
        cout << '\t';

    // display name preference: label > op > ?
    string name;
    if (!v->getlabel().empty())
        name = v->getlabel();
    else if (!v->getop().empty())
        name = v->getop();
    else
        name = "?";

    // print node (annotated or not)
    cout << name;

    if (annot)
    {
        cout << " [op:" << v->getop()
             << ", data=" << v->getdata()
             << ", grad=" << v->getgrad()
             << "]";
    }

    cout << endl;

    // avoid re-printing of shared subgraphs
    if (visited->count(v.get()))
        return;

    visited->insert(v.get());

    // again recurse on it's parents
    for (const auto& p : v->getparents())
        print_tree(p, annot, depth + 1,  visited);

    if (root_call)
        delete visited;
}