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
    node(double d, string l = "", string o = "default") : data(d), grad(0.0), op(o), _backward([](){}), label(l)
    {
        // cout << "node constructor" << endl;
    }
    ~node() {cout << "node destructor" << endl;}

    double getdata() const {return (double)data ;}
    double getgrad() const {return (double)grad ;}
    string getop() const {return op ;}
    string getlabel() const {return label ;}
    const vector<shared_ptr<node>>& getparents() const {return parents ;}
    void add_to_data(double delta)
    {
        data += delta ;
    }
    void add_to_grad(double delta)
    {
        grad += delta ;
    }
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
        double val {pow(data,n)} ;
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

random_device rd ;
mt19937 generator(rd()) ;
uniform_real_distribution<double> distribution(0.0000001,1.0) ;

class Neuron : public enable_shared_from_this<Neuron>
{
    vector<shared_ptr<node>> weight ;
    shared_ptr<node> bias ;
    bool is_output_neuron ;
    public : 
    Neuron(int nin, bool is_out = false) : is_output_neuron(is_out)
    {
        for(int i {0} ; i<nin ; i++) weight.push_back(make_shared<node>(distribution(generator))) ;
        bias = make_shared<node>(distribution(generator)) ;
    }
    shared_ptr<node> operator() (const vector<shared_ptr<node>>& x)
    {
        auto act {bias} ;   
        for(int i {0} ; i<weight.size() ; i++) act = act + weight[i]*x[i] ;
        return ((is_output_neuron) ? act : act->tanh()) ;
    }
    vector<shared_ptr<node>> parameters() const
    {
        vector<shared_ptr<node>> params = weight ;
        params.push_back(bias) ;
        return params ;
    }  
};
class Layer
{
    vector<Neuron> layer ; // no vector<shared_ptr<Neuron>> because layer owns it's neurons !!
    public : 
    Layer(int nin, int nout, bool is_output_layer = false)
    {
        for(int i {0} ; i<nout ; i++) layer.push_back(Neuron(nin, is_output_layer)) ;
    }
    vector<shared_ptr<node>> operator() (const vector<shared_ptr<node>>& x)
    {
        vector<shared_ptr<node>> output ;
        for(auto& neu : this->layer) output.push_back(neu(x));
        return output ;
    }
    vector<shared_ptr<node>> parameters() const
    {
        vector<shared_ptr<node>> params ;
        for(const auto& i : this->layer)
        {
            for(auto& j : i.parameters()) params.push_back(j) ;
        }
        return params ;
    }
};
class MLP
{
    vector<Layer> mlp ;
    public :
    MLP(int nin, vector<int> nouts)
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
    vector<shared_ptr<node>> operator() (vector<shared_ptr<node>>& x) 
    {
        for(auto& l : mlp) x = l(x) ;
        return x ;
    }
    vector<shared_ptr<node>> parameters() const
    {
        vector<shared_ptr<node>> params ;
        for(auto& layer : mlp) for(auto& par : layer.parameters()) params.push_back(par) ;
        return params ;
    }
};

shared_ptr<node> Value(double v, string label = "")
{
    return make_shared<node>(v,label) ;
}
template<typename T>
vector<shared_ptr<node>> Value(const vector<T> v)
{
    vector<shared_ptr<node>> ret ;
    for(const auto& i : v) ret.push_back(Value(static_cast<double>(i))) ;
}
class optimizer
{
    double learning_rate ;
    public :
    optimizer(double lr = 0.001) : learning_rate(lr) {}
    void step(const vector<shared_ptr<node>>& params)
    {
        for(auto& p : params) p->add_to_data((-learning_rate)*(p->getgrad())) ;
    }
    void zero_grad(const vector<shared_ptr<node>>& params)
    {
        for(auto& p : params) p->add_to_grad(-(p->getgrad())) ;
    }
};

class loss_function
{
    string criterion ;
    function<shared_ptr<node>(vector<shared_ptr<node>>,vector<shared_ptr<node>>)> loss_fn ;
    public : 
    loss_function(string type = "mse") : criterion(type)
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
    shared_ptr<node> operator() (vector<shared_ptr<node>> pred, vector<shared_ptr<node>> target)
    {
        return loss_fn(pred,target) ;
    }
    string get_criterion()
    {
        return criterion ;
    }
};

int main()
{
    freopen("output.txt","w",stdout) ;
    vector<int> layerdef {4,4,2} ;
    MLP n {2,layerdef} ;
    auto params = n.parameters() ;
    optimizer opt(0.01) ;
    int epochs {300} ;
    loss_function func("rmse") ;
    for (int i {0} ; i<epochs ; i++)
    {
        vector<shared_ptr<node>> x {Value(1.0), Value(2.0)} ;
        vector<shared_ptr<node>> y {Value(3.0), Value(6.0)} ;
        auto preds = n(x) ;
        auto total_loss = func(preds,y) ;
        total_loss->backward() ;
        opt.step(params) ;
        opt.zero_grad(params) ;
        if((i+1)%10 == 0)
        {
            cout << "Epoch " << i+1 << "/" << epochs << " : " ;
            cout << "Predictions : " << preds[0]->getdata() << "," << preds[1]->getdata() << endl;
            cout << "Loss = " << total_loss->getdata() << endl;
        }
    }
}