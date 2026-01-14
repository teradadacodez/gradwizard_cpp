#include "NN.hpp"

int main()
{
    shared_ptr<node> b {make_shared<node>(2.0)} ;
    shared_ptr<node> c {make_shared<node>(3.0)} ;
    shared_ptr<node> d {make_shared<node>(-1.0)} ;
    vector<shared_ptr<node>> x {b,c,d} ;
    vector<int> layerdef {3,3,1} ;
    MLP n {3,layerdef} ;
    print_tree(n(x)[0],true) ;
}