#include "opt_loss.hpp"

int main()
{
    vector<int> layerdef {4,4,2} ;
    MLP n {2,layerdef} ;
    auto params = n.parameters() ;
    optimizer opt(0.01) ;
    int epochs {200} ;
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