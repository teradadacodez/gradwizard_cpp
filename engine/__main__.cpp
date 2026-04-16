#include "opt_loss.hpp"

int main()
{
    vector<int> layerdef {5,5,1} ;
    MLP net {1,layerdef} ;
    auto params = net.parameters() ;
    optimizer opt(0.0005) ;
    loss_function fn("rmse") ;

    int epochs {1000} ;
    for(int epoch {0} ; epoch<epochs ; epoch++)
    {
        double epoch_loss = 0.0 ;
        vector<shared_ptr<node>> x ;
        vector<shared_ptr<node>> y ;
        for(int i {1} ; i<=10 ; i++)
        {
            vector<shared_ptr<node>> x {Value((double)i)} ;
            vector<shared_ptr<node>> y {Value((double)(i*i+1))} ;

            auto preds = net(x) ;
            auto loss = fn(preds,y) ;
            loss->backward() ;
            opt.step(params) ;
            opt.zero_grad(params) ;
            epoch_loss += loss->getdata() ;
        }
        if((epoch+1)%10==0)
        {
            cout << "Total loss after epoch " << epoch+1 << " is " << epoch_loss << endl;
        }
    }
    cout << "Testing..." << endl;

    for(int i {6} ; i<=15 ; i++)
    {
        vector<shared_ptr<node>> x {Value((double)i)} ;
        auto pred = net(x);
        cout << "input " << i 
            << ", output " << pred[0]->getdata() << endl;
    }
}