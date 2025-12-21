// #include <bits/stdc++.h>
// using namespace std;

// class node
// {
//     double data ; 
//     double grad ;
//     string op ;
//     function<void()> _backward ;
//     set<node> parents
    
//     public : 
//     node(int d, int g = 0, string l = "") : data(d), grad(g), op(l)
//     {
//         _backward = [](){} ;
//         cout << "node constructor" << endl;
//     }
//     ~node() {cout << "node destructor" << endl;}
//     void show() {cout << "(data,grad,op) : (" << data << ", " << grad << ", " << op << ")" << endl;}
//     node operator+(node& other)
//     {
//         node out = node(data+other.data,0,"+") ;
//         _backward = [this,other,out]()
//         {
//             this->grad += 1*out.grad ;
//             other.grad += 1*out.grad ;
//         }
//         return out ;
//     }
//     node operator-(node& other)
//     {
        
//         return node(data-other.data,0,"-") ;
//     }
//     node operator*(node& other)
//     {
//         return node(data*other.data,0,"*") ;
//     }
//     node operator/(node& other)
//     {
//         return node(data/other.data,0,"/") ;
//     }
//     friend ostream& operator<<(ostream& COUT, const node& other) ;
// };

// ostream& operator<<(ostream& COUT, const node& other)
// {
//     COUT << other.data << ", " << other.grad << ", " << other.op << endl;
//     return COUT ;
// }
// int main()
// {   
//     node a(10), b(20) ;
//     node c = a+b ;
//     c.show() ;
//     cout << c << endl;
//     cout << a+b << endl;  
// }