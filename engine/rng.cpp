#include <random>
using namespace std;
static mt19937 generator(random_device{}());
static uniform_real_distribution<double> distribution(0.0000001, 1.0);

double random_uniform()
{
    return distribution(generator);
}
