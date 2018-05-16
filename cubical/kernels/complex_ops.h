#include <complex>

std::complex<double> make_polar(double x,double y)
{
    return std::polar(x,y);
}

std::complex<float> make_polar(float x,float y)
{
    return std::polar(x,y);
}
