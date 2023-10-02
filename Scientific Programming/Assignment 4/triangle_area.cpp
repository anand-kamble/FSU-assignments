#include <vector>
#include <cmath>


using namespace std;

double areaOfTriangle(vector<float> p1, vector<float> p2, vector<float> p3)
{

    auto x1 = p1[0];
    auto x2 = p2[0];
    auto x3 = p3[0];

    auto y1 = p1[1];
    auto y2 = p2[1];
    auto y3 = p3[1];

    auto z1 = p1[2];
    auto z2 = p2[2];
    auto z3 = p3[2];

    /* https://bearboat.net/TriangleArea/Triangle.html */
    return sqrt(pow(x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3, 2) + pow(x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3, 2) + pow(y2 * z1 - y3 * z1 - y1 * z2 + y3 * z2 + y1 * z3 - y2 * z3, 2)) / 2.0;
}
