#include <vector>
#include <cmath>

using namespace std;

/**
 * @brief Calculates the area of a triangle in 3D space using the vertices.
 * 
 * This function computes the area of a triangle in three-dimensional space
 * given its vertices. The vertices are represented as vectors with three 
 * elements each (x, y, z coordinates).
 * 
 * The calculation is based on the absolute value of the determinant of a 
 * matrix formed by the coordinates of the three vertices.
 * 
 * @param p1 Vector representing the coordinates of Vertex 1 (x1, y1, z1).
 * @param p2 Vector representing the coordinates of Vertex 2 (x2, y2, z2).
 * @param p3 Vector representing the coordinates of Vertex 3 (x3, y3, z3).
 * 
 * @return double The calculated area of the triangle.
 * 
 * @note The vertices must be given in a counter-clockwise or clockwise order.
 * 
 * @note see https://bearboat.net/TriangleArea/Triangle.html
 */
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

    return sqrt(pow(x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3, 2) +
                pow(x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3, 2) +
                pow(y2 * z1 - y3 * z1 - y1 * z2 + y3 * z2 + y1 * z3 - y2 * z3, 2)) / 2.0;
}
