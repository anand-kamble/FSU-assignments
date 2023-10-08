#include <iostream>
#include "brain_mesh.h"

using namespace std;

int main()
{

    const char *VTKFileName = "Test.vtk";
    const char *SurroundingAreaFileName = "Surrounding_areas.txt";
    const char *EdgeLengthFileName = "edge_lengths.txt";


    BrainMesh brain_Mesh("First", false);
    brain_Mesh.Load(VTKFileName);
    cout << "Total Area : " << brain_Mesh.TotalSurfaceArea() << endl;
    brain_Mesh.calculateEdgeLengths(EdgeLengthFileName);
    brain_Mesh.SurroundingAreaForVertex(SurroundingAreaFileName);
    return 0;
}