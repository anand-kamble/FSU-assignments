#include <iostream>
#include "brain_mesh.h"

using namespace std;

int main()
{

    // Name of the file to read the mesh
    const char *VTKFileName = "Cort_lobe_poly.vtk";
    // Name of the file to write the surrounding areas
    const char *SurroundingAreaFileName = "Surrounding_areas.txt";
    // Name of the file to write the edge lengths
    const char *EdgeLengthFileName = "edge_lengths.txt";

    // Initialize the brain mesh with the id "First"
    BrainMesh brain_Mesh("First", false);
    
    // Load the mesh from a VTK file
    brain_Mesh.Load(VTKFileName);

    // Calculate the total surface area of the brain mesh
    double totalArea = brain_Mesh.TotalSurfaceArea();
    cout << "Total Area : " << totalArea << endl;

    // Calculate the edge lengths of the brain mesh and write them to a text file
    brain_Mesh.calculateEdgeLengths(EdgeLengthFileName);
    
    // Calculate the surrounding areas of the vertices of brain mesh and write them to a text file
    brain_Mesh.SurroundingAreaForVertex(SurroundingAreaFileName);
    
    return 0;
}