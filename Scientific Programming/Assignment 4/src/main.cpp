#include <iostream>
#include "VTK_Handler.cpp"

using namespace std;

int main()
{
    /**
     * @brief File names which will be used to read and write the inputs and outputs.
     * 
     */
    const char *VTKFileName = "Cort_lobe_poly.vtk";
    const char *SurroundingAreaFileName = "Surrounding_areas.txt";
    const char *EdgeLengthFileName = "edge_lengths.txt";

    /*Creating a new instance of VTK_Handler.*/
    auto BrainStructure = new VTK_Handler();
    cout << "\n\n                         BRAIN STRUCTURE                         " << endl;

    /**
     * @brief Loading the vtk file.
     *
     */
    BrainStructure->Load(VTKFileName);

    /**
     * @brief Calculating the total surface area.
     *
     */
    double totalSurfaceArea = BrainStructure->TotalSurfaceArea();

    // Print the results.
    cout << "Total Area : " << totalSurfaceArea << endl;
    cout << "-----------------------------------------------------------------" << endl;

    /**
     * @brief Calculate the length of all the edges.
     *
     */
    BrainStructure->calculateEdgeLengths(EdgeLengthFileName);

    /**
     * @brief
     *
     */
    BrainStructure->SurroundingAreaForVertex(SurroundingAreaFileName);

    delete BrainStructure;

    return 0;
}
