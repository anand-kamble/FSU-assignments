#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../src/VTK_Handler.cpp"

using namespace std;

bool testTXTFile(const char *fileName, vector<string> contains)
{
    ifstream file(fileName);

    if (!file.is_open())
    {
        cerr << "Error opening the file during test.\n";
        return 1;
    }

    string line;
    int index = 0;
    while (getline(file, line))
    {
        if (contains[index] != line && line != "")
        {
            file.close();
            cout << "Failed for : " << line << endl;
            return false;
        }
        index++;
    }

    file.close();

    return true;
}

int main()
{
    const char *VTKFileName = "../Test.vtk";
    const char *SurroundingAreaFileName = "Surrounding_areas.txt";
    const char *EdgeLengthFileName = "edge_lengths.txt";

    cout << "Running the Tests." << endl;
    auto TestClass = new VTK_Handler();

    /**
     * @brief Loading the test vtk file.
     *
     */
    TestClass->Load(VTKFileName);

    /**
     * @brief Calculating the total surface area.
     *
     */
    double totalSurfaceArea = TestClass->TotalSurfaceArea();

    /**
     * @brief
     *
     */
    TestClass->SurroundingAreaForVertex(SurroundingAreaFileName);

    /**
     * @brief Calculate the length of all the edges.
     *
     */
    TestClass->calculateEdgeLengths(EdgeLengthFileName);


    cout << "\n\n=== RUNNING TESTS ===\n";

    cout << "Testing the 'TotalSurfaceArea' function. \n Result : ";

    const double exactSurfaceArea = 3.646264;

    // Testing in this way since two values that should be equal may not be due to arithmetic rounding errors
    if (totalSurfaceArea - exactSurfaceArea < 0.0001)
    {
        cout << "Passed." << endl;
    }
    else
    {
        cout << "Failed." << endl;
        cout << "Received : " << totalSurfaceArea << endl;
        cout << "Expected : " << exactSurfaceArea << endl;
    }

    vector<string> exactSurroundingAreas;
    exactSurroundingAreas.push_back("1.91421");
    exactSurroundingAreas.push_back("2.43916");
    exactSurroundingAreas.push_back("2.07313");
    exactSurroundingAreas.push_back("1.36603");
    exactSurroundingAreas.push_back("3.14626");

    cout << "Testing the 'SurroundingAreaForVertex' function. \n Result : ";
    if (testTXTFile(SurroundingAreaFileName, exactSurroundingAreas))
    {
        cout << "Passed." << endl;
    }
    else
    {
        cout << "Failed." << endl;
    }

    vector<string> exactEdgeLengths;
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1.73205");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1");
    exactEdgeLengths.push_back("1.73205");
    exactEdgeLengths.push_back("1.41421");
    exactEdgeLengths.push_back("1");

    cout << "Testing the 'calculateEdgeLengths' function. \n Result : ";
    if (testTXTFile(EdgeLengthFileName, exactEdgeLengths))
    {
        cout << "Passed." << endl;
    }
    else
    {
        cout << "Failed." << endl;
    }

    return 0;
}