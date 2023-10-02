#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include "triangle_area.cpp"

using namespace std;

/**
 * @brief The class which includes all the functions required for reading and processing the given VTK file.
 *
 */
class VTK_Handler
{
private:
    /**
     * @brief The variable which holds number of vertices.
     *
     */
    int numberOfPoints = 0;
    /**
     * @brief The variable which holds number of polygons in given VTK file.
     *
     */
    int numberOfPolygnons = 0;
    /**
     * @brief If Debug, the class will print more information which will help for debugging.
     *
     */
    bool Debug = false;
    /**
     * @brief This vector holds all of the vertices in the format [X,Y,Z].
     *
     */
    vector<vector<float>> points;
    /**
     * @brief This vector holds all of the polygons where each polygon includes the indices of the vertices which make up that polygon.
     *
     */
    vector<vector<int>> polygons;

    /**
     * @brief The Function which splits the provided string into parts seperated by the given seperator.

     * @param toSplit The string to split.
     * @param seperator Seperator that will determine how to split the string.
     * @return vector<string>
     */
    vector<string> split(const string &toSplit, char seperator = ' ')
    {
        vector<string> parts;
        string s;
        istringstream stream(toSplit);
        if (Debug)
            cout << "String to split : " << toSplit << endl;
        while (getline(stream, s, seperator))
        {
            if (Debug)
                cout << "- " << s << endl;
            parts.push_back(s);
        }
        return parts;
    }

    /**
     * @brief Common error handler for all the functions inside this class.
     *
     * @param errorMsg The messege to be printed to console.
     */
    void error(string errorMsg)
    {
        cout.rdbuf(NULL);
        cout << "[ERROR] : " << errorMsg << endl;
        delete this;
        exit(EXIT_SUCCESS);
    }
    /**
     * @brief This calculates the length of the edge given two vertices.
     *
     * @param p1 Vertex 1
     * @param p2 Vertex 2
     * @return double
     */
    double edgeLength(vector<float> p1, vector<float> p2)
    {
        auto x1 = p1[0];
        auto x2 = p2[0];

        auto y1 = p1[1];
        auto y2 = p2[1];

        auto z1 = p1[2];
        auto z2 = p2[2];

        /* https://www.varsitytutors.com/hotmath/hotmath_help/topics/distance-formula-in-3d */
        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
    }

public:
    /**
     * @brief This will load the file into memory with dynamic arrays.
     * The points and polygon are stored in their own dynamic  arrays which makes
     * accesing the data easier.
     *
     * @param filename
     */
    void Load(const char *filename)
    {
        string file(filename);
        string vtkString;
        ifstream VTKFileStream;
        VTKFileStream.open(filename);

        if (VTKFileStream.is_open())
        {
            /* Skipping the first 4 lines of the VTK file. */
            for (size_t i = 0; i < 4; i++)
            {
                getline(VTKFileStream, vtkString);
            }

            getline(VTKFileStream, vtkString);
            auto parts = split(vtkString);
            if (vtkString.rfind("POINTS", 0) == 0)
            {
                numberOfPoints = stoi(parts[1]);
            }

            for (size_t i = 0; i < numberOfPoints; i++)
            {
                getline(VTKFileStream, vtkString);
                parts = split(vtkString);
                vector<float> point;
                for (size_t i = 0; i < 3; i++)
                {
                    point.push_back(stof(parts[i]));
                }
                points.push_back(point);
            }

            getline(VTKFileStream, vtkString);
            if (vtkString.rfind("POLYGONS", 0) == 0)
            {
                parts = split(vtkString);
                numberOfPolygnons = stoi(parts[1]);
            }

            for (size_t i = 0; i < numberOfPolygnons; i++)
            {
                getline(VTKFileStream, vtkString);
                parts = split(vtkString);
                vector<int> polygon;
                for (size_t i = 1; i <= 3; i++)
                {
                    polygon.push_back(stoi(parts[i]));
                }
                polygons.push_back(polygon);
            }

            VTKFileStream.close();
            cout << "File Loaded - " << file << endl;
            cout << "Found " << points.size() << " points. " << endl;
            cout << "Found " << polygons.size() << " polygons. " << endl;
        }
        else
        {

            error("Failed to open the file - " + file);
        }
    }

    /**
     * @brief The function which calculates the total surface area of all the polygons.
     *
     * @return double
     */
    double TotalSurfaceArea()
    {
        try
        {
            double area = 0.0;
            for (auto polygon : polygons)
            {
                // area += areaOfTriangle(polygon);
                area += areaOfTriangle(points[polygon[0]], points[polygon[1]], points[polygon[2]]);
            }
            return area;
        }
        catch (const exception &e)
        {
            error("Failed to calculate total area of mesh.");
            return -1.0;
        }
    }

    /**
     * @brief Calculting the total surface area of all the polygons surrounding a vertex.
     *
     * @param filename Name of the output file which will include areas sorted by vertex.
     */
    void SurroundingAreaForVertex(const char *filename)
    {

        printf("\n");
        printf("Calculating area of surrounding polygons : [ %d / %d]\n", 0, numberOfPoints);
        ofstream outfile(filename);
        cout.rdbuf(outfile.rdbuf());
        double surfaceArea;
        for (int i = 0; i < numberOfPoints; i++)
        {
            surfaceArea = 0.0;
            for (auto polygon : polygons)
            {
                if (i == polygon[0] || i == polygon[1] || i == polygon[2])
                {
                    surfaceArea += areaOfTriangle(points[polygon[0]], points[polygon[1]], points[polygon[2]]);
                }
            }
            cout << surfaceArea << endl;
            printf("\x1b[A");
            printf("Calculating area of surrounding polygons : [ %d / %d]\n", i, numberOfPoints);
        }
        outfile.close();
        cout.rdbuf(NULL);
    }

    /**
     * @brief Calculating the length of edges in each polygon and saving those to a file.
     *
     * @param filename Name of the output file which will include edge lenghts.
     */
    void calculateEdgeLengths(const char *filename)
    {
        ofstream outfile(filename);
        cout.rdbuf(outfile.rdbuf());
        printf("\n");
        int i = 1;
        printf("Calculating edge lenghts : [ %d / %d]\n", i, numberOfPolygnons);
        for (auto polygon : polygons)
        {
            cout << edgeLength(points[polygon[0]], points[polygon[1]]) << endl;
            cout << edgeLength(points[polygon[1]], points[polygon[2]]) << endl;
            cout << edgeLength(points[polygon[0]], points[polygon[2]]) << endl;
            printf("\x1b[A");
            printf("Calculating edge lenghts : [ %d / %d]\n", i, numberOfPolygnons);
            i++;
        }
        outfile.close();
        cout.rdbuf(NULL);
    }
};

int main()
{

    auto BrainStructure = new VTK_Handler();

    (*BrainStructure).Load("Cort_lobe_poly.vtk");
    cout << "Total Area : " << (*BrainStructure).TotalSurfaceArea() << endl;

    (*BrainStructure).SurroundingAreaForVertex("vertex_areas.txt");

    (*BrainStructure).calculateEdgeLengths("edge_lenghts.txt");

    return 0;
}