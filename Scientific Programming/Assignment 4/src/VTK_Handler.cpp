#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include "triangle_area.cpp"

using namespace std;

/**
 * @brief A class for reading and processing VTK files, providing methods to load, analyze, and compute properties of 3D mesh data.
 */
class VTK_Handler
{
private:
    /**
     * @brief Number of vertices in the VTK file.
     */
    int numberOfPoints = 0;

    /**
     * @brief Number of polygons in the VTK file.
     */
    int numberOfPolygons = 0;

    /**
     * @brief Debug flag for printing additional information during execution.
     */
    bool Debug = false;

    /**
     * @brief Vector containing all vertices in the format [X, Y, Z].
     */
    vector<vector<float>> points;

    /**
     * @brief Vector containing polygons, where each polygon includes indices of vertices that make up the polygon.
     */
    vector<vector<int>> polygons;

    /**
     * @brief The original buffer needed to reset the cout stream. Since we are using the same function to write to files.
     *
     */
    streambuf *originalCoutBuffer = cout.rdbuf();

    /**
     * @brief Splits the provided string into parts separated by the given separator.
     *
     * @param toSplit The string to split.
     * @param separator Separator that determines how to split the string.
     * @return vector<string>
     */
    vector<string> split(const string &toSplit, char separator = ' ')
    {
        vector<string> parts;
        string s;
        istringstream stream(toSplit);
        if (Debug)
            cout << "String to split: " << toSplit << endl;
        while (getline(stream, s, separator))
        {
            if (Debug)
                cout << "- " << s << endl;
            parts.push_back(s);
        }
        return parts;
    }

    /**
     * @brief Common error handler for all functions inside this class.
     *
     * @param errorMsg The message to be printed to the console.
     */
    void error(string errorMsg)
    {
        cout.rdbuf(originalCoutBuffer);
        cout << "[ERROR] : " << errorMsg << endl;
        delete this;
        exit(EXIT_SUCCESS);
    }

    /**
     * @brief Calculates the length of an edge given two vertices.
     *
     * @param p1 Vertex 1.
     * @param p2 Vertex 2.
     * @return double
     */
    double edgeLength(vector<float> p1, vector<float> p2)
    {
        try
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
        catch (...)
        {
            error("Failed to calculate the edge lenght.");
            return -1;
        }
    }

public:
    /**
     * @brief Set the Debug Mode.
     * 
     * @param mode 
     */
    void setDebug(bool mode)
    {
        Debug = mode;
    }
    /**
     * @brief Loads the VTK file into memory with dynamic arrays for points and polygons.
     *
     * @param filename The name of the VTK file to load.
     */
    void Load(const char *filename)
    {
        try
        {
            string file(filename);
            string vtkString;
            ifstream VTKFileStream;
            VTKFileStream.open(filename);
            if (Debug)
                printf("Opening File : %s\n", filename);

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

                for (int i = 0; i < numberOfPoints; i++)
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
                    numberOfPolygons = stoi(parts[1]);
                }

                for (int i = 0; i < numberOfPolygons; i++)
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
                if (Debug)
                    printf("Closing File : %s\n", filename);
                cout << "-----------------------------------------------------------------" << endl;
                cout << "File Loaded - " << file << endl;
                cout << "Found " << points.size() << " points. " << endl;
                cout << "Found " << polygons.size() << " polygons. " << endl;
                cout << "-----------------------------------------------------------------" << endl;
            }
            else
            {
                error("Failed to open the file - " + file);
            }
        }
        catch (const std::exception &e)
        {
            error(e.what());
        }
        catch (...)
        {
            error("Unknown exception caught.");
        }
    }

    /**
     * @brief Calculates the total surface area of all polygons in the mesh.
     *
     * @return double, if failed will return -1.0.
     */
    double TotalSurfaceArea()
    {
        try
        {
            double area = 0.0;
            for (auto polygon : polygons)
            {
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
         * @brief Calculates the total surface area of all polygons surrounding each vertex and writes the result to a file.
         *
         * @param filename Name of the output file.
         */
        void SurroundingAreaForVertex(const char *filename)
        {
            try
            {
                
                cout << "Area of surrounding polygons will be saved in file : " << filename << endl;
                printf("Calculating area of surrounding polygons : [ %d / %d]\n", 0, numberOfPoints);
                ofstream outfile(filename);
                if (Debug)
                    printf("Opening File : %s\n", filename);
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
                    printf("Calculating area of surrounding polygons : [ %d / %d]\n", i+1, numberOfPoints);
                }
                outfile.close();
                if (Debug)
                    printf("Closing File : %s\n", filename);
                cout.rdbuf(originalCoutBuffer);
                cout << "-----------------------------------------------------------------" << endl;
            }
            catch (...)
            {
                error("Failed to calculate surrounding areas at vertex.\n");
            }
        }

    /**
     * @brief Calculates the length of edges in each polygon and saves the results to a file.
     *
     * @param filename Name of the output file.
     */
    void calculateEdgeLengths(const char *filename)
    {
        try
        {
            
            cout << "Edge lengths will be saved in file : " << filename << endl;
            ofstream outfile(filename);
            if (Debug)
                printf("Opening File : %s\n", filename);
            cout.rdbuf(outfile.rdbuf());
            int i = 1;
            printf("Calculating edge lengths : [ %d / %d]\n", i, numberOfPolygons);
            for (auto polygon : polygons)
            {
                cout << edgeLength(points[polygon[0]], points[polygon[1]]) << endl;
                cout << edgeLength(points[polygon[1]], points[polygon[2]]) << endl;
                cout << edgeLength(points[polygon[0]], points[polygon[2]]) << endl;
                printf("\033[A");
                printf("Calculating edge lengths : [ %d / %d]\n", i, numberOfPolygons);
                i++;
            }
            outfile.close();
            if (Debug)
                printf("Closing File : %s\n", filename);
            cout.rdbuf(originalCoutBuffer);
            cout << "-----------------------------------------------------------------" << endl;
        }
        catch (...)
        {
            error("Failed to calculate edge lengths.\n");
        }
    }
};
