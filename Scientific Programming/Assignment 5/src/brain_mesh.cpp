#ifndef __AMK__BRAIN_MESH__
#define __AMK__BRAIN_MESH__

#include <iostream>     ///< Input and output stream objects for console I/O.
#include <cstring>      ///< C-style string manipulation functions.
#include <string>       ///< String class for string manipulation.
#include <sstream>      ///< String stream processing.
#include <cstdlib>      ///< General utilities library, including memory management and program control functions.
#include <functional>   ///< Function objects and higher-order operations on functions.
#include <any>          ///< Safe and convenient container for single values of any type.
#include <vector>       ///< Dynamic array implementation for storing vectors.
#include <fstream>      ///< Input/output stream class to operate on files.
#include <stdexcept>    ///< Standard exception class.
#include <math.h>       ///< Mathematical functions.
#include "brain_mesh.h" ///< Header file for the brain mesh class.

using namespace std;

/**
 * @brief Constructor for initializing the BrainMesh object with the given ID and optional debug mode.
 *
 * This constructor initializes the BrainMesh object with the provided ID and optional debug mode.
 * It also displays a debug message indicating object creation.
 *
 * @param Id The ID to be assigned to the BrainMesh object.
 * @param debug_mode Whether debug messages should be printed (default is false).
 */
BrainMesh::BrainMesh(const string &id, bool debugMode)
{
    this->id = id;                           // ID of the BrainMesh object.
    this->debugMode = debugMode;             // Indicates whether debug messages should be printed.
    this->numberOfPoints = 0;                // Number of vertices in the mesh.
    this->numberOfPolygons = 0;              // Number of polygons in the mesh.
    this->originalCoutBuffer = cout.rdbuf(); // Stores the original output stream buffer.

    debug("Constructor: " + id);
}
/**
 * @brief Destructor for cleaning up resources.
 *
 * This destructor clears the ID string when the BrainMesh object is destroyed.
 */
BrainMesh::~BrainMesh()
{
    destroy();
}

/**
 * @brief Copy constructor for creating a new BrainMesh object by copying another object's ID and debug mode setting.
 *
 * This constructor creates a new BrainMesh object by copying the ID and debug mode setting
 * from another BrainMesh object. It also displays a debug message indicating object creation.
 *
 * @param other The BrainMesh object to be copied.
 */
BrainMesh::BrainMesh(const BrainMesh &other)
{
    id = "C_C_" + other.id;
    clone(other);
    debug("Copy Constructor: " + id);
}

/**
 * @brief Copy assignment operator for assigning the ID and debug mode from another BrainMesh object to this object.
 *
 * This operator assigns the ID and debug mode from another BrainMesh object to this object.
 * It also displays a debug message indicating object assignment.
 *
 * @param other The BrainMesh object whose ID and debug mode are to be assigned.
 * @return Reference to this BrainMesh object after assignment.
 */
BrainMesh &BrainMesh::operator=(const BrainMesh &other)
{
    if (this != &other)
    {
        id = "C_A_O_" + other.id;
        clone(other);
        debug("Copy Assignment Operator: " + id);
    }

    return *this;
}

/**
 * @brief Calculates the Euclidean distance between two points in 3D space.
 *
 * This function computes the length of the edge (distance) between two vertices
 * represented by the given 3D coordinates.
 *
 * @param p1 Vertex 1 represented as a vector of three floats [x1, y1, z1].
 * @param p2 Vertex 2 represented as a vector of three floats [x2, y2, z2].
 * @return The Euclidean distance between the two vertices.
 */
double BrainMesh::edgeLength(vector<float> p1, vector<float> p2)
{
    return any_cast<double>(this->errorHandler([this, p1, p2]()
                                               {
                               auto x1 = p1[0];
                               auto x2 = p2[0];

                               auto y1 = p1[1];
                               auto y2 = p2[1];

                               auto z1 = p1[2];
                               auto z2 = p2[2];

                               // Applying the 3D distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
                               return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2)); }));
}
/**
 * @brief Displays the ID of the BrainMesh object.
 *
 * This function prints the ID of the BrainMesh object to the standard output.
 */
void BrainMesh::info()
{
    cout << "ID: " << this->id << endl;
}

/**
 * @brief Load mesh data from a VTK file into the BrainMesh object.
 *
 * This function reads vertex and polygon data from a VTK file specified by the given filename.
 * It expects the VTK file to have a specific structure with POINTS and POLYGONS sections.
 * The loaded data is stored in the `points` and `polygons` vectors respectively.
 *
 * @param filename The path to the VTK file to be loaded.
 *
 * @throws std::invalid_argument If the file cannot be opened or if its structure is invalid.
 *
 * @details
 * The function opens the specified VTK file and processes its contents line by line. It first skips
 * the header lines and then reads vertex and polygon data. The function populates the `points`
 * vector with vertex coordinates and the `polygons` vector with polygon indices. After successfully
 * loading the data, the function prints debug information including the number of points and polygons
 * found in the loaded file.
 */
void BrainMesh::Load(const char *filename)
{
    this->errorHandler([this, filename]()
                       {
        string file(filename); ///< The name of the VTK file to load.
        string vtkString;      ///< Temporary variable to store lines read from the file.
        ifstream VTKFileStream;///< Input stream to read data from the VTK file.
        VTKFileStream.open(filename); ///< Opening the VTK file.

        this->debug("Opening File : " + string(filename));

        // Check if the file is successfully opened.
        if (VTKFileStream.is_open())
        {
            // Skipping the first 4 lines of the VTK file.
            for (size_t i = 0; i < 4; i++)
            {
                getline(VTKFileStream, vtkString);
            }

            // Reading and processing POINTS section.
            getline(VTKFileStream, vtkString);
            auto parts = split(vtkString);
            if (vtkString.rfind("POINTS", 0) == 0)
            {
                numberOfPoints = stoi(parts[1]); ///< Number of vertices in the mesh.
            }

            // Reading vertex data and populating the `points` vector.
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

            // Reading and processing POLYGONS section.
            getline(VTKFileStream, vtkString);
            if (vtkString.rfind("POLYGONS", 0) == 0)
            {
                parts = split(vtkString);
                numberOfPolygons = stoi(parts[1]); ///< Number of polygons in the mesh.
            }

            // Reading polygon data and populating the `polygons` vector.
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

            VTKFileStream.close(); ///< Closing the VTK file.
            this->debug("Closing File : " + string(filename));
            cout << "-----------------------------------------------------------------" << endl;
            cout << "File Loaded - " << file << endl;
            cout << "Found " << this->points.size() << " points. " << endl;
            cout << "Found " << this->polygons.size() << " polygons. " << endl;
            cout << "-----------------------------------------------------------------" << endl;
        }
        else
        {
            // Throw an exception if the file cannot be opened.
            throw invalid_argument("Failed to open the file - " + file);
        }
        return 1; });
}

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
double BrainMesh::areaOfTriangle(vector<float> p1, vector<float> p2, vector<float> p3)
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
                pow(y2 * z1 - y3 * z1 - y1 * z2 + y3 * z2 + y1 * z3 - y2 * z3, 2)) /
           2.0;
}

/**
 * @brief Calculates the total surface area of the mesh by summing the areas of all polygons.
 *
 * This function iterates through the polygons of the mesh, calculates the area of each triangle,
 * and accumulates the total surface area. The area of each triangle is computed using the
 * Heron's formula based on the vertices' coordinates.
 *
 * @return The total surface area of the mesh.
 */
double BrainMesh::TotalSurfaceArea()
{
    return any_cast<double>(errorHandler([this]()
                                         {
            double area = 0.0;
            for (auto polygon : polygons)
            {
                // Calculate the area of each triangle formed by the polygon vertices
                area += this->areaOfTriangle(points[polygon[0]], points[polygon[1]], points[polygon[2]]);
            }
            return area; }));
}

/**
 * @brief Calculates edge lengths of polygons and saves them to a file.
 *
 * This method calculates the edge lengths of each polygon in the mesh and saves the lengths to the specified file.
 * It iterates through the polygons, computes the lengths of edges by invoking the edgeLength function,
 * and writes the results to the specified output file. The progress of the calculation is displayed in the console.
 *
 * @param filename The name of the file where edge lengths will be saved.
 * @return void
 */
void BrainMesh::calculateEdgeLengths(const char *filename)
{
    this->errorHandler([this, filename]()
                       {
        cout << "Edge lengths will be saved in file : " << filename << endl;
        ofstream outfile(filename); ///< Output file stream to save edge lengths.
        debug("Opening File : " +  string(filename));

        cout.rdbuf(outfile.rdbuf()); ///< Redirects standard output to the output file.
        int i = 1;
        printf("Calculating edge lengths : [ %d / %d]\n", i, numberOfPolygons);

        for (auto polygon : polygons)
        {
            cout << this->edgeLength(points[polygon[0]], points[polygon[1]]) << endl;
            cout << this->edgeLength(points[polygon[1]], points[polygon[2]]) << endl;
            cout << this->edgeLength(points[polygon[0]], points[polygon[2]]) << endl;

            printf("\033[A"); ///< Moves cursor up one line in the console.
            printf("Calculating edge lengths : [ %d / %d]\n", i, numberOfPolygons);
            i++;
        }

        outfile.close(); ///< Closes the output file stream.
        debug("Closing File : " + string(filename));

        cout.rdbuf(originalCoutBuffer); ///< Restores standard output.
        cout << "-----------------------------------------------------------------" << endl;

        return 0; });
}

/**
 * @brief Calculates the total surface area of polygons surrounding each vertex and saves the results to a file.
 *
 * This method calculates the total surface area of all polygons surrounding each vertex in the mesh and writes
 * the results to the specified output file. It iterates through each vertex, computes the total surface area
 * of polygons connected to the vertex, and saves the area to the output file. The progress of the calculation is
 * displayed in the console.
 *
 * @param filename The name of the file where the calculated surface areas will be saved.
 */
void BrainMesh::SurroundingAreaForVertex(const char *filename)
{
    this->errorHandler([this, filename]()
                       {
        cout << "Area of surrounding polygons will be saved in file : " << filename << endl;
        printf("Calculating area of surrounding polygons : [ %d / %d]\n", 0, numberOfPoints);
        ofstream outfile(filename); ///< Output file stream to save surface areas.
        this->debug("Opening File : " + string(filename));

        cout.rdbuf(outfile.rdbuf()); ///< Redirects standard output to the output file.
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
            printf("\x1b[A"); ///< Moves cursor up one line in the console.
            printf("Calculating area of surrounding polygons : [ %d / %d]\n", i + 1, numberOfPoints);
        }

        outfile.close(); ///< Closes the output file stream.
        debug("Closing File : " +  string(filename));

        cout.rdbuf(originalCoutBuffer); ///< Restores standard output.
        cout << "-----------------------------------------------------------------" << endl;

        return 0; });
}
/**
 * @brief Calculate the total surface area of polygons surrounding a given vertex.
 *
 * This method iterates through all polygons in the mesh and accumulates the
 * surface area of any polygons that contain the vertex with the specified ID.
 *
 * @param vertex_id The ID of the vertex to calculate area around.
 * @return The total surface area of polygons surrounding the vertex.
 */
double BrainMesh::areaAroundVertex(int vertex_id)
{
    return any_cast<double>(this->errorHandler([this, vertex_id]()
                                               {
    double surfaceArea;

        surfaceArea = 0.0;
        for (auto polygon : polygons)
        {
            if (vertex_id == polygon[0] || vertex_id == polygon[1] || vertex_id == polygon[2])
            {
                surfaceArea += areaOfTriangle(points[polygon[0]], points[polygon[1]], points[polygon[2]]);
            }
        }
        return surfaceArea; }));
};

/**
 * @brief Private utility function for debugging.
 *
 * This function prints the given message to the standard output
 * if the debug mode is enabled.
 *
 * @param msg The message to be printed.
 */
void BrainMesh::debug(string msg)
{
    if (this->debugMode)
        cout << msg << endl;
}
/**
 * @brief Clones the debug mode setting and mesh properties from another BrainMesh object.
 *
 * This function copies the debug mode setting, number of points, and number of polygons
 * from another BrainMesh object, allowing the current object to mirror the properties of the source object.
 *
 * @param other The source BrainMesh object to clone the debug mode and properties from.
 */
void BrainMesh::clone(const BrainMesh &other)
{
    debugMode = other.debugMode;
    this->numberOfPoints = other.numberOfPoints;
    this->numberOfPolygons = other.numberOfPolygons;
}
/**
 * @brief Resets the BrainMesh object to its initial state.
 *
 * This function performs a cleanup operation, resetting the BrainMesh object by clearing its points and polygons,
 * resetting the number of points and polygons to zero, turning off debug mode, and clearing the ID string.
 * After calling this function, the BrainMesh object is restored to its initial state, ready for reuse or destruction.
 */
void BrainMesh::destroy()
{
    this->points.clear();       ///< Clears the vector storing vertices.
    this->polygons.clear();     ///< Clears the vector storing polygons.
    this->numberOfPoints = 0;   ///< Resets the number of points to zero.
    this->numberOfPolygons = 0; ///< Resets the number of polygons to zero.
    this->debugMode = false;    ///< Disables debug mode.
    this->id.clear();           ///< Clears the ID string.
}
/**
 * @brief Handles errors occurring during callback execution and terminates the program if necessary.
 *
 * This function takes a callback function as a parameter and executes it. If the callback function encounters
 * an exception of type std::exception, it catches the exception, prints an error message including the ID of the
 * BrainMesh object, and calls the `destroy` method to clean up the object's resources before terminating the program.
 * The error message provides context about the object involved in the error and the specific exception message received.
 * This function ensures graceful error handling for various operations within the BrainMesh class.
 *
 * @param callback A lambda function or function object to execute within a try-catch block.
 * @return The return value of the executed callback function.
 */
any BrainMesh::errorHandler(function<any()> callback)
{
    try
    {
        return callback();
    }
    catch (const std::exception &e)
    {
        auto msg = "[" + this->id + "] : " + string(e.what());
        cerr << msg << endl;
        this->destroy(); ///< Cleans up the current BrainMesh object's resources in case of an error.
        cout << "Exiting the program." << endl;
        exit(EXIT_FAILURE); ///< Exits the program with a failure status code.
    }
}

/**
 * @brief Splits the provided string into parts separated by the given separator.
 *
 * @param toSplit The string to split.
 * @param separator Separator that determines how to split the string.
 * @return vector<string>
 */
vector<string> BrainMesh::split(const string &toSplit, char separator)
{
    vector<string> parts;
    string s;
    istringstream stream(toSplit);
    this->debug("String to split: " + toSplit);
    while (getline(stream, s, separator))
    {
        this->debug("- " + s);
        parts.push_back(s);
    }
    return parts;
}

#endif
