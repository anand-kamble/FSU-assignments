#ifndef BRAIN_MESH_H
#define BRAIN_MESH_H

#include <string>
#include <vector>
#include <any>
#include <functional>

using namespace std;

class BrainMesh
{

public:
  BrainMesh(const string &id, bool debugMode = false);

  ~BrainMesh();

  BrainMesh(const BrainMesh &other);

  BrainMesh &operator=(const BrainMesh &other);

  void info();

  void Load(const char *filename);

  double TotalSurfaceArea();

  void calculateEdgeLengths(const char *filename);

  void SurroundingAreaForVertex(const char *filename);

private:
  string id;
  bool debugMode;
  int numberOfPoints;
  int numberOfPolygons;
  vector<vector<float>> points;
  vector<vector<int>> polygons;

  void debug(string msg);

  void clone(const BrainMesh &other);

  void destroy();

  any errorHandler(function<any()> callback);

  vector<string> split(const string &toSplit, char separator = ' ');
};

#endif