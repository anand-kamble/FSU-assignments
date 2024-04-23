import numpy as np
from util.hdf5 import Dataset
import vtk
import vtk.util

dataset = Dataset("sedov_hdf5_plt_cnt_*","dataset")
dataset.searchFiles()
dataset.loadDataset()

print(len(dataset.data))

points_np = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

# Create a vtkPoints object from the NumPy array
points = vtk.vtkPoints()
for point in points_np:
    points.InsertNextPoint(point)

# Create a vtkPolyData object and add the points to it
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# Create a vtkCellArray to store the cell connectivity
cells = vtk.vtkCellArray()
for i in range(len(points_np)):
    cells.InsertNextCell(1)
    cells.InsertCellPoint(i)
polydata.SetVerts(cells)

# Create a vtkPolyDataMapper and vtkActor to render the points
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polydata)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a vtkRenderer and vtkRenderWindow to display the points
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
renderer.AddActor(actor)

# Render the scene
render_window.Render()

# Start the event loop
vtk.vtkRenderWindowInteractor().Start()


vtk.vtkRenderWindowInteractor().Start()