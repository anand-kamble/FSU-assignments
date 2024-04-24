import numpy as np
from util.hdf5 import Dataset
import vtk
from vtk.util import numpy_support

dataset = Dataset("sedov_hdf5_plt_cnt_*","dataset")
dataset.searchFiles()
dataset.loadDataset()

print(dataset.data[0]['dens'][0])

vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2], [2, 3, 0]])

# Create a vtkPoints object from the NumPy array
vtk_points = vtk.vtkPoints()
vtk_points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))

# for point in points_np:
    # points.InsertNextPoint(point)

vtk_cells = vtk.vtkCellArray()
for face in faces:
    vtk_cells.InsertNextCell(3, face)


# Create a vtkPolyData object and add the points to it
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)
polydata.SetPolys(vtk_cells)

# Create a vtkCellArray to store the cell connectivity
# cells = vtk.vtkCellArray()
# for i in range(len(points_np)):
#     cells.InsertNextCell(1)
#     cells.InsertCellPoint(i)
# polydata.SetVerts(cells)

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
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(render_window)
iren.Start()