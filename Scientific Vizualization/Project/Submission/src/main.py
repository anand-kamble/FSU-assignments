import numpy as np
from util.hdf5 import Dataset
import vtk
from vtk.util import numpy_support

dataset = Dataset("sedov_hdf5_plt_cnt_*","dataset")
dataset.searchFiles()
dataset.loadDataset()




vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2], [2, 3, 0]])

density_data = dataset.data[0]['dens']
print(density_data[0])
grid = vtk.vtkStructuredGrid()
grid.SetDimensions(density_data.shape[0],density_data.shape[2],density_data.shape[3])

# Set the grid points
points = vtk.vtkPoints()
for k in range(density_data.shape[3]):
    for j in range(density_data.shape[2]):
        for i in range(density_data.shape[0]):
            points.InsertNextPoint(i, j, k)
grid.SetPoints(points)

# Set the density data
density_array = vtk.vtkDoubleArray()
density_array.SetName('Density')
for value in density_data.flatten():
    density_array.InsertNextValue(value)
grid.GetPointData().AddArray(density_array)

mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(grid)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

image_data = vtk.vtkImageData()
image_data.SetDimensions(128,density_data.shape[2],density_data.shape[3])
image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

# Copy the NumPy data to the VTK image data
for x in range(density_data.shape[0]):
    for y in range(density_data.shape[2]):
        for z in range(density_data.shape[3]):
            image_data.SetScalarComponentFromFloat(x, y, z, 0, density_data[x][0][y][z])

# Create a volume mapper
volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
volume_mapper.SetInputData(image_data)

# Create a volume property
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(vtk.vtkColorTransferFunction())
volume_property.SetScalarOpacity(vtk.vtkPiecewiseFunction())

# Create a volume
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Create a VTK renderer and add the actor
renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
renderer.AddVolume(volume)




# Create a VTK window and render the scene
window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800, 800)
window.Render()
time_point = 0

def update_volume(obj, event):
    global time_point
    time_point = (time_point + 1) % 100
    density_data = dataset.data[time_point]['dens']
    print(density_data.shape)
    # Update the image data with the new time point
    for x in range(density_data.shape[0]):
        for y in range(density_data.shape[2]):
            for z in range(density_data.shape[3]):
                
                image_data.SetScalarComponentFromFloat(x, y, z, 0, density_data[x][0][y][z])
    
    # Update the volume and render
    volume.Modified()
    window.Render()

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(window)

iren.AddObserver('TimerEvent', update_volume)
iren.CreateRepeatingTimer(100)  # Update every 100 ms


iren.Start()

# # Create a vtkPoints object from the NumPy array
# vtk_points = vtk.vtkPoints()
# vtk_points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))

# # for point in points_np:
#     # points.InsertNextPoint(point)

# vtk_cells = vtk.vtkCellArray()
# for face in faces:
#     vtk_cells.InsertNextCell(3, face)


# # Create a vtkPolyData object and add the points to it
# polydata = vtk.vtkPolyData()
# polydata.SetPoints(vtk_points)
# polydata.SetPolys(vtk_cells)

# # Create a vtkCellArray to store the cell connectivity
# # cells = vtk.vtkCellArray()
# # for i in range(len(points_np)):
# #     cells.InsertNextCell(1)
# #     cells.InsertCellPoint(i)
# # polydata.SetVerts(cells)

# # Create a vtkPolyDataMapper and vtkActor to render the points
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputData(polydata)

# actor = vtk.vtkActor()
# actor.SetMapper(mapper)

# # Create a vtkRenderer and vtkRenderWindow to display the points
# renderer = vtk.vtkRenderer()
# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)
# renderer.AddActor(actor)

# # Render the scene
# render_window.Render()

# # Start the event loop  
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(render_window)
# iren.Start()