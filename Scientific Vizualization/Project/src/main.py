import numpy as np
from util.hdf5 import Dataset
import vtk
from vtk.util import numpy_support
# import matplotlib.pyplot as plt
# from matplotlib import cm

dataset = Dataset("sedov_hdf5_plt_cnt_*","dataset")
dataset.searchFiles()
dataset.loadDataset()

points_3d = np.random.rand(100, 3)
# print(points_3d.shape)

dataSlice = dataset.data[0]['dens']
# print(dataSlice.shape)
points_3d = np.squeeze(dataSlice,axis=1)

print(f"Points: {points_3d[0][0].shape}")
# # print(new_slice.shape)
# x = np.linspace(0, 1, 16)
# y = np.linspace(0, 1, 16)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(x,y,points_3d[0], 50, cmap=cm.coolwarm)


# plt.show()
# Convert the numpy array to a VTK data structure
# vtkPoints = vtk.vtkPoints()
# for point in points_3d:
#     print(f"Point: {point}")
#     vtkPoints.InsertNextPoint(point)
np_array = points_3d
                                                                                     
vtk_image_data = vtk.vtkImageData()
vtk_image_data.SetDimensions(np_array.shape)
vtk_image_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(np_array.ravel(), deep=True))

mc = vtk.vtkMarchingCubes()
mc.SetInputData(vtk_image_data)
mc.ComputeNormalsOn()
mc.SetValue(0, 0.5)
mc.Update()

# Get the output mesh
poly_data = mc.GetOutput()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)

light = vtk.vtkLight()
light.SetPosition(0, 0, 1)
light.SetFocalPoint(0, 0, 0)
renderer.AddLight(light)

light2 = vtk.vtkLight()
light2.SetPosition(0, 0, -1)
light2.SetFocalPoint(0, 0, 0)
renderer.AddLight(light2)

camera = vtk.vtkCamera()
camera.SetPosition(0, 0, 60)
camera.SetFocalPoint(0, 0, 0)
renderer.SetActiveCamera(camera)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800, 800)
window.Render()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.Start()