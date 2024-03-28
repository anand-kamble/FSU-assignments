"""
Assignment - Half Brain Stucture

File: visualize_vtk_data.py
Author: Anand Kamble
Date: 18th March 2024

"""

import vtk

# We start by reading some data.
fran = vtk.vtkPolyDataReader()
fran.SetFileName("C:/Users/91911/Documents/Codes/FSU/Scientific Vizualization/HW5/src/Cort_lobe_poly.vtk")

# Apply normals to the data
normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(fran.GetOutputPort())
normals.FlipNormalsOn()

# Create mapper and actor
franMapper = vtk.vtkPolyDataMapper()
franMapper.SetInputConnection(normals.GetOutputPort())
franActor = vtk.vtkActor()
franActor.SetMapper(franMapper)
franActor.GetProperty().SetColor(1.0, 0.49, 0.25)

# create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set the background and size
ren.AddActor(franActor)
ren.SetBackground(1, 1, 1)
renWin.SetSize(800, 800)
renWin.SetWindowName('Cort_lobe_poly.vtk')

# Set camera properties
cam1 = vtk.vtkCamera()
cam1.SetClippingRange(0.0475572, 2.37786)
cam1.SetFocalPoint(0.052665, -0.129454, -0.0573973)
cam1.SetPosition(0.327637, -0.116299, -0.256418)
cam1.SetViewUp(-0.0225386, 0.999137, 0.034901)
ren.SetActiveCamera(cam1)

# enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()