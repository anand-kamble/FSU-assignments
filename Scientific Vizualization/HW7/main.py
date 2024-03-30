"""
@file: main.py
@author: Anand Kamble
@date: March 30, 2024
"""

import numpy as np
from vtk import (
    vtkFixedPointVolumeRayCastMapper,
    vtkStructuredPointsReader,
    vtkPiecewiseFunction,
    vtkColorTransferFunction,
    vtkVolumeProperty,
    vtkFixedPointVolumeRayCastMapper,
    vtkVolume,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkCommand,
)

# Choose colors from Viridis color map
# Found on: https://www.kennethmoreland.com/color-advice/
Colors = {"SKIN": np.array([74, 194, 109]), "SKULL": np.array([159, 218, 58])}

# Load the mummy dataset
reader = vtkStructuredPointsReader()
reader.SetFileName("mummy.128.vtk")

print()

# Create the transfer mapping scalar value to opacity
opacityTransferFunction = vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0, 0.0)
opacityTransferFunction.AddPoint(70, 0.25)  # Skin
opacityTransferFunction.AddPoint(90, 0.05)
opacityTransferFunction.AddPoint(100, 0.75)  # Skull
opacityTransferFunction.AddPoint(200, 0.3)
opacityTransferFunction.AddPoint(255, 1.0)

# Create the transfer mapping scalar value to color
colorTransferFunction = vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(70, *(Colors["SKIN"] / 255))  # Skin
colorTransferFunction.AddRGBPoint(100, *(Colors["SKULL"] / 255))  # Skull


# Create the volume property
volumeProperty = vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)

# Create the volume mapper
volumeMapper = vtkFixedPointVolumeRayCastMapper()
volumeMapper.SetInputConnection(reader.GetOutputPort())

# Create the volume
volume = vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# Create the renderer
renderer = vtkRenderer()
renderer.AddVolume(volume)
renderer.SetBackground(0.0, 0.0, 0.0)

# Create the render window
renderWindow = vtkRenderWindow()
renderWindow.SetSize(800, 800)
renderWindow.AddRenderer(renderer)

# Create the interactor
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

timer_id = interactor.CreateRepeatingTimer(10)  # 10 milliseconds


def rotate_camera(obj, event):
    """
    Rotating the camera by a small amount every frame.
    Found the Azimuth function on:
    https://vtk.org/doc/nightly/html/classvtkCamera.html
    """

    camera = renderer.GetActiveCamera()
    camera.Azimuth(1.0)
    renderWindow.Render()


# Bind the rotate_camera function to the timer event
interactor.AddObserver(vtkCommand.TimerEvent, rotate_camera)


# Render and interact
renderWindow.Render()
interactor.Start()
