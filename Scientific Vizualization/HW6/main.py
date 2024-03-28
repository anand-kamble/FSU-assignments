import vtk

# Load the VTK file
filename = "head.60.vtk"
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(filename)
reader.Update()

# Extract the isosurfaces for bones and skin
iso_skin = vtk.vtkMarchingCubes()
iso_skin.SetInputConnection(reader.GetOutputPort())
iso_skin.SetValue(0, 25)  # Skin value

iso_bone = vtk.vtkMarchingCubes()
iso_bone.SetInputConnection(reader.GetOutputPort())
iso_bone.SetValue(0, 75)  # Bone value

# Create mapper and actor for skin
skin_mapper = vtk.vtkPolyDataMapper()
skin_mapper.SetInputConnection(iso_skin.GetOutputPort())

skin_actor = vtk.vtkActor()
skin_actor.SetMapper(skin_mapper)
skin_actor.GetProperty().SetColor(1, 0.8, 0.6)  # Skin color

# Create mapper and actor for bones
bone_mapper = vtk.vtkPolyDataMapper()
bone_mapper.SetInputConnection(iso_bone.GetOutputPort())

bone_actor = vtk.vtkActor()
bone_actor.SetMapper(bone_mapper)
bone_actor.GetProperty().SetColor(1, 1, 1)  # Bone color

# Create renderer and render window
renderer = vtk.vtkRenderer()
renderer.AddActor(skin_actor)
renderer.AddActor(bone_actor)
renderer.SetBackground(0.1, 0.2, 0.4)

render_window = vtk.vtkRenderWindow()
render_window.SetSize(800, 800)
render_window.AddRenderer(renderer)


# Create render window interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Start the visualization
render_window.Render()
interactor.Start()
