import vtk

colors = vtk.vtkNamedColors()

colors.SetColor('SkinColor', [240, 184, 160, 255])
colors.SetColor('BackfaceColor', [255, 229, 200, 255])
colors.SetColor('BkgColor', [51, 77, 102, 255])

# Load the VTK file
filename = "FullHead.mhd"
reader = vtk.vtkMetaImageReader()
reader.SetFileName(filename)
reader.Update()

# Extract the isosurfaces for bones and skin
iso_skin = vtk.vtkMarchingCubes()
iso_skin.SetInputConnection(reader.GetOutputPort())
iso_skin.SetValue(0, 500)  # Skin value


iso_bone = vtk.vtkMarchingCubes()
iso_bone.SetInputConnection(reader.GetOutputPort())
iso_bone.SetValue(0, 1150)  # Bone value

# Create mapper and actor for skin
skin_mapper = vtk.vtkPolyDataMapper()
skin_mapper.SetInputConnection(iso_skin.GetOutputPort())

skin_actor = vtk.vtkActor()
skin_actor.SetMapper(skin_mapper)
skin_actor.GetProperty().SetDiffuseColor(colors.GetColor3d('SkinColor'))
skin_actor.GetProperty().SetSpecular(.3)
skin_actor.GetProperty().SetSpecularPower(20)
skin_actor.GetProperty().SetOpacity(0.5)  # Skin opacity

# Create mapper and actor for bones
bone_mapper = vtk.vtkPolyDataMapper()
bone_mapper.SetInputConnection(iso_bone.GetOutputPort())

bone_actor = vtk.vtkActor()
bone_actor.SetMapper(bone_mapper)
bone_actor.GetProperty().SetDiffuseColor(colors.GetColor3d('Ivory'))

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
