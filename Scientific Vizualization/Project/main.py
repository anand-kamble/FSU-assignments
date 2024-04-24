import vtk

# Create a grid
grid = vtk.vtkStructuredGrid()

# Define the dimensions of the grid
dimensions = 5, 5, 5

# Set the dimensions of the grid
grid.SetDimensions(dimensions)

# Create points for the grid
points = vtk.vtkPoints()

# Generate points for the grid
for k in range(dimensions[2]):
    for j in range(dimensions[1]):
        for i in range(dimensions[0]):
            points.InsertNextPoint(i, j, k)

# Set the points for the grid
grid.SetPoints(points)

# Create a mapper
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(grid)

# Create an actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a renderer
renderer = vtk.vtkRenderer()

# Create a render window
render_window = vtk.vtkRenderWindow()
render_window.SetSize(600, 600)
render_window.AddRenderer(renderer)

# Create a render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Add the actor to the scene
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)

# Initialize the interactor and start the rendering loop
render_window.Render()
render_window_interactor.Start()
