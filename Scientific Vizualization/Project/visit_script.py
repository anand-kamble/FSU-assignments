# Import the necessary modules

import visit



# Open the data file

visit.OpenDatabase("dataset/*",0,"FLASH")



# Add a plot

visit.AddPlot("Pseudocolor", "dens")



# Set the plot attributes

pc = visit.PseudocolorAttributes()

pc.colorTableName = "Hot"

pc.min = 0

pc.max = 10

visit.SetPlotOptions(pc)



# Draw the plot

visit.DrawPlots()



# Set the view

visit.View3DAtts = visit.View3DAttributes()

visit.View3DAtts.viewNormal = (0, 0, 1)

visit.View3DAtts.focus = (0, 0, 0)

visit.View3DAtts.viewUp = (0, 1, 0)

visit.View3DAtts.parallelScale = 17.3205

visit.View3DAtts.nearPlane = -34.641

visit.View3DAtts.farPlane = 34.641

visit.View3DAtts.perspective = 1

visit.SetView3D(visit.View3DAtts)



# Save the plot

visit.SaveWindow()
