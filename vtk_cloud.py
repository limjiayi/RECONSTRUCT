"""
VTK shortcut functions

For more about VTK, visit:
    http://www.vtk.org/
    http://www.vtk.org/Wiki/VTK/Examples/Python
"""

import vtk
import sys
import numpy

from vtk.util.colors import peacock

def vtk_point_cloud(points, colors=[], point_size=2):
    """
    Represent a point cloud in VTK
    
    Parameters
    ----------
    points :  numpy array, each row is a point
    colors : list of colors, one per point
    point_size : rendering size for the points
    
    Returns
    -------
    actor : vtkActor representing the point cloud
    """
    nb = len(points);
    vtk_points = vtk.vtkPoints();
    vtk_verts = vtk.vtkCellArray();
    if colors:
        vtk_colors = vtk.vtkUnsignedCharArray();
        vtk_colors.SetNumberOfComponents(3);
        vtk_colors.SetName( "Colors");
        
    for i in range(0,nb):
        
        p = points[i]
        if len(p) >= 3:
            coords = [p[0],p[1],p[2]]
        elif len(p) == 2:
            coords = [p[0],p[1],0]
        elif len(p) == 1:
            coords = [p[0],0,0]
        else:
            print "**ERROR** wrong dimension"
            sys.exit(1)
        
        id = vtk_points.InsertNextPoint( *coords )
        vtk_verts.InsertNextCell(1)
        vtk_verts.InsertCellPoint(id)
        if colors:
            vtk_colors.InsertNextTuple3( *colors[i] )
    
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(vtk_verts)
    if colors:
        poly.GetPointData().SetScalars(vtk_colors)
    poly.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToPoints
    actor.GetProperty().SetPointSize( point_size )

    return actor

def vtk_basic( actors ):
    """
    Create a window, renderer, interactor, add the actors and start the thing
    
    Parameters
    ----------
    actors :  list of vtkActors
    
    Returns
    -------
    nothing
    """     
    
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600,600)
    # ren.SetBackground( 1, 1, 1)
 
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a )

    #enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def vtk_Nviews( actors ):
    """
    Create a window, an interactor and one renderer per actor
    
    Parameters
    ----------
    actors :  list of vtkActors
    
    Returns
    -------
    nothing
    """    
    N = len(actors)
    
    # create a rendering window and renderers
    renderers = [vtk.vtkRenderer() for i in range(N)]
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize( 600, 600 )
    for i in range(N):
        # split the viewport
        renderers[i].SetViewport(0,float(N-i-1)/N,1,float(N-i)/N)
        renderers[i].SetBackground( 1, 1, 1)
        renderers[i].AddActor( actors[i] )
        renWin.AddRenderer(renderers[i])
 
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    #enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def vtk_show_points( points, colors=[] ):
    """
    Display a point cloud
    
    Parameters
    ----------
    points :  numpy array, each row is a point
    colors : list of colors, one per point
    
    Returns
    -------
    nothing
    """     
    point_cloud = vtk_point_cloud(points,colors)
    vtk_basic( [point_cloud] )


def vtk_colored_graph(points, edges, colors=[], line_width=2):
    """
    Represent a graph in VTK
    
    Parameters
    ----------
    points :  numpy array, each row is a point
    edges : numpy array of edges, each row is of the form
            [ point_1, point_2, distance ]      
    colors : list of colors, one per point
    line_width : rendering size for the lines
    
    Returns
    -------
    actor : vtkActor representing the graph
    """
    nb_points = len(points)    
    vtk_points = vtk.vtkPoints()
    vtk_lines = vtk.vtkCellArray()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName( "Colors")

    if (len(colors) ==0):
        for i in range(0,len(edges)):
            colors.append((0, 164, 180))
    
    for i in range(0,nb_points):
        
        p = points[i]
        if len(p) >= 3:
            coords = [p[0],p[1],p[2]]
        elif len(p) == 2:
            coords = [p[0],p[1],0]
        elif len(p) == 1:
            coords = [p[0],0,0]
        else:
            print "**ERROR** wrong dimension"
            sys.exit(1)
        
        id = vtk_points.InsertNextPoint( *coords )

    for i in range(0,len(edges)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,edges[i][0])
        line.GetPointIds().SetId(1,edges[i][1])
        vtk_lines.InsertNextCell(line)
        vtk_colors.InsertNextTuple3( *colors[i] )
        
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(vtk_lines)
    poly.GetCellData().SetScalars(vtk_colors);
    poly.Update()
    
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(poly)

    tubes = vtk.vtkTubeFilter()
    tubes.SetInputConnection(cleaner.GetOutputPort())
    tubes.SetRadius(0.1)
    tubes.SetNumberOfSides(6)
    mapEdges = vtk.vtkPolyDataMapper()
    mapEdges.SetInputConnection(tubes.GetOutputPort())
    edgeActor = vtk.vtkActor()
    edgeActor.SetMapper(mapEdges)
    edgeActor.GetProperty().SetSpecularColor(1, 1, 1)
    edgeActor.GetProperty().SetSpecular(0.3)
    edgeActor.GetProperty().SetSpecularPower(20)
    edgeActor.GetProperty().SetAmbient(0.2)
    edgeActor.GetProperty().SetDiffuse(0.8)
    return edgeActor

def vtk_triangles(points, triangles, colors=[]):
    """
    Display triangles in VTK
    
    Parameters
    ----------
    points :  numpy array, each row is a point
    triangle : numpy array of vertices, each row is of the form
            [ point_1, point_2, point_3 ]      
    colors : list of colors, one per triangle
    
    Returns
    -------
    actor : vtkActor representing the triangles
    """    
    nb_points = len(points)
    vtk_points = vtk.vtkPoints()
    vtk_triangles = vtk.vtkCellArray()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName( "Colors")

    if (len(colors) ==0):
        for i in range(0,nb_points):
            vtk_colors.InsertNextTuple3(0, 164, 180)
    else:
        for i in range(0,nb_points):
            vtk_colors.InsertNextTuple3( *colors[i] )
    
    for i in range(0,nb_points):
        
        p = points[i]
        if len(p) >= 3:
            coords = [p[0],p[1],p[2]]
        elif len(p) == 2:
            coords = [p[0],p[1],0]
        elif len(p) == 1:
            coords = [p[0],0,0]
        else:
            print "**ERROR** wrong dimension"
            sys.exit(1)
        
        id = vtk_points.InsertNextPoint( *coords )

    for i in range(0,len(triangles)):
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0,triangles[i][0])
        triangle.GetPointIds().SetId(1,triangles[i][1])
        triangle.GetPointIds().SetId(2,triangles[i][2])
        vtk_triangles.InsertNextCell(triangle)
        
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(vtk_triangles)
    poly.GetPointData().SetScalars(vtk_colors)
    poly.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(poly)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cleaner.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor    