#!/bin/env python
import numpy as np
import pyvista as pv
import json

# noinspection PyUnresolvedReferences
from vtkmodules.vtkInteractionStyle import (
		vtkInteractorStyleTrackballCamera
)
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonTransforms import(
	vtkTransform,
	vtkLinearTransform
	)
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersProgrammable import vtkProgrammableGlyphFilter
from vtkmodules.vtkFiltersGeneral import(
	vtkTransformFilter,
	vtkTransformPolyDataFilter)
from vtkmodules.vtkFiltersSources import (
    vtkConeSource,
    vtkCubeSource,
    vtkSphereSource,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
	vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

# Import SimpleITK if possible
try:
    import SimpleITK as sitk
    print("SimpleITK imported")
except ImportError:
    sitk = None
    print("SimpleITK is not available")


def read_scanner_lut(filepath: str):
	lut = np.fromfile(filepath, dtype=np.float32)
	lut = lut.reshape([-1,6])
	return lut

def read_json(filepath:str):
	with open(filepath) as json_file:
		data = json.load(json_file)
	return data

def render_scene(lut:np.ndarray, scanner_desc:dict, image_desc:dict=None):
	colors = vtkNamedColors()

	# Create the scanner actor
	# Create points.
	points = vtkPoints()
	for i in range(lut.shape[0]):
		x = lut[i][0]
		y = lut[i][1]
		z = lut[i][2]
		points.InsertNextPoint(lut[i][0],lut[i][1],lut[i][2])

	# Combine into a polydata.
	polydata = vtkPolyData()
	polydata.SetPoints(points)

	glyph_filter = vtkProgrammableGlyphFilter()
	glyph_filter.SetInputData(polydata)
	# Create the observer.
	observer = CalcGlyph(glyph_filter, lut, scanner_desc)
	glyph_filter.SetGlyphMethod(observer)
	# It needs a default glyph, but this should not be used.
	cone_source = vtkConeSource()
	glyph_filter.SetSourceConnection(cone_source.GetOutputPort())

	# Create a mapper and actor.
	mapper = vtkPolyDataMapper()
	mapper.SetInputConnection(glyph_filter.GetOutputPort())
	actor = vtkActor()
	actor.SetMapper(mapper)
	actor.GetProperty().SetColor(colors.GetColor3d('White'))

	if(image_desc != None):
		# Create the Image surface actor
		# Create cube.
		volume = vtkCubeSource()
		volume.SetXLength(image_desc["length_x"])
		volume.SetYLength(image_desc["length_y"])
		volume.SetZLength(image_desc["length_z"])
		volume.SetCenter([image_desc["off_x"],image_desc["off_y"],image_desc["off_z"]])
		volume.Update()

		# mapper
		volumeMapper = vtkPolyDataMapper()
		volumeMapper.SetInputData(volume.GetOutput())

		# Actor.
		volumeActor = vtkActor()
		volumeActor.GetProperty().EdgeVisibilityOn()
		volumeActor.SetMapper(volumeMapper)
		volumeActor.GetProperty().SetColor(colors.GetColor3d('DimGray'))

	# Create a renderer, render window, and interactor.
	renderer = vtkRenderer()
	ren_win = vtkRenderWindow()
	ren_win.AddRenderer(renderer)
	ren_win.SetWindowName('Rendering a scanner')

	iren = vtkRenderWindowInteractor()
	iren.SetRenderWindow(ren_win)
	iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

	# Camera management
	camera = vtkCamera()
	camera.SetPosition(2*max(lut[:,0]), 2*max(lut[:,1]), 2*max(lut[:,2]))
	camera.SetFocalPoint(0, 0, 0)

	# Add the actor to the scene.
	renderer.AddActor(actor)
	if image_desc!=None:
		renderer.AddActor(volumeActor)
	renderer.SetActiveCamera(camera)
	renderer.SetBackground(colors.GetColor3d('black'))

	# Render and interact.
	ren_win.Render()
	renderer.GetActiveCamera().Zoom(0.9)
	iren.Start()


class CalcGlyph(object):
	def __init__(self, glyph_filter, lut, desc):
		self.glyph_filter = glyph_filter
		self.lut  = lut
		self.desc = desc
		self.scale = [desc['crystalDepth'],desc['crystalSize_trans'],desc['crystalSize_z']]

	def __call__(self):
		point_coords = self.glyph_filter.GetPoint()
		i = self.glyph_filter.GetPointId()
		#print('Calling CalcGlyph for point ', self.glyph_filter.GetPointId())
		#print('Point coords are: ', point_coords[0], point_coords[1], point_coords[2])
		cube_source = vtkCubeSource()

		transform = vtkTransform()
		transform.Translate(point_coords)
		transform.RotateZ(np.arctan2(lut[i][4],lut[i][3])*180/np.pi)
		transform.Scale(self.scale)

		transformFilter = vtkTransformFilter()
		transformFilter.SetTransform(transform)
		transformFilter.SetInputConnection(cube_source.GetOutputPort())

		self.glyph_filter.SetSourceConnection(transformFilter.GetOutputPort())


# Main
if(__name__=='__main__'):
	lut  = read_scanner_lut('MOUSE.lut')
	scanner_desc = dict()
	scanner_desc["crystalSize_z"] = 1.1
	scanner_desc["crystalSize_trans"] = 1.1
	scanner_desc["crystalDepth"] = 1.06

	image_desc = read_json('img_params_MOUSE.json')

	render_scene(lut, scanner_desc, image_desc)
