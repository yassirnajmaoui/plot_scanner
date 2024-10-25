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
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonTransforms import(
	vtkTransform,
	vtkLinearTransform
	)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkPiecewiseFunction
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
    vtkRenderer,
	vtkColorTransferFunction,
    vtkVolume,
	vtkVolumeProperty
)
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonMath import vtkMatrix4x4

# TODO: Add arrows representing X, Y and Z

def render_scene(lut:np.ndarray, scanner_desc:dict, image_fname:str=None):
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

	renderer = vtkRenderer()

	if(image_fname is not None):

		# Step 1: Load volumetric data (e.g., from a .vtk or .nii file)
		# For demonstration, we use a VTK sample source (you'll use your actual data reader)
		reader = vtkNIFTIImageReader()  # Change to appropriate reader (e.g., for NIfTI)
		reader.SetFileName(image_fname)  # Path to your volumetric image file
		reader.Update()

		nifti_matrix = reader.GetQFormMatrix()  # or GetSFormMatrix() if applicable

		# Convert the affine matrix from NIFTI to a VTK matrix
		affine_matrix = vtkMatrix4x4()
		if nifti_matrix:
			affine_matrix.DeepCopy(nifti_matrix)
		# Get the max value of the image
		image_data = reader.GetOutput()
		scalars = image_data.GetPointData().GetScalars()
		np_scalars = np.array(scalars)
		mean_val = np.mean(np_scalars)
		max_val = np.max(np_scalars)

		# Step 2: Create a volume mapper and specify how to map the data
		volume_mapper = vtkSmartVolumeMapper()
		volume_mapper.SetInputConnection(reader.GetOutputPort())

		# Step 3: Set volume properties (e.g., opacity, shading)
		volume_property = vtkVolumeProperty()
		volume_property.ShadeOn()
		volume_property.SetInterpolationTypeToLinear()

		# Opacity transfer function (to control transparency)
		opacity_transfer_function = vtkPiecewiseFunction()
		opacity_transfer_function.AddPoint(0, 0.0)  # Completely transparent for low values
		opacity_transfer_function.AddPoint(max_val/3, 1.0)  # Fully opaque for high values
		volume_property.SetScalarOpacity(opacity_transfer_function)

		# Color transfer function (to control color based on intensity)
		color_transfer_function = vtkColorTransferFunction()
		color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black for low intensity
		color_transfer_function.AddRGBPoint(mean_val, 1.0, 0.75, 0.0)  # White for high intensity
		volume_property.SetColor(color_transfer_function)

		# Step 4: Create a volume actor and set its mapper and properties
		volume = vtkVolume()
		volume.SetMapper(volume_mapper)
		volume.SetProperty(volume_property)
		volume.SetUserMatrix(affine_matrix)  # Apply the NIFTI affine transformation

		renderer.AddVolume(volume)



	# Create a renderer, render window, and interactor.
	ren_win = vtkRenderWindow()
	ren_win.AddRenderer(renderer)
	ren_win.SetWindowName('Rendering a scanner')

	iren = vtkRenderWindowInteractor()
	iren.SetRenderWindow(ren_win)
	iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

	# Camera management
	camera = vtkCamera()
	camera.SetPosition(2*np.max(lut[:,0]), 2*np.max(lut[:,1]), 2*np.max(lut[:,2]))
	camera.SetFocalPoint(0, 0, 0)

	# Add the actor to the scene.
	renderer.AddActor(actor)
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

	lut  = np.fromfile('GE.lut', dtype=np.float32).reshape([-1,6])

	scanner_desc = dict()
	scanner_desc["crystalSize_z"] = 5.311
	scanner_desc["crystalSize_trans"] = 3.95
	scanner_desc["crystalDepth"] = 25

	render_scene(lut, scanner_desc, "test_image_GE.nii")
