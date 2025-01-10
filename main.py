#!/bin/env python
import numpy as np
import argparse

# noinspection PyUnresolvedReferences
from vtkmodules.vtkInteractionStyle import (
        vtkInteractorStyleTrackballCamera
)
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import(
    vtkPoints
)
from vtkmodules.vtkCommonTransforms import(
    vtkTransform,
    vtkLinearTransform
    )
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkPiecewiseFunction, vtkColor3d
from vtkmodules.vtkFiltersProgrammable import vtkProgrammableGlyphFilter
from vtkmodules.vtkFiltersGeneral import(
    vtkTransformFilter,
    vtkTransformPolyDataFilter)
from vtkmodules.vtkFiltersSources import (
    vtkConeSource,
    vtkCubeSource,
    vtkArrowSource,
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
from vtkmodules.vtkCommonMath import(
    vtkMatrix4x4
)

def create_arrow(color, direction, arrow_length):
    """
    Creates an arrow actor with a specified color and direction.
    :param color: Color for the arrow in RGB format.
    :param direction: A tuple representing the (x, y, z) direction of the arrow.
    :return: Configured arrow actor.
    """
    # Create arrow source
    arrow_source = vtkArrowSource()

    # Transform to orient the arrow in the specified direction
    transform = vtkTransform()
    transform.Scale(arrow_length, arrow_length, arrow_length)
    if direction == (1, 0, 0):  # X-axis (no rotation needed)
        pass
    elif direction == (0, 1, 0):  # Y-axis
        transform.RotateZ(90)
    elif direction == (0, 0, 1):  # Z-axis
        transform.RotateY(-90)

    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())
    transform_filter.SetTransform(transform)

    # Create a mapper and actor
    arrow_mapper = vtkPolyDataMapper()
    arrow_mapper.SetInputConnection(transform_filter.GetOutputPort())

    arrow_actor = vtkActor()
    arrow_actor.SetMapper(arrow_mapper)
    arrow_actor.GetProperty().SetColor(color)

    return arrow_actor

def render_scene(lut:np.ndarray,
                 scanner_desc:dict,
                 show_arrows:bool=True,
                 image_fname:str=None,
                 maxval_frac:float=3,
                 image_color=(1.0,0.75,0.0),
                 background_color=(0,0,0),
                 crystal_color=(1,1,1)):
    colors = vtkNamedColors()

    # Create the scanner actor
    # Create points.
    points = vtkPoints()
    for i in range(lut.shape[0]):
        points.InsertNextPoint(lut[i,0],lut[i,1],lut[i,2])

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
    actor.GetProperty().SetColor(vtkColor3d(crystal_color))

    renderer = vtkRenderer()

    if(image_fname is not None):

        # Load volumetric data (e.g., from a .vtk or .nii file)
        reader = vtkNIFTIImageReader()
        reader.SetFileName(image_fname)
        reader.Update()

        nifti_matrix = reader.GetQFormMatrix()
        if nifti_matrix is None:
            nifti_matrix = reader.GetSFormMatrix()

        # Convert the affine matrix from NIFTI to a VTK matrix
        affine_matrix = vtkMatrix4x4()
        if nifti_matrix:
            affine_matrix.DeepCopy(nifti_matrix)
        # Flip X axis because of convention (Although not sure why it's not done automatically by VTK)
        affine_matrix.SetElement(0, 0, -affine_matrix.GetElement(0, 0))
        affine_matrix.SetElement(0, 1, -affine_matrix.GetElement(0, 1))
        affine_matrix.SetElement(0, 2, -affine_matrix.GetElement(0, 2))
        affine_matrix.SetElement(0, 3, -affine_matrix.GetElement(0, 3))

        # Get the max value of the image
        image_data = reader.GetOutput()
        scalars = image_data.GetPointData().GetScalars()
        np_scalars = np.array(scalars)
        max_val = np.max(np_scalars)

        # Create a volume mapper and specify how to map the data
        volume_mapper = vtkSmartVolumeMapper()
        volume_mapper.SetInputConnection(reader.GetOutputPort())

        # Set volume properties (e.g., opacity, shading)
        volume_property = vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        # Opacity transfer function (to control transparency)
        max_opacity = max_val/maxval_frac
        opacity_transfer_function = vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)  # Completely transparent for low values
        opacity_transfer_function.AddPoint(max_opacity, 1.0)  # Fully opaque for high values
        volume_property.SetScalarOpacity(opacity_transfer_function)

        # Color transfer function (to control color based on intensity)
        color_transfer_function = vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black for low intensity
        color_transfer_function.AddRGBPoint(max_val,
                                            image_color[0],
                                            image_color[1],
                                            image_color[2])  # Color for high intensity
        volume_property.SetColor(color_transfer_function)

        # Step 4: Create a volume actor and set its mapper and properties
        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        volume.SetUserMatrix(affine_matrix)  # Apply the NIFTI affine transformation

        renderer.AddVolume(volume)

    max_lut_x = np.max(lut[:,0])
    max_lut_y = np.max(lut[:,1])
    max_lut_z = np.max(lut[:,2])

    # Add X-Y-Z arrows
    if show_arrows:
        red = colors.GetColor3d("Red")
        green = colors.GetColor3d("Green")
        blue = colors.GetColor3d("Blue")
        arrow_length = ( 1 / 3 )*((max_lut_x + max_lut_y + max_lut_z) / 3) # X-Y-Z arrows are set to 1/3 the scanner radius
        x_arrow = create_arrow(red, (1, 0, 0), arrow_length)   # X-axis (red)
        y_arrow = create_arrow(green, (0, 1, 0), arrow_length) # Y-axis (green)
        z_arrow = create_arrow(blue, (0, 0, 1), arrow_length)  # Z-axis (blue)

    # Create a renderer, render window, and interactor.
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetWindowName('Rendering a scanner')

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    # Camera management
    camera = vtkCamera()
    camera.SetPosition(2*max_lut_x, 2*max_lut_y, 2*max_lut_z)
    camera.SetFocalPoint(0, 0, 0)

    # Add the actor to the scene.
    renderer.AddActor(actor)
    if show_arrows:
        renderer.AddActor(x_arrow)
        renderer.AddActor(y_arrow)
        renderer.AddActor(z_arrow)
    renderer.SetActiveCamera(camera)
    renderer.SetBackground(vtkColor3d(background_color[0],
                                      background_color[1],
                                      background_color[2]))

    # Render and interact.
    ren_win.Render()
    renderer.GetActiveCamera().Zoom(0.9)
    iren.Start()


class CalcGlyph(object):
    def __init__(self, glyph_filter, lut_masked, desc):
        self.glyph_filter = glyph_filter
        self.lut  = lut_masked
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

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Render a scene based on LUT, scanner description parameters and, optionally, an image.")

    # Add arguments for each of the parameters
    parser.add_argument('--lut', type=str, required=True,
                        help="Path to the LUT file (e.g., 'MYSCANNER.lut').")
    parser.add_argument('--det_mask', help='Path to detector mask array.')
    parser.add_argument('--cz', dest='cz', type=float,
                        help="Crystal size in the z direction.")
    parser.add_argument('--ctrans', dest='ctrans', type=float,
                        help="Crystal size in the transverse direction.")
    parser.add_argument('--cdepth', dest='cdepth', type=float, required=True,
                        help="Depth of the crystal.")
    parser.add_argument('--image', type=str, required=False,
                        help="Path to the NIFTI image file (e.g., 'my_image.nii').")
    parser.add_argument('--maxval_frac', type=float, required=False, default=3,
                        help="Fraction of the max value of the image to use for the display (Default: 3)")
    parser.add_argument('--image_color', type=str, required=False, default='1.0,0.75,0',
                        help="Color of the image (RGB) with values ranging from 0 to 255. Format: \"R,G,B\"")
    parser.add_argument('--crystal_color', type=str, required=False, default='1,1,1',
                        help="Color of the crystals (RGB) with values ranging from 0.0 to 1.0. Format: \"R,G,B\"")
    parser.add_argument('--background_color', type=str, required=False, default='0.0,0.0,0.0',
                        help="Color of the background (RGB) with values ranging from 0.0 to 1.0. Format: \"R,G,B\"")
    parser.add_argument('--hide_arrows', action='store_true',
                        help="Hide X-Y-Z arrows")

    # Parse the arguments
    args = parser.parse_args()

    # Read the LUT file
    lut = np.fromfile(args.lut, dtype=np.float32).reshape([-1,6])
    if args.det_mask is not None:
        mask = np.fromfile(args.det_mask, dtype=bool)
        if mask.ndim != 1 or mask.size != len(lut):
            raise RuntimeError(
                'Detector mask size does not match lookup table.')
        lut = lut[mask]

    # Create scanner description dictionary
    if args.cz is None and args.ctrans is not None:
        ctrans = args.ctrans
        cz = args.ctrans
    elif args.cz is not None and args.ctrans is None:
        ctrans = args.cz
        cz = args.cz
    elif args.cz is not None and args.ctrans is not None:
        cz = args.cz
        ctrans = args.ctrans
    else:
        raise ValueError("Need to specify crystal size")

    scanner_desc = {
        "crystalSize_z": cz,
        "crystalSize_trans": ctrans,
        "crystalDepth": args.cdepth
    }

    image_color = [float(s) for s in args.image_color.split(',')]
    background_color = [float(s) for s in args.background_color.split(',')]
    crystal_color = [float(s) for s in args.crystal_color.split(',')]

    # Call the render function with the provided arguments
    render_scene(lut, scanner_desc,
                 not args.hide_arrows,
                 args.image,
                 args.maxval_frac,
                 image_color,
                 background_color,
                 crystal_color)
