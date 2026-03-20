import io
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from PIL import Image
import matplotlib.colors as mcolors

def visualize_density(smiles: str, spin_population: list, uncertainty: list=None, show_image=True, rgb=[0, 64, 128]):

    #-----------Preprocessing-----------
    mol = Chem.MolFromSmiles(smiles)
    nh_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0]
    if len(nh_indices) > 0:
        mol = Chem.AddHs(mol, onlyOnAtoms=nh_indices)
    rdDepictor.Compute2DCoords(mol)
    num_heavy_atoms = len(spin_population)

    #-----------Map the properties to values-----------
    if uncertainty is not None:
        atom_colors = {
            i: uncertainty_to_alpha(uncertainty[i],alpha_max=0.8, uncertainty_factor=2, rgb=rgb)
            for i in range(num_heavy_atoms)
        }

    else:
        atom_colors = {
            i: spin_to_color(spin_population[i], vmin=np.min(spin_population), vmax=np.max(spin_population), rgb=rgb)
            for i in range(num_heavy_atoms)
        }

    atom_radii = {i: 1.5 * spin_population[i] for i in range(num_heavy_atoms)}

    # -----------Draw the molecule-----------
    drawer, opts = load_acs_style()
    draw_opts = {
        "highlightAtoms": list(range(num_heavy_atoms)),
        "highlightAtomColors": atom_colors,
        "highlightAtomRadii": atom_radii,
    }

    if uncertainty is not None:
        sum_spin_population_uncertainty = np.sum(uncertainty)
        binned_confidence = ['High' if sum_spin_population_uncertainty < 1.0 else 'Medium' if sum_spin_population_uncertainty < 1.5 else 'Low'][0]
        draw_opts["legend"] = f"{binned_confidence} confidence prediction\nSum uncertainty: {np.round(sum_spin_population_uncertainty, 4)}"

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, **draw_opts)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # -----------Plotting-----------
    if show_image:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        plt.imshow(img)
        plt.axis("off")
        r, g, b = np.array(rgb) / 255.0

        if uncertainty is not None:
            alpha_start = 0.8
            alpha_end = 0.0
        else:
            alpha_start = 0.0
            alpha_end = 0.8

        cdict = {
            'red': [(0.0, r, r), (1.0, r, r)],
            'green': [(0.0, g, g), (1.0, g, g)],
            'blue': [(0.0, b, b), (1.0, b, b)],
            'alpha': [(0.0, alpha_start, alpha_start), (1.0, alpha_end, alpha_end)]
        }

        cmap = mcolors.LinearSegmentedColormap('SpindensityAlpha', cdict)
        norm = mcolors.Normalize(vmin=np.min(spin_population), vmax=np.max(spin_population))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if uncertainty is not None:
            cbar.set_label('Uncertainty')
        else:
            cbar.set_label('Spin density')

        plt.tight_layout()
        plt.show()

    return img


def spin_to_color(value, vmin, vmax, rgb=[0, 64, 128]):
    '''
    Transforms spin value into color values for visualization.
    '''
    norm = (value - vmin) / (vmax - vmin)
    norm = max(0, min(0.8, norm))
    rgb_transformed = np.array(rgb)/255.0
    return_value = np.append(rgb_transformed, norm)
    return tuple(return_value)


def uncertainty_to_alpha(value, alpha_max=0.8, uncertainty_factor=2, rgb=[0, 64, 128]):
    '''
    Transforms uncertainty values into alpha values for visualization. Higher uncertainty results in lower alpha (more transparent).
    :param value: Value to transform
    :param alpha_max: Maximum alpha value to use
    :param uncertainty_factor: Factor to control how quickly alpha decreases with increasing uncertainty. Higher values will make alpha decrease more rapidly.
    :param rgb: Color scale
    :return: Transformed value
    '''
    alpha = alpha_max-(uncertainty_factor*value)
    rgb_transformed = np.array(rgb) / 255.0
    return_value = np.append(rgb_transformed, alpha)
    return tuple(return_value)

def load_acs_style():
    '''
    Draw ACS Sytle in RDKit
    '''
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()

    opts.useBWAtomPalette()
    opts.backgroundColour = (1, 1, 1)
    opts.bondLineWidth = 3.0
    opts.minFontSize = 32
    opts.maxFontSize = 42
    opts.padding = 0.05
    opts.addStereoAnnotation = False
    opts.explicitMethyl = False
    opts.addAtomIndices = False
    return drawer, opts
