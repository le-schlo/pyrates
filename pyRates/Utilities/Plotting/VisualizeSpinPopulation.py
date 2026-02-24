import io
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from PIL import Image


def visualize_density(smiles: str, spin_population: list, show_image=True, rgb=[0, 64, 128]):
    mol = Chem.MolFromSmiles(smiles)
    nh_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0]
    mol = Chem.AddHs(mol, onlyOnAtoms=nh_indices)
    rdDepictor.Compute2DCoords(mol)

    num_heavy_atoms = len(spin_population)

    atom_colors = {
        i: spin_to_color(spin_population[i], vmin=np.min(spin_population), vmax=np.max(spin_population), rgb=rgb)
        for i in range(num_heavy_atoms)
    }

    atom_radii = {i: 0.5 for i in range(num_heavy_atoms)}

    # -----------------------
    # Draw (ACS style)
    # -----------------------
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

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(num_heavy_atoms)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii
    )

    drawer.FinishDrawing()


    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return img


def spin_to_color(value, vmin, vmax, rgb=[0, 64, 128]):
    norm = (value - vmin) / (vmax - vmin)
    norm = max(0, min(1, norm))
    rgb_transformed = np.array(rgb)/255.0
    return_value = np.append(rgb_transformed, norm)
    return tuple(return_value)


