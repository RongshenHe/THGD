import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt





class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
          
            atom_symbol = atom_decoder[int(node_list[i])]
            negative_charge = atom_symbol.find("-1")
            positive_charge = atom_symbol.find("+1")

            #if the atom is not neutrally charged:
            if(negative_charge != -1 or positive_charge != -1):
                #TODO: this may require a more robust handling if we
                #will add more infos in the atom_decoder strings
                formal_charge = int(atom_symbol[-2:])   #the formal charge are the last two characters
                atom_symbol = atom_symbol[:-2]          #the string before the last 2 characters is the actual atomic symbol
                                                        #(we MUST do this after getting formal_charge)
            else:
                formal_charge = 0

            a = Chem.Atom(atom_symbol)
            a.SetFormalCharge(formal_charge)

            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")


    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = [self.mol_from_graphs(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            Draw.MolToFile(mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}")
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(os.path.join(path, '{}_grid_image.png'.format(path.split('/')[-1])))
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols


class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix, augmented_node_list=None):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, 
                           number=i, 
                           symbol=node_list[i], 
                           ring_num= augmented_node_list[i] if augmented_node_list is not None else None,
                           color_val=node_list[i]
                           )

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        symbols = nx.get_node_attributes(graph, 'symbol')
        symbol_values = np.array(list(symbols.values()), dtype=np.float32)

        vmin, vmax = np.min(symbol_values), np.max(symbol_values)

        plt.figure()
        
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=True, labels=symbols, 
                node_color=symbol_values, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

        ring_nums = nx.get_node_attributes(graph, 'ring_num')  # 获取每个节点的 'symbol' 标签
        if ring_nums:
            for node, ring_num_value in ring_nums.items():
                if ring_num_value != 0:
                    x, y = pos[node]  # 获取节点的位置
                    plt.text(x, y, s=f"ring_num: {ring_num_value}", fontsize=8, color='black', 
                            horizontalalignment='center', verticalalignment='center', 
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.2'))

        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][2].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        try:
            imgs = [imageio.imread(fn) for fn in save_paths]
            gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
            imgs.extend([imgs[-1]] * 10)
            imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
            if wandb.run:
                wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
        except: print("upload wandb failed")
