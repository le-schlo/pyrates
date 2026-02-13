import pandas as pd
from pathlib import Path
import ast


class DataLoader:
    def __init__(self,
        datapath: Path
    ):
        '''
        Provide a path to the dataset CSV file.
        The DataLoader will read the data and compute the target columns based on the specified input columns.
        The computed target columns will be added to the DataFrame and can be accessed using the show_columns() method.
        '''
        self.input_data = pd.read_csv(datapath).copy()
        self.column_names = self.input_data.columns.tolist()
        self.target_columns = None

    def show_columns(self):
        print("Available columns in the dataset:")
        for col in self.column_names:
            print(f"- {col}")

    def return_entire_df(self):
        return self.input_data

    def return_compact_data(self,
                            idx_column:str = 'idx',
                            smiles_column: str = 'smiles'
    ) -> pd.DataFrame:
        relevant_columns = [idx_column, smiles_column] + [col for col in self.target_columns if col not in [smiles_column]]
        return self.input_data[relevant_columns]

    def compute_targets(self,
        gibbs_singlet_column: str = 'Gibbs_S',
        gibbs_triplet_column: str = 'Gibbs_T',
        energy_singlet_singlet_column: str = 'E_S_geometry_S_energy',
        energy_triplet_triplet_column: str = 'E_T_geometry_T_energy',
        energy_triplet_singlet_column: str = 'E_T_geometry_S_energy',
        energy_singlet_triplet_column: str = 'E_S_geometry_T_energy',
        target_columns: list = ['adiabatic_energy', 'lambda_acceptor_singlet', 'lambda_donor_triplet'],
        return_compact: bool = True,
        Eh_to_kcal: float = 627.50962
    ) -> pd.DataFrame:
        '''

        :param Eh_to_kcal: Conversion factor from Hartree to kcal/mol. Default is 627.50962.
        :param gibbs_singlet_column: Name of the column containing Gibbs free energy for the singlet state (Gibbs_S).
        :param gibbs_triplet_column: Name of the column containing Gibbs free energy for the triplet state (Gibbs_T).
        :param energy_singlet_singlet_column: Name of the column containing energy for the singlet geometry in the singlet state (E_S_geometry_S_energy).
        :param energy_triplet_triplet_column: Name of the column containing energy for the triplet geometry in the triplet state (E_T_geometry_T_energy).
        :param energy_triplet_singlet_column: Name of the column containing energy for the triplet geometry in the singlet state (E_T_geometry_S_energy).
        :param energy_singlet_triplet_column: Name of the column containing energy for the singlet geometry in the triplet state (E_S_geometry_T_energy).
        :param target_columns: List of target columns to compute. Supported values are: adiabatic_energy, lambda_acceptor_singlet, lambda_donor_triplet, vertical_excitation, vertical_emission.
        :return: DataFrame with the computed target columns added.
        '''

        self.target_columns = target_columns
        # Extract numpy arrays
        gibbs_s = self.input_data[gibbs_singlet_column].to_numpy()
        gibbs_t = self.input_data[gibbs_triplet_column].to_numpy()
        es = self.input_data[energy_singlet_singlet_column].to_numpy()
        et = self.input_data[energy_triplet_triplet_column].to_numpy()

        # Compute target columns
        for col in target_columns:
            if col == 'spin_population' and col in self.input_data.columns:
                self.input_data[col] = self.input_data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                continue

            if col == 'adiabatic_energy':
                # Adiabatic energy: ΔE_adiabatic = (Gibbs_T + E_T_geometry_T) - (Gibbs_S + E_S_geometry_S)
                self.input_data['adiabatic_energy'] = ((gibbs_t + et) - (gibbs_s + es)) * Eh_to_kcal

            elif col == 'lambda_acceptor_singlet':
                # Lambda for acceptor going from singlet to triplet: λ_acceptor_singlet = E_T_geometry_S - E_S_geometry_S (corresponds to forward EnT process)
                self.input_data['lambda_acceptor_singlet'] = (self.input_data[energy_triplet_singlet_column] - self.input_data[energy_singlet_singlet_column]) * Eh_to_kcal

            elif col == 'lambda_donor_triplet':
                # Lambda for donor from triplet to singlet: λ_donor_triplet = E_S_geometry_T - E_T_geometry_T (corresponds to back EnT process)
                self.input_data['lambda_donor_triplet'] = (self.input_data[energy_singlet_triplet_column] - self.input_data[energy_triplet_triplet_column]) * Eh_to_kcal

            elif col == 'vertical_excitation':
                # Vertical excitation energy: ΔE_vertical_excitation = E_S_geometry_T - E_S_geometry_S (corresponds to vertical excitation from S0 to T1)
                self.input_data['vertical_excitation'] = (self.input_data[energy_singlet_triplet_column] - self.input_data[energy_singlet_singlet_column]) * Eh_to_kcal

            elif col == 'vertical_emission':
                # Vertical emission energy: ΔE_vertical_emission = E_T_geometry_T - E_T_geometry_S (corresponds to vertical emission from T1 to S0)
                self.input_data['vertical_emission'] = (self.input_data[energy_triplet_triplet_column] - self.input_data[energy_triplet_singlet_column]) * Eh_to_kcal

            else:
                raise ValueError(f"Unknown target column: {col}")

        if return_compact:
            return self.return_compact_data()
        return self.input_data
