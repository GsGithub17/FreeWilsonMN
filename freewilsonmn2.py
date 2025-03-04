import streamlit as st
import pandas as pd
import numpy as np
import io, base64
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem.TemplateAlign import AlignMolToTemplate2D

from pyvis.network import Network
import streamlit.components.v1 as components

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

# For model prediction visualization
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# For displaying molecule grids
import mols2grid

# ---------------------------
# Helper Functions
# ---------------------------
def generate_molecule_image(mol, size=(300,300)):
    """Generate a base64-encoded PNG image of the molecule."""
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_color(score):
    """Return a discrete color for a regression coefficient:
       green if positive, red if negative, grey otherwise."""
    if score > 0:
        return "green"
    elif score < 0:
        return "red"
    else:
        return "grey"

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("Molecular Data Analysis: Free-Wilson & Molecule Networks")

st.markdown("""
This app demonstrates an integrated workflow for Free‑Wilson analysis, R‑group decomposition, and molecular networking.
It includes:
- **Display of the core(s)** found by the R‑group decomposition.
- **Model predictions visualization:** A Ridge regression model is trained using one‑hot encoded R‑groups and its predictions are compared to experimental pIC50 values.
- Interactive network visualizations of both regression coefficients and original molecules.
""")

# Sidebar: select analysis type
analysis_type = st.sidebar.radio("Select Analysis", ["Free-Wilson Analysis", "Molecule Networks"])

# ========================================
# Free-Wilson Analysis Section
# ========================================
if analysis_type == "Free-Wilson Analysis":
    st.header("Free-Wilson Analysis Workflow")

    # --- 1. Data Source Selection ---
    with st.expander("1. Data Source Selection"):
        st.markdown("Select the dataset to use for the Free‑Wilson analysis.")
        data_source = st.radio("Data Source:", ["Example Dataset", "Upload CSV"], key="fw_data_source")
        if data_source == "Upload CSV":
            uploaded_file_fw = st.file_uploader("Upload CSV for Free‑Wilson Analysis", type=["csv"], key="fw_csv")
            if uploaded_file_fw is not None:
                fw_df = pd.read_csv(uploaded_file_fw)
            else:
                st.warning("No CSV uploaded. Using example dataset instead.")
                data_source = "Example Dataset"
        if data_source == "Example Dataset":
            try:
                fw_df = pd.read_csv("data/CHEMBL313_sel.smi")
            except Exception as e:
                st.error(f"Failed to load local dataset from data/CHEMBL313_sel.smi: {e}")
                st.stop()
        st.write("Dataset preview:")
        st.write(fw_df[['SMILES']].head())

    # --- 2. Molecule Preparation and R-Group Decomposition ---
    with st.expander("2. Molecule Preparation and R-Group Decomposition"):
        st.markdown("**Convert SMILES to molecules, align them to a core, and perform R‑group decomposition.**")
        # Convert SMILES to RDKit molecules
        fw_df['mol'] = fw_df.SMILES.apply(Chem.MolFromSmiles)
        st.write("Molecule conversion complete. (Displaying SMILES only)")
        st.write(fw_df[['SMILES']].head())
        
        # Define the core molecule
        core_smiles = "c1ccc(C2CC3CCC(C2)N3)cc1"
        core_mol = Chem.MolFromSmiles(core_smiles)
        if core_mol is None:
            st.error("Error: Core molecule could not be parsed.")
            st.stop()
        if core_mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(core_mol)
        st.write("Core molecule SMILES:", core_smiles)
        
        # Align molecules to the core template
        aligned_mols = []
        for mol in fw_df['mol']:
            if mol is not None:
                mol_copy = Chem.Mol(mol)
                if mol_copy.GetNumConformers() == 0:
                    AllChem.Compute2DCoords(mol_copy)
                try:
                    AlignMolToTemplate2D(mol_copy, core_mol)
                except Exception as e:
                    st.warning(f"Alignment failed for one molecule: {e}")
                aligned_mols.append(mol_copy)
            else:
                aligned_mols.append(None)
        fw_df['aligned_mol'] = aligned_mols
        
        # Perform R-group decomposition
        rgroups, unmatched = RGroupDecompose(core_mol, list(fw_df['aligned_mol']), asSmiles=True)
        rgroup_df = pd.DataFrame(rgroups)
        st.write("R‑group decomposition result (first 5 rows):")
        st.write(rgroup_df.drop(columns=["Core"], errors="ignore").head())
        
        # Get unique R groups from each column (excluding "Core")
        unique_list = []
        st.write("Unique R groups summary:")
        for r in rgroup_df.columns:
            if r == "Core":
                continue
            num_rgroups = len(rgroup_df[r].unique())
            st.write(f"- **{r}**: {num_rgroups} unique groups")
            unique_list.append(list(rgroup_df[r].unique()))
            
    # --- 3. Display Core(s) Found by R-group Decomposition ---
    with st.expander("Display Core(s) Found by R-group Decomposition"):
        core_df = pd.DataFrame({"mol": [Chem.MolFromSmiles(x) for x in rgroup_df.Core.unique()]})
        mols2grid.display(core_df, size=(300,200), mol_col="mol")
        
    # --- 4. Regression Model Preparation and Training ---
    with st.expander("Regression Model Preparation and Training"):
        st.markdown("**Encode R-groups and perform Ridge regression.**")
        if len(unique_list) == 0:
            st.error("No R groups found. Check your input data and core molecule definition.")
            st.stop()
        else:
            enc = OneHotEncoder(categories=unique_list, sparse_output=False)
            rgroup_columns = [col for col in rgroup_df.columns if col != "Core"]
            one_hot_mat = enc.fit_transform(rgroup_df[rgroup_columns].values)
            ridge = Ridge()
            if "pIC50" not in fw_df.columns:
                st.error("The dataset does not contain a 'pIC50' column. Please check your dataset.")
                st.stop()
            else:
                ridge.fit(one_hot_mat, fw_df.pIC50)
                st.success("Ridge regression model trained successfully!")
                st.write("Model Intercept:", ridge.intercept_)
                rg_df_dict = {}
                start = 0
                for col, cat in zip(rgroup_columns, enc.categories_):
                    coef_list = ridge.coef_[start:start+len(cat)]
                    start += len(cat)
                    rg_df = pd.DataFrame({"smiles": cat, "coef": coef_list})
                    rg_df.sort_values("coef", inplace=True)
                    rg_df_dict[col] = rg_df
                st.write("Regression coefficients for R groups:")
                for key, value in rg_df_dict.items():
                    st.write(f"**R group {key}:**")
                    st.write(value[['smiles','coef']])
                    
    # --- 5. Visualize Model Predictions ---
    with st.expander("Visualize Model Predictions"):
        st.markdown("**Train-test split, model fitting, and prediction visualization.**")
        # Re-encode R-groups (using all columns except the first)
        enc2 = OneHotEncoder(categories=unique_list, sparse_output=False)
        one_hot_mat2 = enc2.fit_transform(rgroup_df.values[:,1:])
        # Split the data into training and test sets
        train_X, test_X, train_y, test_y = train_test_split(one_hot_mat2, fw_df.pIC50, test_size=0.2, random_state=42)
        # Define and train a Ridge regression model
        ridge2 = Ridge()
        ridge2.fit(train_X, train_y)
        pred = ridge2.predict(test_X)
        # Visualize model performance using seaborn
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.set_style('whitegrid')
        sns.set_context('talk')
        res_df = pd.DataFrame({'Exp_pIC50': test_y, 'Pred_pIC50': pred})
        r2 = r2_score(test_y, pred)
        fgrid = sns.lmplot(x='Exp_pIC50', y='Pred_pIC50', data=res_df)
        ax = fgrid.axes[0, 0]
        ax.text(6, 9, f"$R^2$={r2:.2f}")
        st.pyplot(fgrid.fig)
                    
    # --- 6. Network Visualization (Regression Coefficients) ---
    with st.expander("Network Visualization (Regression Coefficients)"):
        st.markdown("**Interactive network of Free-Wilson coefficients:**")
        core_df2 = pd.DataFrame({"mol": [core_mol]})
        net = Network(height="1000px", width=1900, notebook=False)
        core_image_base64 = generate_molecule_image(core_df2.iloc[0]["mol"])
        net.add_node("Core", label="Core", title="Core", shape="circularImage",
                     image=f"data:image/png;base64,{core_image_base64}", borderWidth=3, color="grey")
        for r in rg_df_dict:
            net.add_node(r, label=r, title=r, color="grey")
            net.add_edge("Core", r, width=3, color="grey", label=r)
            for index, row in rg_df_dict[r].iterrows():
                node_id = f"{row['smiles']}"
                score = row['coef']
                score_string = str(round(score, 3))
                color = "green" if score > 0 else "red" if score < 0 else "grey"
                mol_img = generate_molecule_image(Chem.MolFromSmiles(row["smiles"]))
                net.add_node(node_id,
                             label=score_string,
                             title=node_id,
                             shape="circularImage",
                             image=f"data:image/png;base64,{mol_img}",
                             borderWidth=3,
                             color=color)
                net.add_edge(r, node_id, width=3, color=color)
        net.save_graph("free_wilson_coefficient_network.html")
        with open("free_wilson_coefficient_network.html", 'r', encoding='utf-8') as f:
            html_str = f.read()
        components.html(html_str, height=1000, width=1900)
    
    # --- 7. Additional Network Visualization (Molecules by pIC50) ---
    with st.expander("Additional Network Visualization (Molecules by pIC50)"):
        st.markdown("""
        **Interactive network of original molecules:**  
        Nodes represent individual molecules from the dataset and are colored with a continuous colormap based on their pIC50 values.
        """)
        def get_fingerprint(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                return fp
            except Exception as e:
                return None

        fw_df['Fingerprint'] = fw_df.SMILES.apply(get_fingerprint)
        fw_df = fw_df[fw_df['Fingerprint'].notnull()].reset_index(drop=True)
        
        molecule_threshold = st.slider("Molecule Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="mol_thresh")
        mol_edges = []
        fps = fw_df['Fingerprint'].tolist()
        st.info("Calculating pairwise similarities for molecules... This may take a moment.")
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim > molecule_threshold:
                    mol_edges.append((i, j, sim))
        st.write(f"Number of edges in the molecule network: {len(mol_edges)}")
        
        cmap_viridis = mpl.colormaps["viridis"]
        norm = mcolors.Normalize(vmin=fw_df['pIC50'].min(), vmax=fw_df['pIC50'].max())
        
        mol_net = Network(height='600px', width=1000, notebook=False)
        for idx, row in fw_df.iterrows():
            label = row['Name'] if 'Name' in fw_df.columns and pd.notnull(row['Name']) else str(row['SMILES'])
            pIC50_val = row['pIC50']
            rgba = cmap_viridis(norm(pIC50_val))
            hex_color = mcolors.to_hex(rgba)
            mol_net.add_node(idx, label=label,
                             title=f"SMILES: {row['SMILES']}<br>pIC50: {pIC50_val}",
                             color=hex_color, shape="circularImage")
        for edge in mol_edges:
            mol_net.add_edge(edge[0], edge[1], value=edge[2])
        mol_net.show_buttons(filter_=['physics'])
        mol_net.save_graph("molecule_pIC50_network.html")
        
        fig, ax = plt.subplots(figsize=(4, 0.5))
        fig.subplots_adjust(bottom=0.5)
        smap_obj = cm.ScalarMappable(norm=norm, cmap=cmap_viridis)
        smap_obj.set_array([])
        cbar = plt.colorbar(smap_obj, cax=ax, orientation='horizontal')
        cbar.set_label('pIC50')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.image(buf, caption="pIC50 Color Bar", use_column_width=False)
        
        with open("molecule_pIC50_network.html", 'r', encoding='utf-8') as f:
            html_str2 = f.read()
        components.html(html_str2, height=650, width=1000)

# ========================================
# Molecule Networks Visualization Section
# ========================================
elif analysis_type == "Molecule Networks":
    st.header("Molecule Networks Visualization")
    st.markdown("""
    **Concept:**  
    Compute the Morgan fingerprint for each molecule and calculate pairwise Tanimoto similarities.
    An edge is drawn between two molecules if their similarity exceeds a user-defined threshold.
    The resulting network is rendered interactively using PyVis, with nodes colored by pIC50.
    """)
    uploaded_file_mn = st.file_uploader("Upload your CSV file for Molecule Networks", type=["csv"], key="mn_csv")
    if uploaded_file_mn is not None:
        df = pd.read_csv(uploaded_file_mn)
    else:
        st.error("Please upload a CSV file with at least 'SMILES' and 'pIC50' columns for Molecule Networks.")
        st.stop()

    def get_fingerprint(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return fp
        except Exception as e:
            return None

    df['Fingerprint'] = df['SMILES'].apply(get_fingerprint)
    df = df[df['Fingerprint'].notnull()].reset_index(drop=True)

    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    edges = []
    fps = df['Fingerprint'].tolist()

    st.info("Calculating pairwise similarities... This may take a moment for large datasets.")
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim > threshold:
                edges.append((i, j, sim))
    st.write(f"Number of edges in the network: {len(edges)}")

    norm = mcolors.Normalize(vmin=df['pIC50'].min(), vmax=df['pIC50'].max())
    cmap_viridis = mpl.colormaps["viridis"]

    nt = Network(height='600px', width=1000, notebook=False)
    for idx, row in df.iterrows():
        label = row['Name'] if 'Name' in df.columns and pd.notnull(row['Name']) else str(row['SMILES'])
        pIC50_val = row['pIC50']
        rgba = cmap_viridis(norm(pIC50_val))
        hex_color = mcolors.to_hex(rgba)
        nt.add_node(idx, label=label, 
                    title=f"SMILES: {row['SMILES']}<br>pIC50: {pIC50_val}",
                    color=hex_color, shape="circularImage")
    for edge in edges:
        nt.add_edge(edge[0], edge[1], value=edge[2])
    nt.show_buttons(filter_=['physics'])
    nt.save_graph("molecule_network.html")
    with open("molecule_network.html", 'r', encoding='utf-8') as HtmlFile:
        html_str = HtmlFile.read()
    components.html(html_str, height=650, width=1000)

