import streamlit as st
import pandas as pd
import numpy as np
import io, base64
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem.TemplateAlign import AlignMolToTemplate2D

from pyvis.network import Network
import streamlit.components.v1 as components

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

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
    """Return a color based on the regression coefficient value."""
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
This app demonstrates two complementary workflows:

1. **Free-Wilson Analysis with Network Visualization:**  
   Decomposes molecules into a core and R groups using RDKit’s built-in tools, fits a Ridge regression to obtain coefficients for each R group, and visualizes the results as an interactive network.

2. **Molecule Networks Visualization:**  
   Computes molecular fingerprints and visualizes molecular similarity as an interactive network using PyVis.
""")

# Sidebar to select analysis type
analysis_type = st.sidebar.radio("Select Analysis", ["Free-Wilson Analysis", "Molecule Networks"])

# ========================================
# Free-Wilson Analysis Section
# ========================================
if analysis_type == "Free-Wilson Analysis":
    st.header("Free-Wilson Analysis Workflow")

    # --- 1. Data Source Selection ---
    with st.expander("1. Data Source Selection"):
        st.markdown("Select the dataset to use for the Free-Wilson analysis.")
        data_source = st.radio("Data Source:", ["Example Dataset", "Upload CSV"], key="fw_data_source")
        if data_source == "Upload CSV":
            uploaded_file_fw = st.file_uploader("Upload CSV for Free-Wilson Analysis", type=["csv"], key="fw_csv")
            if uploaded_file_fw is not None:
                fw_df = pd.read_csv(uploaded_file_fw)
            else:
                st.warning("No CSV uploaded. Defaulting to Example Dataset.")
                default_url = "https://raw.githubusercontent.com/PatWalters/practical-cheminformatics_tutorials/main/data/CHEMBL313_sel.smi"
                fw_df = pd.read_csv(default_url)
        else:
            default_url = "https://raw.githubusercontent.com/PatWalters/practical-cheminformatics_tutorials/main/data/CHEMBL313_sel.smi"
            fw_df = pd.read_csv(default_url)
        st.write("Dataset preview:")
        st.write(fw_df.head())

    # --- 2. Molecule Preparation and R-Group Decomposition ---
    with st.expander("2. Molecule Preparation and R-Group Decomposition"):
        st.markdown("**Converting SMILES to molecules, aligning them to a core, and performing R-group decomposition.**")
        # Convert SMILES to RDKit molecule objects
        fw_df['mol'] = fw_df.SMILES.apply(Chem.MolFromSmiles)
        st.write("Molecule conversion complete. Example:")
        st.write(fw_df[['SMILES', 'mol']].head())
        
        # Define the core molecule (adjust as needed)
        core_smiles = "c1ccc(C2CC3CCC(C2)N3)cc1"
        core_mol = Chem.MolFromSmiles(core_smiles)
        st.write("Core molecule SMILES:", core_smiles)
        
        # Align molecules to the core template (modifies molecules in-place)
        aligned_mols = []
        for mol in fw_df['mol']:
            if mol is not None:
                # Make a copy to preserve the original
                mol_copy = Chem.Mol(mol)
                AlignMolToTemplate2D(mol_copy, core_mol)
                aligned_mols.append(mol_copy)
            else:
                aligned_mols.append(None)
        fw_df['aligned_mol'] = aligned_mols
        
        # Perform R-group decomposition using RDKit's built-in function
        # Note: RGroupDecompose returns a tuple (rgroups, unmatched)
        rgroups, unmatched = RGroupDecompose(core_mol, list(fw_df['aligned_mol']), asSmiles=True)
        # Convert the dictionary of R groups to a DataFrame
        rgroup_df = pd.DataFrame.from_dict(rgroups, orient='index')
        st.write("R-group decomposition result (first 5 rows):")
        st.write(rgroup_df.head())
        
        # Construct core DataFrame for later visualization (unique core images)
        core_df = pd.DataFrame({"mol": [core_mol]})
        
        # Get unique R groups from each column except the 'Core'
        unique_list = []
        st.write("Unique R groups summary:")
        for r in rgroup_df.columns:
            if r == "Core":
                continue
            num_rgroups = len(rgroup_df[r].unique())
            st.write(f"- **{r}**: {num_rgroups} unique groups")
            unique_list.append(list(rgroup_df[r].unique()))
            
    # --- 3. Regression Model Preparation and Training ---
    with st.expander("3. Regression Model Preparation and Training"):
        st.markdown("**Featurizing R groups and performing Ridge regression.**")
        # One-hot encode the R groups using the unique categories obtained (for all columns except 'Core')
        if len(unique_list) == 0:
            st.error("No R groups found. Check your input data and core molecule definition.")
        else:
            enc = OneHotEncoder(categories=unique_list, sparse_output=False)
            # Extract only the R-group columns (skip the 'Core' column)
            rgroup_columns = [col for col in rgroup_df.columns if col != "Core"]
            one_hot_mat = enc.fit_transform(rgroup_df[rgroup_columns].values)
            ridge = Ridge()
            # Ensure that the dataset contains the target column 'pIC50'
            if "pIC50" not in fw_df.columns:
                st.error("The dataset does not contain a 'pIC50' column. Please check your dataset.")
            else:
                ridge.fit(one_hot_mat, fw_df.pIC50)
                st.success("Ridge regression model trained successfully!")
                st.write("Model Intercept:", ridge.intercept_)
                # Extract regression coefficients for each R-group category
                rg_df_dict = {}
                start = 0
                for col, cat in zip(rgroup_columns, enc.categories_):
                    coef_list = ridge.coef_[start:start+len(cat)]
                    start += len(cat)
                    # Build a DataFrame for each R-group column with its fragments and coefficients
                    rg_df = pd.DataFrame({"smiles": cat, "coef": coef_list})
                    # For visualization, generate molecule objects from SMILES and sort by coefficient
                    rg_df["mol"] = rg_df.smiles.apply(Chem.MolFromSmiles)
                    rg_df.sort_values("coef", inplace=True)
                    rg_df_dict[col] = rg_df
                st.write("Regression coefficients for R groups:")
                for key, value in rg_df_dict.items():
                    st.write(f"**R group {key}:**")
                    st.write(value)
                
    # --- 4. Network Visualization ---
    with st.expander("4. Network Visualization"):
        st.markdown("**Building an interactive network of Free-Wilson coefficients.**")
        net = Network(height="1000px", width=1900, notebook=False)
        # Add the core molecule node with its image
        core_image_base64 = generate_molecule_image(core_df.iloc[0]["mol"])
        net.add_node("Core", label="Core", title="Core", shape="circularImage",
                     image=f"data:image/png;base64,{core_image_base64}", borderWidth=3, color="grey")
        # For each R-group type, add nodes and edges
        for r in rg_df_dict:
            # Add an R-group "vector" node
            net.add_node(r, label=r, title=r, color="grey")
            net.add_edge("Core", r, width=3, color="grey", label=r)
            # Add nodes for each individual R-group fragment with its regression coefficient
            for index, row in rg_df_dict[r].iterrows():
                node_id = f"{row['smiles']}"
                score = row['coef']
                score_string = str(round(score, 3))
                color = get_color(score)
                mol_image = generate_molecule_image(row["mol"])
                net.add_node(node_id,
                             label=score_string,
                             title=node_id,
                             shape="circularImage",
                             image=f"data:image/png;base64,{mol_image}",
                             borderWidth=3,
                             color=color)
                net.add_edge(r, node_id, width=3, color=color)
        net.save_graph("free_wilson_coefficient_network.html")
        with open("free_wilson_coefficient_network.html", 'r', encoding='utf-8') as f:
            html_str = f.read()
        components.html(html_str, height=1000, width=1900)

# ========================================
# Molecule Networks Visualization Section
# ========================================
elif analysis_type == "Molecule Networks":
    st.header("Molecule Networks Visualization")
    st.markdown("""
    **Concept:**  
    Compute the Morgan fingerprint for each molecule and calculate pairwise Tanimoto similarities.
    An edge is drawn between two molecules if their similarity exceeds a user-defined threshold.
    The resulting network is rendered interactively using PyVis.
    """)
    # File uploader for molecule network (if not using the same uploaded file)
    uploaded_file_mn = st.file_uploader("Upload your CSV file for Molecule Networks", type=["csv"], key="mn_csv")
    if uploaded_file_mn is not None:
        df = pd.read_csv(uploaded_file_mn)
    else:
        st.error("Please upload a CSV file with at least 'SMILES' and 'Potency' columns for Molecule Networks.")
        st.stop()

    # Compute fingerprints for each molecule
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

    # Similarity threshold slider
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    edges = []
    fps = df['Fingerprint'].tolist()

    st.info("Calculating pairwise similarities... This may take a moment for large datasets.")
    # Compute pairwise Tanimoto similarity (naive O(n²) approach)
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim > threshold:
                edges.append((i, j, sim))
    st.write(f"Number of edges in the network: {len(edges)}")

    # Build PyVis network graph
    nt = Network(height='600px', width=1000, notebook=False)
    for idx, row in df.iterrows():
        label = row['Name'] if 'Name' in df.columns and pd.notnull(row['Name']) else str(row['SMILES'])
        nt.add_node(idx, label=label, title=f"SMILES: {row['SMILES']}<br>Potency: {row['Potency']}")
    for edge in edges:
        nt.add_edge(edge[0], edge[1], value=edge[2])
    nt.show_buttons(filter_=['physics'])
    nt.save_graph("molecule_network.html")
    with open("molecule_network.html", 'r', encoding='utf-8') as HtmlFile:
        source_code = HtmlFile.read()
    components.html(source_code, height=650, width=1000)

