import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Other imports (for Molecule Networks) are retained but not modified here.
from rdkit.Chem import AllChem, DataStructs
from pyvis.network import Network
import streamlit.components.v1 as components

st.title("Molecular Data Analysis: Free-Wilson & Molecule Networks")

st.markdown("""
This app demonstrates a workflow combining two analyses:
- **Free-Wilson Analysis:** We extract molecular scaffolds from SMILES strings, encode them as categorical variables, and fit a linear regression model to understand scaffold contributions to potency.
- **Molecule Networks:** We compute molecular fingerprints and build an interactive network visualization using PyVis.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())
    except Exception as e:
        st.error("Error reading CSV file. Please check the file format.")

    # Check for required columns
    required_columns = ['SMILES', 'Potency']
    if not all(col in df.columns for col in required_columns):
        st.error("CSV file must contain at least the 'SMILES' and 'Potency' columns.")
    else:
        # Sidebar analysis selection
        analysis_type = st.sidebar.radio("Select Analysis", ["Free-Wilson Analysis", "Molecule Networks"])

        # =====================================================
        # Free-Wilson Analysis Section
        # =====================================================
        if analysis_type == "Free-Wilson Analysis":
            st.header("Free-Wilson Analysis Workflow")

            # 1. Data Loading Section (already loaded above)
            with st.expander("1. Data Loading"):
                st.write("Uploaded data preview:")
                st.write(df.head())

            # 2. Free-Wilson Analysis: Scaffold Extraction
            with st.expander("2. Scaffold Extraction"):
                st.write("Extracting molecular scaffolds from SMILES strings...")
                def get_scaffold(smiles):
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            return None
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold is not None else None
                        return scaffold_smiles
                    except Exception as e:
                        return None

                df['Scaffold'] = df['SMILES'].apply(get_scaffold)
                st.write("Data with extracted scaffolds (first 5 rows):", df[['SMILES', 'Scaffold']].head())

            # 3. Model Preparation
            with st.expander("3. Model Preparation"):
                st.write("Encoding the extracted scaffolds as dummy variables (one-hot encoding).")
                X = pd.get_dummies(df['Scaffold'], prefix="scaf")
                y = df['Potency']
                st.write("Feature matrix preview:")
                st.write(X.head())
                st.write("Target variable preview:")
                st.write(y.head())

            # 4. Model Training
            with st.expander("4. Model Training"):
                st.write("Training a linear regression model using the one-hot encoded scaffold features...")
                model = LinearRegression()
                model.fit(X, y)
                st.success("Model trained successfully!")
                st.write("Intercept:", model.intercept_)

            # 5. Model Result Display
            with st.expander("5. Model Results"):
                st.write("Displaying regression coefficients for each scaffold feature:")
                coeff = model.coef_
                coeff_df = pd.DataFrame({'Scaffold': X.columns, 'Coefficient': coeff})
                coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)
                st.write(coeff_df)

                # Plot the coefficients
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(coeff_df['Scaffold'], coeff_df['Coefficient'])
                ax.set_xlabel("Scaffold")
                ax.set_ylabel("Coefficient")
                ax.set_title("Scaffold Contribution (Free-Wilson Analysis)")
                plt.xticks(rotation=90)
                st.pyplot(fig)

        # =====================================================
        # Molecule Networks Section
        # =====================================================
        elif analysis_type == "Molecule Networks":
            st.header("Molecule Networks Visualization")
            st.markdown("""
            **Concept:**  
            We compute the Morgan fingerprint for each molecule and calculate pairwise Tanimoto similarities.
            An edge is drawn between two molecules if their similarity exceeds a user-defined threshold.
            The resulting network is rendered interactively using PyVis.
            """)

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
            nt = Network(height='600px', width='100%', notebook=False)
            for idx, row in df.iterrows():
                label = row['Name'] if 'Name' in df.columns and pd.notnull(row['Name']) else str(row['SMILES'])
                nt.add_node(idx, label=label, title=f"SMILES: {row['SMILES']}<br>Potency: {row['Potency']}")
            for edge in edges:
                nt.add_edge(edge[0], edge[1], value=edge[2])

            nt.show_buttons(filter_=['physics'])
            nt.save_graph("molecule_network.html")

            # Embed the network graph in Streamlit with a numeric width value
            with open("molecule_network.html", 'r', encoding='utf-8') as HtmlFile:
                source_code = HtmlFile.read()
            components.html(source_code, height=650, width=1000)

