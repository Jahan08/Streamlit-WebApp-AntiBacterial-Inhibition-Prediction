import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.Draw import SimilarityMaps
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    with open("AntiBacterial_Inhibition_DNN_Regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    model.load_weights("best_weights.weights.h5")
    return model

model = load_trained_model()

# Function to convert SDF to DataFrame
def load_sdf_to_df(sdf_file):
    mols = [m for m in Chem.SDMolSupplier(sdf_file) if m is not None]  # Filter out invalid molecules
    df = pd.DataFrame({
        "Molecule": mols,
        "SMILES": [Chem.MolToSmiles(mol) for mol in mols]
    })
    return df

# Function to generate Morgan fingerprints
def mol2arr(mol, radius=10, nBits=2048):
    arr = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Function to generate similarity map
def plot_similarity_map(mol, model):
    d = Draw.MolDraw2DCairo(400, 400)
    SimilarityMaps.GetSimilarityMapForModel(
        mol,
        SimilarityMaps.GetMorganFingerprint,
        lambda x: model.predict(np.array([list(x)]))[0],
        draw2d=d
    )
    d.FinishDrawing()
    return d.GetDrawingText()

# Streamlit app
st.title("Broad-Spectrum Anti-Bacterial Inhibitors Predictor")
st.write("Predict pIC50 values for molecules and visualize similarity maps.")

# Sidebar for file upload or SMILES input
st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Choose input method:", ("Upload SDF File", "Enter SMILES"))

if input_option == "Upload SDF File":
    uploaded_file = st.sidebar.file_uploader("Upload an SDF file", type=["sdf"])
    if uploaded_file:
        # Load the uploaded SDF file
        df = load_sdf_to_df(uploaded_file)
        st.write("Uploaded Molecules:")
        st.write(df)

        # Predict pIC50 values
        st.write("Predicting pIC50 values...")
        df["Predicted pIC50"] = df["Molecule"].apply(lambda mol: model.predict(np.array([mol2arr(mol)]))[0][0])
        st.write(df[["SMILES", "Predicted pIC50"]])

        # Visualize similarity map for a selected molecule
        st.sidebar.header("Visualize Similarity Map")
        selected_index = st.sidebar.selectbox("Select a molecule", df.index)
        selected_mol = df.loc[selected_index, "Molecule"]

        st.write(f"Similarity Map for Molecule {selected_index}:")
        img_data = plot_similarity_map(selected_mol, model)
        st.image(Image.open(io.BytesIO(img_data)), use_column_width=True)

else:
    smiles_input = st.sidebar.text_input("Enter a SMILES string:")
    if smiles_input:
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is None:
                st.error("Invalid SMILES string. Please enter a valid SMILES.")
            else:
                st.write("Input Molecule:")
                st.image(Draw.MolToImage(mol), caption=f"SMILES: {smiles_input}", use_column_width=True)

                # Predict pIC50 value
                predicted_pic50 = model.predict(np.array([mol2arr(mol)]))[0][0]
                st.write(f"Predicted pIC50: {predicted_pic50:.2f}")

                # Visualize similarity map
                st.write("Similarity Map:")
                img_data = plot_similarity_map(mol, model)
                st.image(Image.open(io.BytesIO(img_data)), use_column_width=True)
        except Exception as e:
            st.error(f"Error processing SMILES: {e}")
