import streamlit as st

# Set page config for a clean, dedicated layout
st.set_page_config(layout="wide", page_title="MA UHI Dashboard - Sources")

st.title("Sources & Scientific Methodology")
st.subheader("Derivation of Urban Heat Island Coefficients")

st.markdown("""
This section documents the academic literature, empirical studies, and remote sensing frameworks 
used to establish the microclimate coefficients driving our Simulation Lab.
""")

st.divider()

# Massive text sections with ample space
st.markdown("### 1. Ambient Air Impact Coefficients (Per unit change)")
st.markdown("###### Air Impact = ((Impervious * $A_{IMP}$) - (Tree Canopy * $A_{TREE}$) - ([Albedo] - 0.15) * $A_{ALB}$)")
st.markdown("*With a 35°C baseline of 0% impervious, 0% tree canopy, 0.15 albedo. All temperatures are in °C.*")
st.markdown("""
    **Impervious surface | $A_{IMP}$ = 0.07/%** For each percent more concrete/asphalt in a given area, the simulation adds 0.07°C.  
    **Tree canopy | $A_{TREE}$ = 0.01/%** Tree canopy functions as the offset to higher temperatures, so the above formula **subtracts** tree impact from  total temp.  
    Green roofs | $A_{GROOF}$ = 0.04  
    Albedo | $A_{ALB}$ = 15.0  
    """)

st.markdown("""### 2. Human "Real Feel" Formula (Mean Radiant Temperature)""")
            
st.markdown("""### 3. Environmental Justice & Microclimate Vulnerability""")
st.write(
    "Document the studies regarding neighborhood-sized microclimates, historic redlining trends "
    "in Massachusetts, and the socio-economic disparities linked to surface albedo and tree cover distribution."
)
