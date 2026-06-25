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
st.write(
    "$A_{IMP}$ = 0.07 Impervious surface"
    "$A_{TREE}$ = 0.05 Tree canopy"
    "$A_{GROOF}$ = 0.04 Green roofs"
    "$A_{ALB}$ = 15.0 Albedo"
    Long-form analysis text goes here. You can reference specific urban forestry papers, "
    "thermal imaging metrics, and ambient temperature modeling methodologies without "
    "worrying about crowding out your map interfaces."
)

st.markdown("### 2. Environmental Justice & Microclimate Vulnerability")
st.write(
    "Document the studies regarding neighborhood-sized microclimates, historic redlining trends "
    "in Massachusetts, and the socio-economic disparities linked to surface albedo and tree cover distribution."
)
