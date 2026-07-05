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
st.markdown("###### Air Impact = ((Impervious % * $A_{IMP}$) - (Tree Canopy % * $A_{TREE}$) - ([Albedo] - 0.15) * $A_{ALB}$)")
st.markdown("*With a 35°C baseline of 0% impervious, 0% tree canopy, 0.15 albedo. All temperatures are in °C.*")
st.markdown("""**Impervious surface | $A_{IMP}$ = 0.0075**  
    For each percent additional concrete/asphalt in a given area, the simulation adds 0.0075°C.
    """)
st.markdown(
    "Sources: "
    "[Resilient Cambridge Urban Heat Island Report](https://www.cambridgema.gov/-/media/files/cdd/climate/resilientcambridge/urbanheatislandtechnicalreport.pdf?), "
    "[nature.com Heat Exposure Reductions in Boston](https://www.nature.com/articles/s43247-025-02462-3)"
)
st.markdown("*This Cambridge study identifies a -0.056°C/-1% impervious surface cooling rate <u>in an isolated region</u>. This impact was corroborated with the nature.com study and scaled down accordingly to represent the air temperature of the <u>whole</u> city. While impervious surface might make a small region warm, air movement dilutes that heating to be spread across the whole city.*", unsafe_allow_html=True)
st.write("")
st.markdown("""**Tree canopy | $A_{TREE}$ = 0.01**  
    Tree canopy functions as the offset to higher temperatures, so the above formula **subtracts** tree impact from total temp.
    """)
st.write("")
st.markdown("""**Green roofs | $A_{GROOF}$ = 0.003**  
    Green roofs introduce cooling on the rooftop level. While they are powerful with evapotranspiration and more reflective leaves, their soil traps and releases heat into the night.
    """)
st.write("")
st.markdown("""**Albedo | $A_{ALB}$ = 6.1**  
    Albedo is measured on a scale of 0 to 1, or 0% to 100% of light absorbed. This coefficient highlights the 6.1°C range across the albedo scale.  
    """)
st.markdown("""### 2. Human "Real Feel" Formula (Mean Radiant Temperature)""")

st.markdown("""**Impervious surface | $MRT_{IMP}$ = 0.12**""")
st.markdown(
    "Source: "
    "[Resilient Cambridge Urban Heat Island Report](https://www.cambridgema.gov/-/media/files/cdd/climate/resilientcambridge/urbanheatislandtechnicalreport.pdf?)"
)
st.markdown("*Identifies the relationship between tree canopy and impervious surface that removing impervious surface has ~83% the cooling impact as adding an equal amount of tree canopy.*")
st.markdown("""**Tree canopy | $MRT_{TREE}$ = 0.15**  
    Tree canopy functions as the offset to higher temperatures, so the above formula **subtracts** tree impact from total temp.
    """)
st.write("")
st.markdown("""**Green roofs | $MRT_{GROOF}$ = 0.002**  
    Green roofs introduce cooling on the rooftop level. While they are powerful with evapotranspiration and more reflective leaves, their soil traps and releases heat into the night.
    """)
st.write("")
st.markdown("""**Albedo | $MRT_{ALB}$ = 5.0**  
    Albedo is measured on a scale of 0 to 1, or 0% to 100% of light absorbed. This coefficient highlights the 6.1°C range across the albedo scale.  
    """)
st.markdown("""### 3. Environmental Justice & Microclimate Vulnerability""")
st.write(
    "Document the studies regarding neighborhood-sized microclimates, historic redlining trends "
    "in Massachusetts, and the socio-economic disparities linked to surface albedo and tree cover distribution."
)
