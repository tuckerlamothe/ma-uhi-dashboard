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
st.markdown("*With a 35°C baseline of 0% impervious, 0% tree canopy, 0.15 albedo. All coefficients are in °C.*")
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
    Tree canopy functions as the offset to higher temperatures, so the above formula **subtracts** tree impact from total temp. Every +1% tree canopy = -0.01°C air temperature.
    """)
st.markdown(
    "Sources: "
    "[nature.com Heat Exposure Reductions in Boston](https://www.nature.com/articles/s43247-025-02462-3), "
    "[Cambridge Green Infrastructure Analysis](https://www.cambridgema.gov/-/media/files/cdd/climate/ccpr/ccpralewifeappendixbgianalysisanduhimodeling_processed.pdf)"
)
st.write("")
st.markdown("""**Green roofs | $A_{GROOF}$ = 0.003**  
    Green roofs introduce cooling on the rooftop level by transpiring water vapor into the air. While their leaves reflect solar radiation, their soil traps and releases heat through the night.
    """)
st.markdown(
    "Sources: "
    "[nature.com Heat Exposure Reductions in Boston](https://www.nature.com/articles/s43247-025-02462-3), "
    "[Cooling the cities - A review of reflective and green roof mitigation technologies](https://www.sciencedirect.com/science/article/abs/pii/S0038092X12002447?via%3Dihub)"
)
st.write("")
st.markdown("""**Albedo | $A_{ALB}$ = 6.1**  
    Albedo is measured on a scale of 0 to 1, or 0% to 100% of light absorbed. This coefficient highlights the 6.1°C range across the albedo scale.  
    """)
st.markdown("""### 2. Human "Real Feel" Formula (Mean Radiant Temperature)""")

st.markdown("""**Impervious surface | $MRT_{IMP}$ = 0.12**  
    Since concrete both reflects the sun's shortwaves back at your body and also radiates longwaves as it absorbs and releases heat, removing impervious surface can have high cooling potential."
    """)
st.markdown(
    "Source: "
    "[Resilient Cambridge Urban Heat Island Report](https://www.cambridgema.gov/-/media/files/cdd/climate/resilientcambridge/urbanheatislandtechnicalreport.pdf?)"
)
st.markdown("*Identifies the relationship between tree canopy and impervious surface that removing impervious surface has ~83% the cooling impact to humans as adding an equal area of tree canopy.*")
st.write("")
st.markdown("""**Tree canopy | $MRT_{TREE}$ = 0.15**  
    Trees have multiple benefits to human thermal comfort: they cool the air, they cool the ground, and they prevent direct shortwave solar radiation from reaching the human body.
    """)
st.markdown(
    "Sources: "
    "[BU Combatting Urban Heat Island in Boston](https://www.bu.edu/rccp/files/2026/01/EE538-Madeline-Hale-Urban-Heat-Island-Effect-Paper.pdf) | "
    "[Resilient Cambridge Urban Heat Island Report](https://www.cambridgema.gov/-/media/files/cdd/climate/resilientcambridge/urbanheatislandtechnicalreport.pdf?) | "
    "[nature.com Heat Exposure Reductions in Boston](https://www.nature.com/articles/s43247-025-02462-3)"
)
st.markdown("*The BU paper notes an 11-25°C cooling impact on <u>surfaces</u> shaded by trees. This means that the longwave heat radiated by sidewalks and asphalt under trees feels between 11-25°C cooler to the human body. Additionally, trees cool the localized air temperature for humans by up to 3-5°C, say nature.com and Cambridge, through higher albedo and <u>evapotranspiration</u>.*", unsafe_allow_html=True)
st.markdown(
    "*What is evapotranspiration?*", 
    help="Evapotranspiration is like nature's air conditioner. As water **evaporates** from soil and **transpires** from plants' leaves, vegetation absorbs heat from the air, which cools the urban environment."
)
st.write("")
st.markdown("""**Green roofs | $MRT_{GROOF}$ = 0.002**  
    Green roofs introduce cooling on the rooftop level. While they are powerful with evapotranspiration and more reflective leaves, their soil traps and releases heat into the night.
    """)
st.markdown(
    "Source: "
    "[Cooling the cities - A review of reflective and green roof mitigation technologies](https://www.sciencedirect.com/science/article/abs/pii/S0038092X12002447?via%3Dihub)"
)
st.markdown("*Since green roofs provide neither shade nor evapotranspirative benefits to humans at the ground level, their impact is largely beneficial only to air temperature.*")
st.write("")
st.markdown("""**Albedo | $MRT_{ALB}$ = 5.0**  
    Albedo is measured on a scale of 0 to 1, or 0% to 100% of light absorbed. This coefficient highlights the 6.1°C range across the albedo scale.  
    """)
st.markdown("""### 3. Environmental Justice & Microclimate Vulnerability""")
st.write(
    "Document the studies regarding neighborhood-sized microclimates, historic redlining trends "
    "in Massachusetts, and the socio-economic disparities linked to surface albedo and tree cover distribution."
)
