import streamlit as st
import geemap.foliumap as geemap
import ee
import pandas as pd
from streamlit_folium import st_folium
import folium

# 1. AUTHENTICATE AND INITIALIZE
# ---------------------------------------------------------
# Using the specific project ID provided for Earth Engine access
my_project = 'massachusetts-uhi'

try:
    ee.Initialize(project=my_project)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=my_project)

# 2. APP SETUP & SESSION STATE (CRITICAL FOR STABILITY)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="MA UHI Dashboard")
st.title("Massachusetts Urban Heat Island Dashboard")
st.markdown("""
This dashboard allows you to analyze the **Urban Heat Island (UHI)** effect across Massachusetts. 
Select a town or draw a custom area to see current conditions, then use the **Simulation Lab** to model how land-use changes impact local air temperature and human thermal comfort.
""")

# Replace your current session state block in Section 2 with this:
if 'map_center' not in st.session_state:
    st.session_state.map_center = [42.3, -71.8]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 9
if 'current_data' not in st.session_state:
    st.session_state.current_data = {'tree': 45.0, 'imperv': 15.0, 'albedo': 0.150}
if 'saved_polygon' not in st.session_state:
    st.session_state.saved_polygon = None
if 'bounds' not in st.session_state:
    st.session_state.bounds = None
if 'map_id' not in st.session_state:
    st.session_state.map_id = 0

# 3. DATASETS & CONSTANTS
# ---------------------------------------------------------
# Load Town Boundaries asset
towns = ee.FeatureCollection("projects/massachusetts-uhi/assets/ma_townboundaries")

# Load Land Cover Data (ESA WorldCover 10m)
esa_landcover = ee.ImageCollection("ESA/WorldCover/v100").first()
tree_canopy = esa_landcover.eq(10).multiply(100).rename('tree').selfMask()

# Load NLCD Impervious Surface (2021 Release)
impervious = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
              .filter(ee.Filter.eq('system:index', '2021'))
              .first()
              .select(['impervious'], ['imperv'])
              .unmask(0).selfMask())

# Landsat 8 for Albedo Calculation (Summer median to avoid snow/cloud interference)
l8_col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterDate('2023-06-01', '2023-09-30')
          .filter(ee.Filter.lt('CLOUD_COVER', 20)))

# Regional Baselines (MA State Averages for context)
STATE_AVG = {'tree': 45.0, 'imperv': 15.0, 'albedo': 0.150}

# --- SCIENTIFIC COEFFICIENTS (IMPACT PER UNIT CHANGE) ---
# Ambient Air Impact (Celsius change per 1% cover or 0.1 Albedo)
A_TREE = 0.05  
A_GROOF = 0.04   # Green roofs are ~80% as effective as trees for ambient air
A_IMP = 0.07   
A_ALB = 15.0   

# Human Comfort Impact (Mean Radiant Temperature / "Real Feel")
MRT_IMP = 0.12   # Radiant heat "penalty" from pavement
MRT_TREE = 0.15  # Radiant relief from direct canopy shade
MRT_GROOF = 0.02 # Minimal radiant relief for pedestrians (no shade)
MRT_ALB = 5.0    # Ambient cooling from reflective urban membranes

# Helper Functions for Unit Conversion
def to_f(c):
    """Converts absolute Celsius to Fahrenheit."""
    return (c * 9/5) + 32

def delta_to_f(c_delta):
    """Converts a temperature CHANGE (delta) from Celsius to Fahrenheit."""
    return c_delta * 1.8

# 4. ANALYSIS LOGIC
# ---------------------------------------------------------
def get_uhi_metrics(geometry):
    """Performs spatial reduction to get average values for the ROI."""
    safe_geom = geometry.buffer(1)
    
    # Calculate Albedo using a standard Landsat 8 formula
    clip_l8 = l8_col.filterBounds(safe_geom).median()
    scaled = clip_l8.multiply(0.0000275).add(-0.2)
    clip_albedo = scaled.expression(
        '0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018', {
        'B2': scaled.select('SR_B2'), 'B4': scaled.select('SR_B4'),
        'B5': scaled.select('SR_B5'), 'B6': scaled.select('SR_B6'),
        'B7': scaled.select('SR_B7')
    }).rename('albedo')

    # Combine all layers into one image for efficient reduction
    combined = tree_canopy.unmask(0).addBands(impervious.unmask(0)).addBands(clip_albedo)
    
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=safe_geom,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    return {
        "tree": stats.get('tree', 0) or 0.0,
        "imperv": stats.get('imperv', 0) or 0.0,
        "albedo": stats.get('albedo', 0) or 0.0
    }

# 5. SIDEBAR & SIMULATION CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Simulation Lab")

# A. BASELINE INPUT (The Weather Station "Neutral" Temp)
sim_temp = st.sidebar.number_input(
    "Base Weather Station Temp (°C)", 
    value=35.0, 
    step=0.1, 
    help="The temperature measured at a standard grassy, shaded weather station (Stevenson Screen)."
)
st.sidebar.caption(f"Standardized Baseline: **{to_f(sim_temp):.1f}°F**")

# B. VEGETATION STRATEGY (Trees vs Green Roofs)
st.sidebar.markdown("---")
st.sidebar.subheader("Vegetation Strategy")
total_veg_target = st.sidebar.number_input(
    "Total Vegetation Goal (%)", 
    min_value=0.0, max_value=100.0, 
    value=float(st.session_state.current_data['tree']),
    help="Target for total vegetative cover (Trees + Rooftops)."
)

use_green_roofs = st.sidebar.checkbox("Allocate portion to Green Roofs?")
sim_groof = 0.0
if use_green_roofs:
    sim_groof = st.sidebar.number_input(
        "Green Roof Allocation (%)", 
        min_value=0.0, max_value=total_veg_target, 
        value=0.0,
        help="Percentage of the total area to be used as green roofs."
    )

# Tree canopy is the ground-level remainder of the goal
sim_tree = total_veg_target - sim_groof
st.sidebar.write(f"Resulting Ground Canopy: **{sim_tree:.1f}%**")

# C. SURFACE STRATEGY (Pavement and Albedo)
st.sidebar.markdown("---")
st.sidebar.subheader("Surface Strategy")
sim_imp = st.sidebar.number_input(
    "Proposed Pavement Cover (%)", 
    min_value=0.0, max_value=100.0, 
    value=float(st.session_state.current_data['imperv']),
    help="Percentage of roads, parking lots, and building footprints."
)

sim_alb = st.sidebar.number_input(
    "Roof & Surface Reflectivity (Albedo)", 
    min_value=0.0, max_value=1.0, 
    value=float(st.session_state.current_data['albedo']), 
    step=0.01,
    help="Ability of surfaces to reflect solar radiation. Standard city avg is ~0.15."
)

# 1. THE RESET LOGIC
if st.sidebar.button("🗑️ Clear Map & Selections"):
    st.session_state.saved_polygon = None
    st.session_state.bounds = None
    st.session_state.town_selector = "None (Explore/Draw Map)"
    # INCREMENT THIS: It forces the map component to fully reset
    st.session_state.map_id += 1 
    st.session_state.show_reset_toast = True
    st.rerun()

# 2. THE TOAST (Optional placement)
if st.session_state.get('show_reset_toast'):
    st.toast("Success! You can now select a new town or polygon!", icon="✅")
    del st.session_state.show_reset_toast

st.sidebar.markdown("---")

# 3. NOW DRAW THE SELECTBOX
@st.cache_data
def get_clean_town_list():
    return towns.distinct('TOWN').aggregate_array('TOWN').sort().getInfo()

selected_town = st.sidebar.selectbox(
    "Select a Town to Focus", 
    ["None (Explore/Draw Map)"] + get_clean_town_list(),
    key="town_selector"
)

if selected_town != "None (Explore/Draw Map)" and st.session_state.saved_polygon is not None:
    st.sidebar.warning("💡 **Multi-Selection Active**")
    st.sidebar.info(
        "You are viewing a custom polygon over a town boundary. "
        "The analysis below is prioritizing your **custom drawing**. "
        "To go back to full town analysis, please use the **Clear Map** button."
    )

# 6. MAP INTERFACE (SILENT MODE)
# ---------------------------------------------------------
# Change map_key to be dynamic
map_key = f"uhi_map_{st.session_state.map_id}"

if map_key in st.session_state:
    map_state = st.session_state[map_key]
    if map_state.get("last_active_drawing"):
        poly_geom = map_state["last_active_drawing"]["geometry"]
        
        # Only update and rerun if it's a new polygon
        if st.session_state.saved_polygon != poly_geom:
            st.session_state.saved_polygon = poly_geom
            all_coords = poly_geom['coordinates'][0]
            lats = [c[1] for c in all_coords]; lngs = [c[0] for c in all_coords]
            st.session_state.bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]
            st.rerun()

m = geemap.Map(center=st.session_state.map_center, zoom=st.session_state.map_zoom)
m.add_basemap("HYBRID")

# Vis params (as you had them)
vis_tree = {'min': 0, 'max': 100, 'palette': ['#ffffff', '#228B22']}
vis_imp = {'min': 0, 'max': 100, 'palette': ['#ffffff', '#444444']}

m.add_layer(tree_canopy, vis_tree, 'Tree Canopy (%)')
m.add_layer(impervious, vis_imp, 'Impervious Surface (%)', False)
m.add_layer(towns.style(color='white', width=1, fillColor='00000000'), {}, 'Town Boundaries')

if st.session_state.saved_polygon:
    folium.GeoJson(st.session_state.saved_polygon,
                   style_function=lambda x: {'fillColor': '#00ff00', 'color': '#00ff00', 'weight': 3, 'fillOpacity': 0.3}
                  ).add_to(m)

# Logic for highlighting and priority zooming
selected_town_geom = None
if selected_town != "None (Explore/Draw Map)":
    roi_features = towns.filter(ee.Filter.eq('TOWN', selected_town))
    selected_town_geom = roi_features.geometry()
    
    # Add neon highlight regardless of zoom
    highlight_style = {'color': '#CCFF00', 'width': 3, 'fillColor': '#00000000'}
    m.add_layer(roi_features.style(**highlight_style), {}, f'Outline: {selected_town}')

# --- THE CAMERA PRIORITY ---
if st.session_state.bounds:
    # If a polygon is drawn (Lynn), focus there
    m.fit_bounds(st.session_state.bounds)
elif selected_town_geom:
    # If no polygon is drawn but a town is selected (Amherst), focus there
    m.center_object(selected_town_geom, 12)

# Render the Map
map_output = st_folium(
    m, 
    key=map_key, 
    height=500, 
    use_container_width=True,
    returned_objects=["last_active_drawing"]
)

# --- THE PRECISION CATCHER (MOVED HERE) ---
# We check the map_output directly for new drawings
if map_output and map_output.get("last_active_drawing"):
    poly_geom = map_output["last_active_drawing"]["geometry"]
    
    if st.session_state.saved_polygon != poly_geom:
        st.session_state.saved_polygon = poly_geom
        
        # Calculate bounds for zooming
        all_coords = poly_geom['coordinates'][0]
        lats = [c[1] for c in all_coords]
        lngs = [c[0] for c in all_coords]
        st.session_state.bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]
        
        # Refresh to trigger Section 7 analysis
        st.rerun()

# 7. RESULTS & CALCULATIONS
# ---------------------------------------------------------
# CONSOLIDATED PRIORITY LOGIC: 
# We prioritize the custom drawing. If none exists, we use the selected town.
active_geom = None

if st.session_state.saved_polygon:
    active_geom = ee.Geometry(st.session_state.saved_polygon)
elif selected_town != "None (Explore/Draw Map)":
    active_geom = selected_town_geom

# Run analysis only if we have a valid geometry
if active_geom:
    try:
        with st.spinner("Analyzing high-resolution climate data..."):
            res = get_uhi_metrics(active_geom)
            
            # SESSION REFRESH: If the new area's data differs from what we have, 
            # update state and rerun to sync the simulation inputs.
            if abs(st.session_state.current_data['tree'] - res['tree']) > 0.01:
                st.session_state.current_data = res
                st.rerun()
            # --- AMBIENT AIR TEMP CALCULATION ---
            # 35C Baseline = 0% Imp, 0% Veg, 0.15 Albedo
            current_air_impact = (res['imperv'] * A_IMP) - (res['tree'] * A_TREE) - ((res['albedo'] - 0.15) * A_ALB)
            current_air_c = sim_temp + current_air_impact
            
            sim_air_impact = (sim_imp * A_IMP) - (sim_tree * A_TREE) - (sim_groof * A_GROOF) - ((sim_alb - 0.15) * A_ALB)
            pred_air_c = sim_temp + sim_air_impact
            
            air_diff_c = pred_air_c - current_air_c

            # --- HUMAN REAL FEEL CALCULATION ---
            # Uses Mean Radiant Temp coefficients to reflect the "broiler" vs "oven" distinction
            curr_feel_c = sim_temp + (res['imperv'] * MRT_IMP) - (res['tree'] * MRT_TREE) - ((res['albedo'] - 0.15) * MRT_ALB)
            sim_feel_c = sim_temp + (sim_imp * MRT_IMP) - (sim_tree * MRT_TREE) - (sim_groof * MRT_GROOF) - ((sim_alb - 0.15) * MRT_ALB)
            
            feel_diff_c = sim_feel_c - curr_feel_c

            # Susceptibility Score Calculation
            score = (res['imperv'] * 0.4) + ((0.4 - res['albedo']) * 100) - (res['tree'] * 0.4)
            risk_color = "red" if score > 50 else "orange" if score > 25 else "green"

            # --- UI RENDERING ---
            st.divider()
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.subheader(f"Current Susceptibility: :{risk_color}[{score:.2f}]")
                st.info("**Why two temps?** Predicted Air Temp measures heated molecules in the atmosphere. Real Feel measures the direct radiant heat hitting your skin from hot pavement.")
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Ground Tree Canopy', 'Green Roofs', 'Pavement Cover', 'Reflectivity (Albedo)'],
                    'Current Status': [f"{res['tree']:.1f}%", "0.0%", f"{res['imperv']:.1f}%", f"{res['albedo']:.3f}"],
                    'Simulated Goal': [f"{sim_tree:.1f}%", f"{sim_groof:.1f}%", f"{sim_imp:.1f}%", f"{sim_alb:.3f}"],
                    'MA State Avg': [f"{STATE_AVG['tree']:.1f}%", "N/A", f"{STATE_AVG['imperv']:.1f}%", f"{STATE_AVG['albedo']:.3f}"]
                })
                st.table(comparison_df)

            with col_b:
                st.subheader("Simulation Results")
                
                # Absolute Air Temperature Metric
                st.metric(
                    label="Predicted Air Temp", 
                    value=f"{pred_air_c:.1f} °C | {to_f(pred_air_c):.1f} °F", 
                    delta=f"{air_diff_c:+.1f} °C | {delta_to_f(air_diff_c):+.1f} °F Change",
                    delta_color="inverse" 
                )
                
                # Human Thermal Comfort Metric
                st.metric(
                    label="Human 'Real Feel'", 
                    value=f"{sim_feel_c:.1f} °C | {to_f(sim_feel_c):.1f} °F", 
                    delta=f"{feel_diff_c:+.1f} °C | {delta_to_f(feel_diff_c):+.1f} °F vs Current",
                    delta_color="inverse"
                )

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("💡 Draw a polygon on the map or select a town from the sidebar to analyze a specific neighborhood.")

