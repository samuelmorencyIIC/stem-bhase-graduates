import dash
from dash import html, dcc, Output, Input, State, callback_context, dash_table  # Updated import
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import functools
import dash_leaflet as dl
import colorlover as cl
from dash_extensions.javascript import assign  # Import assign for JavaScript functions
import json

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load spatial data
# Load shapefiles for provinces and Census Metropolitan Areas (CMA)
province_sf = gpd.read_file("lcsd000b21a_e.shp")
cma_sf = gpd.read_file("lcma000b21a_e.shp")
csd_sf = gpd.read_file("lcsd000b21a_e.shp")
# Convert the coordinate reference system to WGS84 (EPSG:4326) for compatibility with Leaflet
province_longlat = province_sf.to_crs(epsg=4326)
cma_longlat = cma_sf.to_crs(epsg=4326)
csd_longlat = csd_sf.to_crs(epsg=4326)

cma_longlat_clean = cma_longlat.rename(columns={'CMATYPE': 'TYPE'})
cma_longlat_clean = cma_longlat_clean.rename(columns={'CMANAME': 'NAME'})
cma_longlat_clean = cma_longlat_clean.drop(columns=['DGUIDP', 'CMAUID', 'CMAPUID'])
cma_longlat_clean['CLASS'] = "CMA_CA"

csd_longlat_clean = csd_longlat.rename(columns={'CSDTYPE': 'TYPE'})
csd_longlat_clean = csd_longlat_clean.rename(columns={'CSDNAME': 'NAME'})
csd_longlat_clean = csd_longlat_clean.drop(columns=['CSDUID'])
csd_longlat_clean['CLASS'] = "CSD"

combined_longlat_clean = pd.concat([cma_longlat_clean, csd_longlat_clean], ignore_index=True)


# Load educational data
# Load educational data from an Excel file
data = pd.read_excel("data.xlsx", sheet_name="CMAGrads", dtype={'DGUID': str})

# Transform data
# Reshape the data from wide to long format to make it easier to work with
data = data.melt(
    id_vars=[
        "STEM/BHASE",  # Science, Technology, Engineering, Math / Business, Humanities, Arts, Social Sciences
        "Province_Territory",
        "CMA_CA",  # Census Metropolitan Area / Census Agglomeration
        "Institution",
        "ISCED_level_of_education",  # International Standard Classification of Education level
        "Credential_Type",
        "CIP6_Code",  # Classification of Instructional Programs code
        "CIP_Name",  # Name corresponding to CIP code
        "Student_Status",
        "DGUID"  # Unique identifier for CMA/CA
    ],
    value_vars=["2019_2020", "2020_2021", "2021_2022"],  # Academic years
    var_name="year",  # Create a new column for the year
    value_name="value",  # Create a new column for the value (number of graduates)
)

data = data.dropna(subset=['value']).sort_values(by=data.columns.difference(['value']).tolist())
# Change column type to int32 for column: 'value'
data = data.astype({'value': 'int32'})
# Drop column: 'Student_Status'
data = data.drop(columns=['Student_Status'])

# Performed 1 aggregation grouped on columns: 'STEM/BHASE', 'Province_Territory' and 8 other columns
data = data.groupby(
    ['STEM/BHASE', 
     'Province_Territory', 
     'CMA_CA', 'Institution', 
     'ISCED_level_of_education', 
     'Credential_Type', 
     'CIP6_Code', 
     'CIP_Name', 
     'DGUID', 
     'year']
     ).agg(
         value=('value', 'sum')
         ).reset_index().sort_values(by=data.columns.difference(['value']).tolist())

final_gdf = combined_longlat_clean[combined_longlat_clean['DGUID'].isin(data['DGUID'])]

############################################################################################################

# Initial filter options (full dataset)
# Generate options for each filter based on the unique values in the dataset
stem_bhase_options_full = [{'label': stem, 'value': stem} for stem in sorted(data['STEM/BHASE'].unique())]
year_options_full = [{'label': year, 'value': year} for year in sorted(data['year'].unique())]
prov_options_full = [{'label': prov, 'value': prov} for prov in sorted(data['Province_Territory'].unique())]
isced_options_full = [{'label': level, 'value': level} for level in sorted(data['ISCED_level_of_education'].unique())]
credential_options_full = [{'label': cred, 'value': cred} for cred in sorted(data['Credential_Type'].unique())]
institution_options_full = [{'label': inst, 'value': inst} for inst in sorted(data['Institution'].unique())]

# Create the app layout
app.layout = dbc.Container([
    html.H1("Interactive Choropleth Map of Graduates in Canada", className="my-4"),

    # Define the layout for filters and map
    dbc.Row([
        dbc.Col([
            html.H5("Filters"),
            html.Label("STEM/BHASE:"),
            dcc.Checklist(
                id='stem-bhase-filter',
                options=stem_bhase_options_full,
                value=[option['value'] for option in stem_bhase_options_full],
                inputStyle={"margin-right": "5px", "margin-left": "20px"},
                style={"margin-bottom": "15px"}
            ),
            html.Label("Academic Year:"),
            dcc.Checklist(
                id='year-filter',
                options=year_options_full,
                value=[option['value'] for option in year_options_full],
                inputStyle={"margin-right": "5px", "margin-left": "20px"},
                style={"margin-bottom": "15px"}
            ),
            html.Label("Province:"),
            dcc.Dropdown(
                id='prov-filter',
                options=prov_options_full,
                value=[],
                multi=True,
                placeholder="All Provinces",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Label("ISCED Level:"),
            dcc.Dropdown(
                id='isced-filter',
                options=isced_options_full,
                value=[],
                multi=True,
                placeholder="All Levels",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Label("Credential Type:"),
            dcc.Dropdown(
                id='credential-filter',
                options=credential_options_full,
                value=[],
                multi=True,
                placeholder="All Credential Types",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Label("Institution:"),
            dcc.Dropdown(
                id='institution-filter',
                options=institution_options_full,
                value=[],
                multi=True,
                placeholder="All Institutions",
                searchable=True,
                style={"margin-bottom": "15px"}
            ),
            html.Button('Reset Filters', id='reset-filters', n_clicks=0, style={"margin-top": "15px"}),
            html.Button('Clear Selection', id='clear-selection', n_clicks=0, style={"margin-top": "15px"}),
            # Add dcc.Store components to store selected data for cross-filtering
            dcc.Store(id='selected-isced', data=None),
            dcc.Store(id='selected-province', data=None),
            dcc.Store(id='selected-cma', data=None),
        ], width=3, style={"background-color": "#f8f9fa", "padding": "20px"}),

        dbc.Col([
            html.Div([
                dl.Map(
                    id='map',
                    center=[56, -96],  # Approximate center of Canada
                    zoom=4,
                    children=[
                        dl.TileLayer(),  # Base map layer
                        dl.GeoJSON(
                            id='cma-geojson',
                            data=None,  # Placeholder for the GeoJSON data
                            style=assign("""
                            function(feature) {
                                return feature.properties.style;  // Set style for each feature
                            }
                            """),
                            hoverStyle=dict(
                                weight=2, color='black', dashArray='',
                                fillOpacity=0.7
                            ),
                            onEachFeature=assign("""
                            function(feature, layer) {
                                if (feature.properties && feature.properties.tooltip) {
                                    layer.bindTooltip(feature.properties.tooltip);  // Add tooltip to each feature
                                }
                            }
                            """),
                            options=dict(interactive=True),
                            # Add custom event handler for clicking on features
                            eventHandlers=dict(
                                click=assign("""
                                function(e, ctx) {
                                    e.originalEvent._stopped = true;  // Prevent event from bubbling up to the map
                                    ctx.setProps({ clickedFeature: e.sourceTarget.feature.properties.DGUID });
                                }
                                """)
                            ),
                        ),
                    ],
                    style={'width': '100%', 'height': '600px'},
                ),
            ], style={"height": "600px"}),

            # Arrange the two graphs side by side with chart type selection
            dbc.Row([
                dbc.Col([
                    html.Label("Chart Type:"),
                    dcc.RadioItems(
                        id='chart-type-isced',
                        options=[
                            {'label': 'Bar', 'value': 'bar'},
                            {'label': 'Line', 'value': 'line'},
                            {'label': 'Pie', 'value': 'pie'}
                        ],
                        value='bar',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    ),
                    dcc.Graph(id='graph-isced'),  # Graph for ISCED level of education
                ], width=6),
                dbc.Col([
                    html.Label("Chart Type:"),
                    dcc.RadioItems(
                        id='chart-type-province',
                        options=[
                            {'label': 'Bar', 'value': 'bar'},
                            {'label': 'Line', 'value': 'line'},
                            {'label': 'Pie', 'value': 'pie'}
                        ],
                        value='bar',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    ),
                    dcc.Graph(id='graph-province'),  # Graph for provinces
                ], width=6)
            ]),

            # Add the scrollable table at the bottom
            html.H3("Number of Graduates by CMA/CA"),
            dash_table.DataTable(
                id='table-cma',
                columns=[],  # Placeholder for table columns
                data=[],  # Placeholder for table data
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left'},
                page_action='none',  # Disable pagination
                sort_action='native',  # Enable sorting
                filter_action='native',  # Enable filtering
            ),
        ], width=9)
    ])
], fluid=True)

# Function to preprocess and cache data
@functools.lru_cache(maxsize=128)
def preprocess_data(selected_stem_bhase, selected_years, selected_provs, selected_isced, selected_credentials, selected_institutions):
    # Convert selections to tuples for hashing (required for caching)
    selected_stem_bhase = tuple(selected_stem_bhase) if selected_stem_bhase else ()
    selected_years = tuple(selected_years) if selected_years else ()
    selected_provs = tuple(selected_provs) if selected_provs else ()
    selected_isced = tuple(selected_isced) if selected_isced else ()
    selected_credentials = tuple(selected_credentials) if selected_credentials else ()
    selected_institutions = tuple(selected_institutions) if selected_institutions else ()

    # Apply filters to the data based on user selection
    filtered_data = data
    if selected_stem_bhase:
        filtered_data = filtered_data[filtered_data['STEM/BHASE'].isin(selected_stem_bhase)]
    if selected_years:
        filtered_data = filtered_data[filtered_data['year'].isin(selected_years)]
    if selected_provs:
        filtered_data = filtered_data[filtered_data['Province_Territory'].isin(selected_provs)]
    if selected_isced:
        filtered_data = filtered_data[filtered_data['ISCED_level_of_education'].isin(selected_isced)]
    if selected_credentials:
        filtered_data = filtered_data[filtered_data['Credential_Type'].isin(selected_credentials)]
    if selected_institutions:
        filtered_data = filtered_data[filtered_data['Institution'].isin(selected_institutions)]

    # Prepare data for mapping - aggregate graduates by CMA/CA
    cma_grads = (
        filtered_data.groupby(["CMA_CA", "DGUID"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "graduates"})
    )
    # Data for ISCED level graph - aggregate graduates by ISCED level
    isced_grads = (
        filtered_data.groupby("ISCED_level_of_education", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "graduates"})
    )
    # Data for Province graph - aggregate graduates by province
    province_grads = (
        filtered_data.groupby("Province_Territory", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "graduates"})
    )

    return filtered_data, cma_grads, isced_grads, province_grads

# Callback to update selected CMA/CA from the map
@app.callback(
    Output('selected-cma', 'data'),
    Input('cma-geojson', 'clickedFeature'),
    Input('map', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-cma', 'data'),
)
def update_selected_cma(clicked_feature, map_click_data, n_clicks, stored_cma):
    ctx = callback_context
    if not ctx.triggered:
        return stored_cma
    else:
        triggered_prop = ctx.triggered[0]['prop_id']
        if 'clear-selection' in triggered_prop:
            return None  # Clear selection when 'Clear Selection' button is clicked
        elif 'cma-geojson' in triggered_prop:
            if clicked_feature:
                cmapuid = str(clicked_feature)
                if stored_cma == cmapuid:
                    # Toggle off the selection if the same feature is clicked again
                    return None
                else:
                    # Update the selection with the new feature
                    return cmapuid
            else:
                # Clicked on a blank area or invalid data
                return None
        elif 'map' in triggered_prop:
            # Clear the selection when clicking outside of a feature
            if map_click_data:
                return None
    return stored_cma

# Callback to update selected ISCED level
@app.callback(
    Output('selected-isced', 'data'),
    Input('graph-isced', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-isced', 'data')
)
def update_selected_isced(clickData, n_clicks, stored_isced):
    ctx = callback_context
    if not ctx.triggered:
        return stored_isced
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'clear-selection':
            return None  # Clear selection when 'Clear Selection' button is clicked
        elif triggered_id == 'graph-isced':
            if clickData and 'points' in clickData:
                clicked_value = clickData['points'][0]['y']
                if stored_isced == clicked_value:
                    # Toggle off the selection if the same value is clicked again
                    return None
                else:
                    # Update the selection with the new value
                    return clicked_value
            else:
                # Clicked on a blank area or invalid data
                return None
    return stored_isced

# Callback to update selected Province/Territory
@app.callback(
    Output('selected-province', 'data'),
    Input('graph-province', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('selected-province', 'data'),
    State('graph-province', 'figure')
)
def update_selected_province(clickData, n_clicks, stored_province, figure):
    ctx = callback_context
    if not ctx.triggered:
        return stored_province
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'clear-selection':
            return None  # Clear selection when 'Clear Selection' button is clicked
        elif triggered_id == 'graph-province':
            if clickData and 'points' in clickData:
                # Determine orientation to extract the correct axis value
                orientation = figure['data'][0].get('orientation', 'v')
                if orientation == 'h':
                    clicked_value = clickData['points'][0]['y']
                else:
                    clicked_value = clickData['points'][0]['x']
                if stored_province == clicked_value:
                    # Toggle off the selection if the same value is clicked again
                    return None
                else:
                    # Update the selection with the new value
                    return clicked_value
            else:
                # Clicked on a blank area or invalid data
                return None
    return stored_province

# Define main callback to update visualizations
@app.callback(
    Output('cma-geojson', 'data'),
    Output('graph-isced', 'figure'),
    Output('graph-province', 'figure'),
    Output('table-cma', 'data'),
    Output('table-cma', 'columns'),
    Output('map', 'viewport'),  # Output to viewport for animated transitions
    Input('stem-bhase-filter', 'value'),
    Input('year-filter', 'value'),
    Input('prov-filter', 'value'),
    Input('isced-filter', 'value'),
    Input('credential-filter', 'value'),
    Input('institution-filter', 'value'),
    Input('chart-type-isced', 'value'),
    Input('chart-type-province', 'value'),
    Input('selected-isced', 'data'),
    Input('selected-province', 'data'),
    Input('selected-cma', 'data'),
)
def update_visualizations(selected_stem_bhase, selected_years, selected_provs, selected_isced,
                          selected_credentials, selected_institutions, chart_type_isced,
                          chart_type_province, selected_isced_store, selected_province_store, selected_cma):
    # Preprocess and cache data based on filters
    filtered_data, cma_grads, isced_grads, province_grads = preprocess_data(
        tuple(selected_stem_bhase) if selected_stem_bhase else (),
        tuple(selected_years) if selected_years else (),
        tuple(selected_provs) if selected_provs else (),
        tuple(selected_isced) if selected_isced else (),
        tuple(selected_credentials) if selected_credentials else (),
        tuple(selected_institutions) if selected_institutions else ()
    )

    # Apply cross-filtering based on stored selections
    if selected_isced_store:
        filtered_data = filtered_data[filtered_data['ISCED_level_of_education'] == selected_isced_store]

    if selected_province_store:
        filtered_data = filtered_data[filtered_data['Province_Territory'] == selected_province_store]

    if selected_cma:
        filtered_data = filtered_data[filtered_data['DGUID'] == selected_cma]

    # Now regenerate the aggregated data after cross-filtering
    cma_grads = (
        filtered_data.groupby(["CMA_CA", "DGUID"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "graduates"})
    )
    isced_grads = (
        filtered_data.groupby("ISCED_level_of_education", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "graduates"})
    )
    province_grads = (
        filtered_data.groupby("Province_Territory", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "graduates"})
    )

    # Merge filtered data with spatial data for mapping
    cma_data = combined_longlat_clean.merge(cma_grads, on='DGUID', how='left')

    # Fill NaN values with 0 (for areas with no graduates)
    cma_data['graduates'] = cma_data['graduates'].fillna(0)

    # Filter out regions with zero graduates
    cma_data = cma_data[cma_data['graduates'] > 0]

    if not cma_data.empty:
        # Calculate bounds for the map
        minx, miny, maxx, maxy = cma_data.total_bounds
        bounds = [[miny, minx], [maxy, maxx]]

        # Prepare the GeoJSON data for the map
        cma_geojson = cma_data.__geo_interface__

        # Define colorscale for visualization
        colorscale = cl.scales['9']['seq']['Reds']

        # Map graduates to colors for choropleth representation
        max_graduates = cma_data['graduates'].max()
        min_graduates = cma_data['graduates'].min()

        for feature in cma_geojson['features']:
            graduates = feature['properties']['graduates']
            cmapuid = str(feature['properties']['DGUID'])
            cma_ca_name = feature['properties']['CMA_CA']

            # Normalize graduate count for color assignment
            if max_graduates > min_graduates:
                normalized_value = (graduates - min_graduates) / (max_graduates - min_graduates)
            else:
                normalized_value = 0  # Avoid division by zero if all values are equal

            # Highlight selected CMA/CA in yellow
            if selected_cma and cmapuid == selected_cma:
                color = 'yellow'
            elif graduates > 0:
                color_index = int(normalized_value * (len(colorscale) - 1))
                color = colorscale[color_index]
            else:
                color = 'lightgray'

            feature['properties']['style'] = {
                'fillColor': color,
                'color': 'black' if selected_cma and cmapuid == selected_cma else 'gray',
                'weight': 2 if selected_cma and cmapuid == selected_cma else 0.5,
                'fillOpacity': 0.8,
            }

            # Add tooltip information for each feature
            feature['properties']['tooltip'] = f"CMA/CA: {cma_ca_name}<br>Graduates: {int(graduates)}"

        # Create the ISCED level chart
        fig_isced = create_chart(isced_grads, 'ISCED_level_of_education', 'graduates', chart_type_isced, 'ISCED Level of Education', colorscale, selected_isced_store)

        # Create the Province chart
        fig_province = create_chart(province_grads, 'Province_Territory', 'graduates', chart_type_province, 'Province/Territory', colorscale, selected_province_store)

        # Prepare data and columns for the DataTable
        cma_grads_sorted = cma_grads.sort_values('graduates', ascending=False)
        table_data = cma_grads_sorted.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in cma_grads_sorted.columns]

        # Return the GeoJSON data, chart figures, table data, and viewport for animated map transition
        return cma_geojson, fig_isced, fig_province, table_data, table_columns, dict(bounds=bounds, transition="flyToBounds")
    else:
        # Return empty data and default bounds if no data is available
        empty_fig = {}
        empty_data = []
        empty_columns = []
        empty_geojson = {'type': 'FeatureCollection', 'features': []}
        default_bounds = [[41, -141], [83, -52]]  # Approximate bounds of Canada
        return empty_geojson, empty_fig, empty_fig, empty_data, empty_columns, dict(bounds=default_bounds, transition="flyToBounds")

# Modify the create_chart function to highlight selected elements
def create_chart(dataframe, x_column, y_column, chart_type, x_label, colorscale, selected_value):
    if not dataframe.empty:
        # Calculate vmin_bar and vmax_bar for color scaling
        vmin_bar = dataframe[y_column].quantile(0.01)
        vmax_bar = dataframe[y_column].max()
        if vmin_bar == vmax_bar:
            vmin_bar = 0

        if chart_type == 'bar':
            # Sort the dataframe for better visualization
            dataframe = dataframe.sort_values(y_column, ascending=True)
            dataframe = dataframe.reset_index(drop=True)  # Reset index to ensure correct indexing
            fig = px.bar(
                dataframe,
                x=y_column,
                y=x_column,
                orientation='h',
                title=f'Number of Graduates by {x_label}',
                labels={y_column: 'Number of Graduates', x_column: x_label},
                color=y_column,
                color_continuous_scale=px.colors.sequential.Reds,
                range_color=[vmin_bar, vmax_bar]
            )
            if selected_value:
                # Highlight the selected value in red
                num_bars = len(dataframe)
                colors = ['grey'] * num_bars
                idx = dataframe[dataframe[x_column] == selected_value].index
                if not idx.empty:
                    idx = idx[0]
                    colors[idx] = 'red'
                fig.data[0].marker.color = colors
                fig.update_coloraxes(showscale=False)  # Hide color scale when custom colors are used
            else:
                fig.update_coloraxes(colorbar_title='Graduates')
            fig.update_layout(
                xaxis_title='Number of Graduates',
                yaxis_title=x_label,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                clickmode='event+select'  # Enable click events
            )
        elif chart_type == 'line':
            # Sort the dataframe by x_column
            dataframe = dataframe.sort_values(x_column, ascending=True)
            fig = px.line(
                dataframe,
                x=x_column,
                y=y_column,
                title=f'Number of Graduates by {x_label}',
                labels={y_column: 'Number of Graduates', x_column: x_label},
            )
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title='Number of Graduates',
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                clickmode='event+select'  # Enable click events
            )
        elif chart_type == 'pie':
            fig = px.pie(
                dataframe,
                names=x_column,
                values=y_column,
                title=f'Number of Graduates by {x_label}',
                color_discrete_sequence=px.colors.sequential.Reds,
            )
            if selected_value:
                # Highlight the selected value by pulling the pie slice
                fig.update_traces(pull=[0.1 if name == selected_value else 0 for name in dataframe[x_column]])
            fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                clickmode='event+select'  # Enable click events
            )
        else:
            fig = {}
    else:
        fig = {}
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
