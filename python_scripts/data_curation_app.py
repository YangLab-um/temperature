from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app_style = {
    'track-marker-size': 10,
    'peak-trough-marker-size': 12.5,
    'peak-marker-color': 'red',
    'peak-marker-symbol': 'square',
    'trough-marker-color': 'limegreen',
    'trough-marker-symbol': 'diamond'
}
output_order = {'peak': 0, 'trough': 1}

# Initial dataset
all_tracks = pd.DataFrame({
    "TIME": [np.NaN],
    "RATIO": [np.NaN],
    "TRACK_ID": [np.NaN],
})
all_peaks = pd.DataFrame({
    "TIME": [np.NaN],
    "RATIO": [np.NaN],
    "CYCLE": [np.NaN],
    "TRACK_ID": [np.NaN],
})
all_troughs = pd.DataFrame({
    "TIME": [np.NaN],
    "RATIO": [np.NaN],
    "CYCLE": [np.NaN],
    "TRACK_ID": [np.NaN],
})

min_id = min(all_tracks['TRACK_ID'])
id_list = all_tracks['TRACK_ID'].unique()

def get_current_data(df, id):
    return df[df['TRACK_ID'] == id]

def plot_track_with_peaks_and_troughs(id, df_tracks, df_peaks, df_troughs, min_time, max_time):
    # Filter data
    tracks = get_current_data(df_tracks, id)
    peaks = get_current_data(df_peaks, id)
    troughs = get_current_data(df_troughs, id)
    # Sort data
    tracks = tracks.sort_values('TIME')
    # Track
    fig = px.scatter(tracks, x="TIME", y="RATIO")
    fig.update_traces(mode='lines+markers')
    fig.update_layout(clickmode='event')
    fig.update_traces(marker_size=app_style['track-marker-size'])
    # Peaks
    fig.add_trace(px.scatter(peaks, x="TIME", y="RATIO").data[0])
    fig.data[1].marker.color = app_style['peak-marker-color']
    fig.data[1].marker.size = app_style['peak-trough-marker-size']
    fig.data[1].marker.symbol = app_style['peak-marker-symbol']
    # Troughs
    fig.add_trace(px.scatter(troughs, x="TIME", y="RATIO").data[0])
    fig.data[2].marker.color = app_style['trough-marker-color']
    fig.data[2].marker.size = app_style['peak-trough-marker-size']
    fig.data[2].marker.symbol = app_style['trough-marker-symbol']
    # Labels
    fig.update_xaxes(title_text='Time (min)', tickfont_size=14)
    fig.update_yaxes(title_text='FRET/CFP Ratio (a.u.)', tickfont_size=14)
    # Margins
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20))
    if min_time < max_time:
        fig.update_xaxes(range=[float(min_time), float(max_time)])
    return fig

# Upload peaks and troughs
def upload_peaks_troughs(contents, filename):
    all_peaks = pd.DataFrame({})
    all_troughs = pd.DataFrame({})
    message = html.Div([f"{filename} (Click to upload a different file)"])
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        path = io.BytesIO(decoded)
        all_peaks_and_troughs = pd.read_csv(path)
        all_peaks = all_peaks_and_troughs[all_peaks_and_troughs['TYPE'] == 'PEAK']
        all_troughs = all_peaks_and_troughs[all_peaks_and_troughs['TYPE'] == 'TROUGH']
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ]), all_peaks.to_dict('records'), all_troughs.to_dict('records')
    return message, all_peaks.to_dict('records'), all_troughs.to_dict('records')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Data Curation App"),
        ], width=4, style={'padding-top': '20px'}),
        dbc.Col([
            dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Load processed spots (Click to upload)'
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
        ], width=4, style={'padding-top': '10px'}),
        dbc.Col([
            dcc.Upload(
                    id='upload-peaks-troughs',
                    children=html.Div([
                        'Load peak/trough data (Click to upload)'
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='output-peaks-troughs-upload'),
        ], width=4, style={'padding-top': '10px'}),
    ]),
    dbc.Row([
        dcc.Graph(
            id='track-plot',
            figure=plot_track_with_peaks_and_troughs(min_id, all_tracks, all_peaks, all_troughs, 0, 1)
        ),
    ], className="g-0"),
    dbc.Row([
        dbc.Col([
            html.H3("Track ID: "),
        ], width=2, style={'padding-top': '10px'}),
        dbc.Col([
            dcc.Dropdown(
                id='track-id',
                options=[],
            ),
        ], width=2, style={'padding-top': '10px'}),
        dbc.Col([
            html.H4("Time range: "),
        ], width=2, style={'padding': '10px'}),
        dbc.Col([
            dcc.Input(id='min-time', type='number', value=0),
        ], width=2, style={'padding': '10px'}),
        dbc.Col([
            dcc.Input(id='max-time', type='number', value=1),
        ], width=2, style={'padding': '10px'}),
    ]),
    dbc.Row([
        dbc.Col([
            html.H6("Click on the plot to add/remove peaks or troughs"),
        ], width=2, style={'padding': '10px'}),
        dbc.Col([
            dbc.RadioItems(
                id='peak-trough-selection',
                options=[
                    {'label': 'Peaks', 'value': 'peak'},
                    {'label': 'Troughs', 'value': 'trough'}
                ],
                value='peak',
                labelStyle={'display': 'inline-block'}
            ),
        ], width=1, style={'padding': '10px'}),
        dbc.Col([
            dbc.RadioItems(
                id='add-remove-points',
                options=[
                    {'label': 'Add', 'value': 'add'},
                    {'label': 'Remove', 'value': 'remove'}
                ],
                value='add',
                labelStyle={'display': 'inline-block'}
            ),
        ], width=1, style={'padding': '10px'}),
        dbc.Col([
            dbc.Button("Previous", id="previous-track-button", color="primary", className="mr-1"),
        ], width=1, style={'padding': '10px'}),
        dbc.Col([
            dbc.Button("Next", id="next-track-button", color="primary", className="mr-1"),
        ], width=1, style={'padding': '10px'}),
        dbc.Col([
            dbc.Button("Assign cycle numbers", id="cycle-number-button", color="secondary", className="mr-1"),
        ], width=2, style={'padding': '10px'}),
        dbc.Col([
            dbc.Button("Download", id="download-button", color="success", className="mr-1"),
            dcc.Download(id="download-data"),
        ], width=1, style={'padding': '10px'}),
    ], className="g-0"),
    dbc.Row([
        dash_table.DataTable(
            id='all-tracks-table',
            columns=[{"name": i, "id": i} for i in all_tracks.columns],
            data=all_tracks.to_dict('records'),
            editable=False
        )
    ], style={'display': 'none'}),
    dbc.Row([
        dash_table.DataTable(
            id='all-peaks-table',
            columns=[{"name": i, "id": i} for i in all_peaks.columns],
            data=all_peaks.to_dict('records'),
            editable=False
        )
    ], style={'display': 'none'}),
    dbc.Row([
        dash_table.DataTable(
            id='all-troughs-table',
            columns=[{"name": i, "id": i} for i in all_troughs.columns],
            data=all_troughs.to_dict('records'),
            editable=False
        )
    ], style={'display': 'none'}),
    dbc.Row([
        dbc.Col([
            html.H3("Peaks", style={'color': 'red'}),
            dash_table.DataTable(
                id='current-peaks-table',
                columns=[{"name": i, "id": i} for i in get_current_data(all_peaks, min_id).columns],
                data=get_current_data(all_peaks, min_id).to_dict('records'),
                editable=True,
                sort_action='native',
                row_deletable=True,
            )], width=6, style={'padding': '10px'}),
       dbc.Col([
            html.H3("Troughs", style={'color': 'limegreen'}),
            dash_table.DataTable(
                id='current-troughs-table',
                columns=[{"name": i, "id": i} for i in get_current_data(all_troughs, min_id).columns],
                data=get_current_data(all_troughs, min_id).to_dict('records'),
                editable=True,
                sort_action='native',
                row_deletable=True,
            )], width=6, style={'padding': '10px'}),
       ], className="g-0"),
])

@callback(
    Output('upload-data', 'children'),
    Output('all-tracks-table', 'data'),
    Output('track-id', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True)
def upload_data(contents, filename, last_modified):
    all_tracks = pd.DataFrame({})
    message = html.Div([f"{filename} (Click to upload a different file)"])
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        path = io.BytesIO(decoded)
        all_tracks = pd.read_csv(path)
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ]), all_tracks.to_dict('records'), []
    # Update track ids
    id_options = [{'label': i, 'value': i} for i in all_tracks['TRACK_ID'].unique()]
    return message, all_tracks.to_dict('records'), id_options


@callback(
    Output('download-data', 'data'),
    Input('download-button', 'n_clicks'),
    State('all-peaks-table', 'data'),
    State('all-troughs-table', 'data'),
    prevent_initial_call=True)
def download_data(n_clicks, all_peaks, all_troughs):
    all_peaks = pd.DataFrame(all_peaks)
    all_troughs = pd.DataFrame(all_troughs)
    # Combine into a single dataframe
    all_peaks['TYPE'] = 'PEAK'
    all_troughs['TYPE'] = 'TROUGH'
    all_peaks.columns = ['TIME', 'RATIO', 'CYCLE', 'TRACK_ID', 'TYPE']
    all_troughs.columns = ['TIME', 'RATIO', 'CYCLE', 'TRACK_ID', 'TYPE']
    all_peaks_and_troughs = pd.concat([all_peaks, all_troughs], ignore_index=True)
    return dcc.send_data_frame(all_peaks_and_troughs.to_csv, "peaks_and_troughs.csv", index=False)

@callback(
    Output('track-id', 'value'),
    Input('previous-track-button', 'n_clicks'),
    Input('next-track-button', 'n_clicks'),
    Input('track-id', 'options'),
    State('track-id', 'value'),
    State('all-tracks-table', 'data'))
def update_track_id(prev_clicks, next_clicks, id_options, current_id, all_tracks):
    all_ids = pd.DataFrame(all_tracks)['TRACK_ID'].unique()
    if ctx.triggered[0]['prop_id'] == 'previous-track-button.n_clicks':
        if current_id == min(all_ids):
            new_id = current_id
        else:
            new_id = all_ids[all_ids < current_id].max()
    elif ctx.triggered[0]['prop_id'] == 'next-track-button.n_clicks':
        if current_id == max(all_ids):
            new_id = current_id
        else:
            new_id = all_ids[all_ids > current_id].min()
    elif ctx.triggered[0]['prop_id'] == 'track-id.options':
        new_id = min(all_ids)
    else:
        new_id = current_id
    return new_id

@callback(
    Output('upload-peaks-troughs', 'children'),
    Output('all-peaks-table', 'data'),
    Output('all-troughs-table', 'data'),
    Input('all-tracks-table', 'data'),
    Input('current-peaks-table', 'data'),
    Input('current-troughs-table', 'data'),
    Input('upload-peaks-troughs', 'contents'),
    State('track-id', 'value'),
    State('all-peaks-table', 'data'),
    State('all-troughs-table', 'data'),
    State('upload-peaks-troughs', 'filename'),
    prevent_initial_call=True)
def update_all_peaks_and_troughs(all_tracks, current_peaks, current_troughs, contents,
                                 current_id, all_peaks, all_troughs, filename):
    # New peaks and troughs uploaded
    if ctx.triggered[0]['prop_id'] == 'upload-peaks-troughs.contents':
        return upload_peaks_troughs(contents, filename)
    # Same tracks, update peaks and troughs
    current_peaks = pd.DataFrame(current_peaks)
    current_troughs = pd.DataFrame(current_troughs)
    all_peaks = pd.DataFrame(all_peaks)
    all_troughs = pd.DataFrame(all_troughs)
    if current_peaks.empty:
        current_peaks = pd.DataFrame({
            "TIME": [np.nan],
            "RATIO": [np.nan],
            "CYCLE": [np.nan],
            "TRACK_ID": current_id,
        })
    if current_troughs.empty:
        current_troughs = pd.DataFrame({
            "TIME": [np.nan],
            "RATIO": [np.nan],
            "CYCLE": [np.nan],
            "TRACK_ID": current_id,
        })
    all_peaks = pd.concat([all_peaks[all_peaks['TRACK_ID'] != current_id], current_peaks], ignore_index=True)
    all_troughs = pd.concat([all_troughs[all_troughs['TRACK_ID'] != current_id], current_troughs], ignore_index=True)
    return [no_update, all_peaks.to_dict('records'), all_troughs.to_dict('records')]

@callback(
    Output('current-peaks-table', 'data'),
    Output('current-troughs-table', 'data'),
    Input('track-plot', 'clickData'),
    Input('track-id', 'value'),
    Input('cycle-number-button', 'n_clicks'),
    State('current-peaks-table', 'data'),
    State('current-troughs-table', 'data'),
    State('all-peaks-table', 'data'),
    State('all-troughs-table', 'data'),
    State('add-remove-points', 'value'),
    State('peak-trough-selection', 'value'))
def update_click_data(click_data, current_id, update_cycle_number, current_peak_data, current_trough_data, all_peaks, all_troughs, add_or_remove, peak_or_trough):
    if ctx.triggered[0]['prop_id'] == 'track-id.value':
        # Track ID changed, update current data
        all_peaks = pd.DataFrame(all_peaks)
        all_troughs = pd.DataFrame(all_troughs)
        current_peaks = get_current_data(all_peaks, current_id)
        current_troughs = get_current_data(all_troughs, current_id)
        output = [current_peaks.to_dict('records'), current_troughs.to_dict('records')]
        return output
    elif ctx.triggered[0]['prop_id'] == 'cycle-number-button.n_clicks':
        # Assign cycle numbers
        current_peak_data = pd.DataFrame(current_peak_data).sort_values('TIME').reset_index(drop=True)
        current_trough_data = pd.DataFrame(current_trough_data).sort_values('TIME').reset_index(drop=True)
        current_peak_data['CYCLE'] = np.arange(len(current_peak_data)) % len(current_peak_data)
        current_trough_data['CYCLE'] = np.arange(len(current_trough_data)) % len(current_trough_data)
        output = [current_peak_data.to_dict('records'), current_trough_data.to_dict('records')]
        return output

    output = [no_update, no_update]
    if click_data is None:
        return output
    else:
        if peak_or_trough == 'peak':
            old_data = pd.DataFrame(current_peak_data, dtype=float)
        elif peak_or_trough == 'trough':
            old_data = pd.DataFrame(current_trough_data, dtype=float)

        if old_data.empty:
            if add_or_remove == 'add':
                new_data = pd.DataFrame({
                    "TIME": [click_data['points'][0]['x']],
                    "RATIO": [click_data['points'][0]['y']],
                    "CYCLE": np.NaN,
                    "TRACK_ID": current_id,
                })
                output[output_order[peak_or_trough]] = new_data.to_dict('records')
                return output
            elif add_or_remove == 'remove':
                return output
        else:
            same_x = old_data['TIME'] == click_data['points'][0]['x']
            same_y = old_data['RATIO'] == click_data['points'][0]['y']
            if add_or_remove == 'remove':
                new_data = old_data[~(same_x & same_y)]
                output[output_order[peak_or_trough]] = new_data.to_dict('records')
                return output
            elif add_or_remove == 'add':
                point_exists = (same_x & same_y).any()
                if point_exists:
                    return output
                else:
                    datapoint = pd.DataFrame({
                        "TIME": [click_data['points'][0]['x']],
                        "RATIO": [click_data['points'][0]['y']],
                        "CYCLE": np.NaN,
                        "TRACK_ID": current_id,
                    })
                    new_data = pd.concat([old_data, datapoint], ignore_index=True)
                    output[output_order[peak_or_trough]] = new_data.to_dict('records')
                    return output

@callback(
    Output('track-plot', 'figure'),
    Input('track-id', 'value'),
    Input('current-peaks-table', 'data'),
    Input('current-troughs-table', 'data'),
    Input('min-time', 'value'),
    Input('max-time', 'value'),
    State('all-tracks-table', 'data'))
def update_plot(current_id, current_peaks, current_troughs, min_time, max_time, all_tracks):
    current_track = get_current_data(pd.DataFrame(all_tracks), current_id)
    current_peaks = pd.DataFrame(current_peaks)
    current_troughs = pd.DataFrame(current_troughs)
    if current_peaks.empty:
        current_peaks = pd.DataFrame({
            "TIME": [np.nan],
            "RATIO": [np.nan],
            "CYCLE": [np.nan],
            "TRACK_ID": current_id,
        })
    if current_troughs.empty:
        current_troughs = pd.DataFrame({
            "TIME": [np.nan],
            "RATIO": [np.nan],
            "CYCLE": [np.nan],
            "TRACK_ID": current_id,
        })
    fig = plot_track_with_peaks_and_troughs(current_id, current_track, current_peaks, current_troughs,
                                            min_time, max_time)
    return fig

if __name__ == '__main__':
    app.run(debug=True)