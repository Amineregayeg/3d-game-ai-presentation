
import plotly.graph_objects as go
import numpy as np

# Create a decision tree visualization using Plotly
fig = go.Figure()

# Define nodes with positions (x, y) and styling
nodes = {
    # Level 0 - Start
    'START': {'pos': (0.5, 1.0), 'text': 'START:<br>Choose TTS +<br>Lip-Sync', 'color': '#2E8B57', 'type': 'start'},
    
    # Level 1 - Main priority
    'PRIORITY': {'pos': (0.5, 0.85), 'text': "What's your<br>priority?", 'color': '#1FB8CD', 'type': 'decision'},
    
    # Level 2 - Latency path
    'LATENCY': {'pos': (0.2, 0.7), 'text': 'Sub-100ms<br>TTFB needed?', 'color': '#1FB8CD', 'type': 'decision'},
    'BUDGET': {'pos': (0.15, 0.55), 'text': 'Budget<br>available?', 'color': '#1FB8CD', 'type': 'decision'},
    'QUALITY': {'pos': (0.25, 0.55), 'text': 'Quality ><br>Speed?', 'color': '#1FB8CD', 'type': 'decision'},
    
    # Level 2 - Emotion path
    'EMOTION': {'pos': (0.5, 0.7), 'text': 'Emotion/<br>expression<br>needed?', 'color': '#1FB8CD', 'type': 'decision'},
    
    # Level 2 - Scale path
    'SCALE': {'pos': (0.8, 0.7), 'text': 'Scale<br>(DAU)?', 'color': '#1FB8CD', 'type': 'decision'},
    
    # Level 3 - Recommendations
    'REC1': {'pos': (0.08, 0.4), 'text': 'ElevenLabs<br>Flash 2.5 +<br>Wav2Lip<br>(75ms+250ms)', 'color': '#2E8B57', 'type': 'recommended'},
    'REC2': {'pos': (0.22, 0.4), 'text': 'Coqui XTTS<br>v2 + Wav2Lip<br>(150ms+250ms)', 'color': '#D2BA4C', 'type': 'alternative'},
    'REC3': {'pos': (0.18, 0.4), 'text': 'SadTalker<br>for cinematics<br>(400ms)', 'color': '#2E8B57', 'type': 'recommended'},
    'REC4': {'pos': (0.32, 0.4), 'text': 'Any<br>commercial<br>TTS OK', 'color': '#D2BA4C', 'type': 'alternative'},
    'REC5': {'pos': (0.44, 0.55), 'text': 'SadTalker +<br>ElevenLabs<br>(Emotional)', 'color': '#2E8B57', 'type': 'recommended'},
    'REC6': {'pos': (0.56, 0.55), 'text': 'Wav2Lip +<br>Smallest.ai<br>(Fast/cheap)', 'color': '#D2BA4C', 'type': 'alternative'},
    'REC7': {'pos': (0.7, 0.55), 'text': 'Self-hosted<br>Coqui + GPU<br>($180/mo)', 'color': '#5D878F', 'type': 'scale'},
    'REC8': {'pos': (0.8, 0.55), 'text': 'Hybrid<br>ElevenLabs +<br>Coqui<br>($650/mo)', 'color': '#2E8B57', 'type': 'recommended'},
    'REC9': {'pos': (0.9, 0.55), 'text': 'Multi-provider<br>strategy<br>($50K+/mo)', 'color': '#5D878F', 'type': 'scale'},
}

# Define edges (connections)
edges = [
    ('START', 'PRIORITY'),
    ('PRIORITY', 'LATENCY'),
    ('PRIORITY', 'EMOTION'),
    ('PRIORITY', 'SCALE'),
    ('LATENCY', 'BUDGET'),
    ('LATENCY', 'QUALITY'),
    ('BUDGET', 'REC1'),
    ('BUDGET', 'REC2'),
    ('QUALITY', 'REC3'),
    ('QUALITY', 'REC4'),
    ('EMOTION', 'REC5'),
    ('EMOTION', 'REC6'),
    ('SCALE', 'REC7'),
    ('SCALE', 'REC8'),
    ('SCALE', 'REC9'),
]

# Draw edges
for start, end in edges:
    x0, y0 = nodes[start]['pos']
    x1, y1 = nodes[end]['pos']
    
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines',
        line=dict(color='#13343B', width=2),
        hoverinfo='skip',
        showlegend=False
    ))

# Draw nodes
for node_id, node_data in nodes.items():
    x, y = node_data['pos']
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='markers+text',
        marker=dict(
            size=60,
            color=node_data['color'],
            line=dict(color='#13343B', width=2)
        ),
        text=node_data['text'],
        textposition='middle center',
        textfont=dict(size=9, color='#fff' if node_data['color'] != '#D2BA4C' else '#000'),
        hoverinfo='text',
        hovertext=node_data['text'].replace('<br>', ' '),
        showlegend=False
    ))

# Add legend manually
legend_items = [
    {'name': 'Recommended', 'color': '#2E8B57'},
    {'name': 'Alternative', 'color': '#D2BA4C'},
    {'name': 'Decision', 'color': '#1FB8CD'},
    {'name': 'Scale Option', 'color': '#5D878F'},
]

for i, item in enumerate(legend_items):
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=12, color=item['color']),
        name=item['name'],
        showlegend=True
    ))

fig.update_layout(
    title='TTS + Lip-Sync Decision Tree',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.25, 1.05]),
    plot_bgcolor='#F3F3EE',
    paper_bgcolor='#F3F3EE',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)

fig.update_traces(cliponaxis=False)

# Save the figure
fig.write_image('tts_decision_tree.png')
fig.write_image('tts_decision_tree.svg', format='svg')

print("Chart saved successfully")
