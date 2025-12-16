
import plotly.graph_objects as go
import pandas as pd
import json

# Load the data
data = {"lip_sync_solutions": [{"name": "Wav2Lip", "accuracy": 95, "latency": 250, "production_readiness": 95, "type": "Production", "emotion_control": False, "head_movement": False, "multilingual": True, "best_for": "Real-time dialogue", "use_case": "Gaming standard"}, {"name": "SadTalker", "accuracy": 92, "latency": 400, "production_readiness": 90, "type": "Production", "emotion_control": True, "head_movement": True, "multilingual": True, "best_for": "Emotional cutscenes", "use_case": "Cinematic narrative"}, {"name": "UE5 MetaHuman", "accuracy": 90, "latency": 150, "production_readiness": 85, "type": "Enterprise", "emotion_control": False, "head_movement": True, "multilingual": True, "best_for": "In-engine avatars", "use_case": "AAA games"}, {"name": "First Order Motion", "accuracy": 88, "latency": 75, "production_readiness": 60, "type": "Research", "emotion_control": False, "head_movement": True, "multilingual": False, "best_for": "Reference-driven", "use_case": "Procedural animation"}, {"name": "Neural Lip-Sync (Generic)", "accuracy": 93, "latency": 200, "production_readiness": 80, "type": "Production", "emotion_control": False, "head_movement": False, "multilingual": True, "best_for": "General-purpose", "use_case": "Any game"}]}

df = pd.DataFrame(data['lip_sync_solutions'])

# Color mapping
color_map = {
    'Production': '#2E8B57',  # Green
    'Enterprise': '#D2BA4C',  # Yellow
    'Research': '#DB4545'     # Red
}

# Create figure
fig = go.Figure()

# Add background zones with better visibility
# Real-time capable zone (latency < 200ms)
fig.add_shape(
    type="rect",
    x0=85, x1=100,
    y0=0, y1=200,
    fillcolor="#2E8B57",
    opacity=0.1,
    layer="below",
    line=dict(color="#2E8B57", width=2, dash="dash"),
)

# Batch processing zone (latency > 250ms)
fig.add_shape(
    type="rect",
    x0=85, x1=100,
    y0=250, y1=500,
    fillcolor="#D2BA4C",
    opacity=0.1,
    layer="below",
    line=dict(color="#D2BA4C", width=2, dash="dash"),
)

# Best for gaming zone (high accuracy 90%+, low latency <200ms)
fig.add_shape(
    type="rect",
    x0=89.5, x1=100,
    y0=0, y1=200,
    fillcolor="#1FB8CD",
    opacity=0.15,
    layer="below",
    line=dict(color="#1FB8CD", width=2),
)

# Best for cinematic zone (high accuracy 91%+, accepts higher latency)
fig.add_shape(
    type="rect",
    x0=91, x1=100,
    y0=300, y1=500,
    fillcolor="#5D878F",
    opacity=0.15,
    layer="below",
    line=dict(color="#5D878F", width=2),
)

# Add zone labels as text traces
fig.add_trace(go.Scatter(
    x=[92.5, 92.5, 94.5, 96],
    y=[100, 375, 100, 400],
    mode='text',
    text=['Real-time<br>Capable', 'Batch<br>Processing', 'Best for<br>Gaming', 'Best for<br>Cinematic'],
    textfont=dict(size=11, color='gray'),
    showlegend=False,
    hoverinfo='skip'
))

# Group by type for legend
for solution_type in ['Production', 'Enterprise', 'Research']:
    df_type = df[df['type'] == solution_type]
    
    # Create hover text with features
    hover_text = []
    display_text = []
    for idx, row in df_type.iterrows():
        features = []
        if row['emotion_control']:
            features.append("Emotion")
        if row['head_movement']:
            features.append("Head Move")
        if row['multilingual']:
            features.append("Multilingual")
        
        features_str = ", ".join(features) if features else "Basic"
        feature_icons = ""
        if row['emotion_control']:
            feature_icons += "😊"
        if row['head_movement']:
            feature_icons += "↔️"
        if row['multilingual']:
            feature_icons += "🌐"
        
        hover_text.append(
            f"<b>{row['name']}</b><br>" +
            f"Accuracy: {row['accuracy']}%<br>" +
            f"Latency: {row['latency']}ms<br>" +
            f"Readiness: {row['production_readiness']}/100<br>" +
            f"Features: {features_str}<br>" +
            f"Best for: {row['best_for']}<br>" +
            f"Use: {row['use_case']}"
        )
        
        # Add feature icons to name
        name_with_features = row['name'].replace("Neural Lip-Sync (Generic)", "Neural Lip-Sync")
        if feature_icons:
            display_text.append(f"{name_with_features}<br>{feature_icons}")
        else:
            display_text.append(name_with_features)
    
    fig.add_trace(go.Scatter(
        x=df_type['accuracy'],
        y=df_type['latency'],
        mode='markers+text',
        name=solution_type,
        marker=dict(
            size=df_type['production_readiness'],
            color=color_map[solution_type],
            sizemode='diameter',
            sizeref=1.8,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        text=display_text,
        textposition='top center',
        textfont=dict(size=11),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=True
    ))

# Add reference bubbles for size legend
fig.add_trace(go.Scatter(
    x=[86.5, 86.5, 86.5],
    y=[450, 420, 390],
    mode='markers+text',
    marker=dict(
        size=[95, 80, 60],
        color='lightgray',
        sizemode='diameter',
        sizeref=1.8,
        line=dict(width=1, color='gray'),
        opacity=0.5
    ),
    text=['Large<br>(95)', 'Medium<br>(80)', 'Small<br>(60)'],
    textposition='middle right',
    textfont=dict(size=9, color='gray'),
    name='Readiness',
    showlegend=True,
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    title="Lip-Sync Solution Comparison",
    xaxis_title="Accuracy (%)",
    yaxis_title="Latency (ms)",
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5,
        title=dict(text='Type & Bubble Size = Production Readiness')
    ),
    xaxis=dict(
        range=[85, 100],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        dtick=2
    ),
    yaxis=dict(
        range=[0, 500],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        dtick=100
    )
)

fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image("lipsync_comparison.png")
fig.write_image("lipsync_comparison.svg", format="svg")
