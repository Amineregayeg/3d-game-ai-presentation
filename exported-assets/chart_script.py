
import plotly.graph_objects as go

# Data from JSON in the order specified in instructions
data = {
    "tts_solutions": [
        {"name": "ElevenLabs Flash 2.5", "latency_ttfb": 75, "mos_score": 4.14, "cost_per_1m": 165, "streaming": True},
        {"name": "Smallest.ai Lightning", "latency_ttfb": 187, "mos_score": 4.0, "cost_per_1m": 9.5, "streaming": True},
        {"name": "Coqui XTTS v2", "latency_ttfb": 150, "mos_score": 4.5, "cost_per_1m": 0, "streaming": True},  # Capped MOS at 4.5
        {"name": "Play.ht", "latency_ttfb": 150, "mos_score": 4.0, "cost_per_1m": 12, "streaming": True},
        {"name": "OpenAI TTS", "latency_ttfb": 200, "mos_score": 3.4, "cost_per_1m": 15, "streaming": False},
        {"name": "Google Cloud TTS", "latency_ttfb": 150, "mos_score": 3.8, "cost_per_1m": 16, "streaming": True},
        {"name": "AWS Polly", "latency_ttfb": 200, "mos_score": 3.1, "cost_per_1m": 4, "streaming": True},
        {"name": "Speechmatics", "latency_ttfb": 180, "mos_score": 4.2, "cost_per_1m": 11, "streaming": True}
    ]
}

solutions = data["tts_solutions"]

# Assign cost tiers based on cost: green=cheap (<$10), yellow=medium ($10-$50), red=expensive (>$50)
def get_cost_tier_and_color(cost):
    if cost < 10:
        return "Cheap", "#2E8B57"  # Green
    elif cost < 50:
        return "Medium", "#D2BA4C"  # Yellow
    else:
        return "Expensive", "#DB4545"  # Red

# Create shortened names for X-axis (under 15 chars)
short_names = []
for s in solutions:
    name = s["name"]
    if "ElevenLabs" in name:
        short_names.append("ElevenLabs")
    elif "Smallest.ai" in name:
        short_names.append("Smallest.ai")
    elif "Coqui" in name:
        short_names.append("Coqui XTTS")
    elif "Play.ht" in name:
        short_names.append("Play.ht")
    elif "OpenAI" in name:
        short_names.append("OpenAI")
    elif "Google" in name:
        short_names.append("Google TTS")
    elif "AWS" in name:
        short_names.append("AWS Polly")
    elif "Speechmatics" in name:
        short_names.append("Speechmatics")
    else:
        short_names.append(name[:15])

# Group by cost tier
cost_tier_groups = {"Cheap": [], "Medium": [], "Expensive": []}
cost_tier_colors = {"Cheap": "#2E8B57", "Medium": "#D2BA4C", "Expensive": "#DB4545"}

for i, s in enumerate(solutions):
    tier, color = get_cost_tier_and_color(s["cost_per_1m"])
    cost_tier_groups[tier].append(i)

# Create figure
fig = go.Figure()

# Add bars grouped by cost tier for proper legend
for tier in ["Cheap", "Medium", "Expensive"]:
    if not cost_tier_groups[tier]:
        continue
    
    indices = cost_tier_groups[tier]
    x_vals = [short_names[i] for i in indices]
    y_vals = [solutions[i]["latency_ttfb"] for i in indices]
    
    # Create detailed hover text with all specs
    hover_texts = []
    for i in indices:
        s = solutions[i]
        stream_mark = "✓" if s["streaming"] else "✗"
        hover_texts.append(
            f"{s['name']}<br>" +
            f"Latency: {s['latency_ttfb']}ms<br>" +
            f"MOS: {s['mos_score']}<br>" +
            f"Stream: {stream_mark}<br>" +
            f"Cost: ${s['cost_per_1m']}/1M"
        )
    
    # Add text labels on bars
    text_labels = []
    for i in indices:
        s = solutions[i]
        stream_mark = "✓" if s["streaming"] else "✗"
        text_labels.append(
            f"{s['latency_ttfb']}ms<br>MOS:{s['mos_score']}<br>{stream_mark}<br>${s['cost_per_1m']}"
        )
    
    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        name=f"{tier} Cost",
        marker=dict(color=cost_tier_colors[tier]),
        hovertext=hover_texts,
        hoverinfo="text",
        text=text_labels,
        textposition="outside",
        textfont=dict(size=9),
        yaxis="y",
        showlegend=True
    ))

# Add MOS score line on secondary axis
mos_x = short_names
mos_y = [s["mos_score"] for s in solutions]

fig.add_trace(go.Scatter(
    x=mos_x,
    y=mos_y,
    name="MOS Score",
    mode="lines+markers",
    line=dict(color="#1FB8CD", width=3),
    marker=dict(size=10, color="#1FB8CD"),
    yaxis="y2",
    hovertemplate="%{x}<br>MOS: %{y:.2f}<extra></extra>"
))

# Update layout with dual y-axes
fig.update_layout(
    title="TTS Solutions for Gaming",
    xaxis=dict(title="TTS Solution"),
    yaxis=dict(
        title="Latency (ms)",
        side="left",
        range=[0, 250]
    ),
    yaxis2=dict(
        title="MOS Score",
        overlaying="y",
        side="right",
        range=[0, 5]
    ),
    barmode='group'
)

fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image("tts_comparison.png")
fig.write_image("tts_comparison.svg", format="svg")
