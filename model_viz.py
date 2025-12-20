"""
Model Visualization Script

Generates detailed visual diagrams of all model architectures using matplotlib.
Each diagram shows:
- Layer types with color coding
- Input/output shapes at each stage
- Parameter counts
- Data flow arrows
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Color scheme for different layer types
COLORS = {
    'input': '#E3F2FD',       # Light blue
    'embedding': '#E8F5E9',    # Light green
    'lstm': '#FFF3E0',         # Light orange
    'attention': '#FCE4EC',    # Light pink
    'dense': '#E1F5FE',        # Cyan
    'output': '#C8E6C9',       # Green
    'dropout': '#F5F5F5',      # Light gray
    'norm': '#FFF9C4',         # Light yellow
    'transformer': '#E1BEE7',  # Light purple
    'concat': '#FFECB3',       # Amber
}

BORDER_COLORS = {
    'input': '#1976D2',
    'embedding': '#388E3C',
    'lstm': '#F57C00',
    'attention': '#C2185B',
    'dense': '#0288D1',
    'output': '#2E7D32',
    'dropout': '#9E9E9E',
    'norm': '#F9A825',
    'transformer': '#7B1FA2',
    'concat': '#FF8F00',
}


def create_layer_box(ax, x, y, width, height, layer_type, name, shape=None, params=None, details=None):
    """Create a styled layer box with label."""
    color = COLORS.get(layer_type, '#FFFFFF')
    border_color = BORDER_COLORS.get(layer_type, '#000000')
    
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2
    )
    ax.add_patch(box)
    
    # Layer name (bold)
    text_y = y + height * 0.7 if shape or details else y + height * 0.5
    ax.text(x + width/2, text_y, name, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#333333')
    
    # Shape info
    if shape:
        ax.text(x + width/2, y + height * 0.35, shape, ha='center', va='center',
                fontsize=8, color='#666666', style='italic')
    
    # Parameter count or details
    if params:
        ax.text(x + width/2, y + height * 0.15, f'{params:,} params', ha='center', va='center',
                fontsize=7, color='#888888')
    elif details:
        ax.text(x + width/2, y + height * 0.15, details, ha='center', va='center',
                fontsize=7, color='#888888')


def draw_arrow(ax, start, end, color='#666666', style='->', curved=False):
    """Draw an arrow between two points."""
    if curved:
        connectionstyle = "arc3,rad=0.2"
    else:
        connectionstyle = "arc3,rad=0"
    
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5,
                               connectionstyle=connectionstyle))


def draw_multi_arrow(ax, start, ends, color='#666666'):
    """Draw arrows from one point to multiple endpoints."""
    for end in ends:
        draw_arrow(ax, start, end, color)


def create_legend(ax, layer_types):
    """Create a legend for the layer types used."""
    handles = []
    for lt in layer_types:
        patch = mpatches.Patch(
            facecolor=COLORS.get(lt, '#FFFFFF'),
            edgecolor=BORDER_COLORS.get(lt, '#000000'),
            linewidth=2,
            label=lt.replace('_', ' ').title()
        )
        handles.append(patch)
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)


# ============================================================================
# ORIGINAL MODEL (Seq2Seq with Attention)
# ============================================================================

def visualize_original_model():
    """Create detailed architecture diagram for the Original ILotto model."""
    fig, ax = plt.subplots(figsize=(18, 22))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Title
    ax.text(9, 21.5, 'Original Model: Seq2Seq with Bahdanau Attention', 
            fontsize=16, ha='center', fontweight='bold')
    ax.text(9, 20.8, 'Bidirectional LSTM Encoder + LSTM Decoder with Additive Attention',
            fontsize=11, ha='center', color='#666666')
    
    # ========== INPUT SECTION ==========
    ax.text(9, 20, '━━━━━━━━━━━━━━━ INPUT ━━━━━━━━━━━━━━━', ha='center', fontsize=10, color='#1976D2')
    
    create_layer_box(ax, 6.5, 18.5, 5, 1.2, 'input', 'Input Layer',
                    shape='(batch, 10, 7)', details='10 draws × 7 numbers')
    
    # ========== EMBEDDING SECTION ==========
    ax.text(9, 17.8, '━━━━━━━━━━━━━━━ EMBEDDING ━━━━━━━━━━━━━━━', ha='center', fontsize=10, color='#388E3C')
    
    # 7 separate embeddings
    embed_width = 1.8
    embed_start = 2
    for i in range(7):
        x = embed_start + i * 2
        label = f'Emb {i+1}' if i < 6 else 'Emb Bonus'
        create_layer_box(ax, x, 16, embed_width, 1.2, 'embedding', label,
                        shape='37→30' if i < 6 else '7→30', params=1110 if i < 6 else 210)
    
    # SpatialDropout
    for i in range(7):
        x = embed_start + i * 2
        create_layer_box(ax, x, 14.5, embed_width, 0.8, 'dropout', 'SpatialDrop',
                        details='rate=0.5')
    
    # Concatenate
    create_layer_box(ax, 5, 13, 8, 0.9, 'concat', 'Concatenate',
                    shape='(batch, 10, 210)', details='7 × 30 = 210 features')
    
    # Arrows from embeddings to concat
    for i in range(7):
        x = embed_start + i * 2 + embed_width/2
        draw_arrow(ax, (x, 14.5), (9, 13.9))
    
    # ========== ENCODER SECTION ==========
    ax.text(9, 12.3, '━━━━━━━━━━━━━━━ ENCODER ━━━━━━━━━━━━━━━', ha='center', fontsize=10, color='#F57C00')
    
    # Bidirectional LSTM 1
    create_layer_box(ax, 3, 10.5, 5.5, 1.4, 'lstm', 'Bidirectional LSTM 1',
                    shape='→ (batch, 10, 128)', params=87552)
    ax.text(5.75, 10.2, 'Forward: 64 units', fontsize=7, ha='center', color='#666')
    
    create_layer_box(ax, 9.5, 10.5, 5.5, 1.4, 'lstm', 'Bidirectional LSTM 1',
                    shape='← (batch, 10, 128)', params=87552)
    ax.text(12.25, 10.2, 'Backward: 64 units', fontsize=7, ha='center', color='#666')
    
    # Concat states
    create_layer_box(ax, 5, 9, 8, 0.8, 'concat', 'Concat States',
                    shape='h: (batch, 128), c: (batch, 128)')
    
    # Bidirectional LSTM 2
    create_layer_box(ax, 3, 7.5, 5.5, 1.2, 'lstm', 'Bidirectional LSTM 2',
                    shape='→ (batch, 10, 64)', params=24832)
    ax.text(5.75, 7.2, 'Forward: 32 units', fontsize=7, ha='center', color='#666')
    
    create_layer_box(ax, 9.5, 7.5, 5.5, 1.2, 'lstm', 'Bidirectional LSTM 2',
                    shape='← (batch, 10, 64)', params=24832)
    ax.text(12.25, 7.2, 'Backward: 32 units', fontsize=7, ha='center', color='#666')
    
    # Final encoder states
    create_layer_box(ax, 5, 6, 8, 0.8, 'concat', 'Encoder Output',
                    shape='seq: (batch, 10, 64), h: (batch, 64), c: (batch, 64)')
    
    # ========== DECODER SECTION ==========
    ax.text(9, 5.3, '━━━━━━━━━━━━━━━ DECODER ━━━━━━━━━━━━━━━', ha='center', fontsize=10, color='#C2185B')
    
    # RepeatVector
    create_layer_box(ax, 6, 4, 6, 0.9, 'dense', 'RepeatVector(7)',
                    shape='(batch, 7, 64)', details='Repeat h 7 times')
    
    # Decoder LSTMs
    create_layer_box(ax, 6, 2.8, 6, 0.9, 'lstm', 'LSTM Decoder 1',
                    shape='(batch, 7, 128)', params=98816)
    ax.text(9, 2.5, 'init with h1, c1 from encoder', fontsize=7, ha='center', color='#666')
    
    create_layer_box(ax, 6, 1.6, 6, 0.9, 'lstm', 'LSTM Decoder 2',
                    shape='(batch, 7, 64)', params=49408)
    ax.text(9, 1.3, 'init with h2, c2 from encoder', fontsize=7, ha='center', color='#666')
    
    # Attention
    create_layer_box(ax, 12.5, 2.2, 4.5, 1.5, 'attention', 'Bahdanau\nAttention',
                    shape='(batch, 7, 64)', params=8256)
    ax.text(14.75, 1.9, 'Additive attention', fontsize=7, ha='center', color='#666')
    ax.text(14.75, 1.6, 'Query: decoder', fontsize=6, ha='center', color='#888')
    ax.text(14.75, 1.4, 'Key/Value: encoder', fontsize=6, ha='center', color='#888')
    
    # Arrow from encoder to attention
    draw_arrow(ax, (13, 6), (14.75, 3.7), curved=True)
    
    # Concat decoder + attention
    create_layer_box(ax, 6, 0.3, 6, 0.8, 'concat', 'Concat',
                    shape='(batch, 7, 128)', details='decoder + context')
    
    # ========== OUTPUT SECTION ==========
    # Dense output
    create_layer_box(ax, 0.5, 0.3, 4.5, 0.8, 'output', 'Dense Output',
                    shape='(batch, 7, 37)', params=4773)
    ax.text(2.75, 0, 'Softmax activation', fontsize=7, ha='center', color='#666')
    
    # Arrows
    draw_arrow(ax, (9, 18.5), (9, 17.2))
    draw_arrow(ax, (9, 13), (9, 11.9))
    draw_arrow(ax, (9, 9), (9, 8.7))
    draw_arrow(ax, (9, 6), (9, 4.9))
    draw_arrow(ax, (9, 2.8), (9, 2.5))
    draw_arrow(ax, (12, 2.2), (12.5, 2.2))
    draw_arrow(ax, (12, 1.6), (12, 1.1))
    draw_arrow(ax, (6, 0.7), (5, 0.7))
    
    # Parameter summary
    total_params = 1110*6 + 210 + 87552*2 + 24832*2 + 98816 + 49408 + 8256 + 4773
    ax.text(1, 21, f'Total Parameters: ~{total_params:,}', fontsize=10, color='#333')
    
    # Legend
    create_legend(ax, ['input', 'embedding', 'lstm', 'attention', 'concat', 'output', 'dropout'])
    
    plt.tight_layout()
    os.makedirs("model/diagrams", exist_ok=True)
    plt.savefig("model/diagrams/original_architecture_detailed.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/original_architecture_detailed.png")


# ============================================================================
# MULTI-OUTPUT MODEL
# ============================================================================

def visualize_multi_output_model():
    """Create detailed architecture diagram for the Multi-Output model."""
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(9, 15.5, 'Multi-Output Model: Shared Encoder + Independent Heads', 
            fontsize=16, ha='center', fontweight='bold')
    ax.text(9, 14.9, 'Each ball position has its own Dense output layer to prevent mode collapse',
            fontsize=11, ha='center', color='#666666')
    
    # ========== INPUT ==========
    create_layer_box(ax, 6.5, 13.5, 5, 1, 'input', 'Input Layer',
                    shape='(batch, 10, 7)', details='10 draws × 7 numbers')
    
    # ========== EMBEDDINGS ==========
    create_layer_box(ax, 3.5, 11.8, 5, 1, 'embedding', 'Main Ball Embedding',
                    shape='37 → 32 dim', params=1184)
    create_layer_box(ax, 9.5, 11.8, 5, 1, 'embedding', 'Bonus Ball Embedding',
                    shape='7 → 32 dim', params=224)
    
    # Concat
    create_layer_box(ax, 5.5, 10.3, 7, 0.9, 'concat', 'Concatenate',
                    shape='(batch, 10, 224)', details='6×32 + 32 = 224')
    
    # ========== LSTM ENCODER ==========
    create_layer_box(ax, 5.5, 8.8, 7, 1.1, 'lstm', 'LSTM Layer 1',
                    shape='(batch, 10, 64)', params=74240)
    ax.text(9, 8.5, 'return_sequences=True, dropout=0.3', fontsize=7, ha='center', color='#666')
    
    create_layer_box(ax, 5.5, 7.2, 7, 1.1, 'lstm', 'LSTM Layer 2',
                    shape='(batch, 64)', params=33024)
    ax.text(9, 6.9, 'return_sequences=False, dropout=0.3', fontsize=7, ha='center', color='#666')
    
    # Dropout
    create_layer_box(ax, 6.5, 5.8, 5, 0.8, 'dropout', 'Dropout',
                    details='rate=0.3')
    
    # ========== OUTPUT HEADS ==========
    ax.text(9, 5.2, '━━━━━━━━━━━━ INDEPENDENT OUTPUT HEADS ━━━━━━━━━━━━', 
            ha='center', fontsize=10, color='#2E7D32')
    
    # 7 output heads
    head_width = 2.0
    head_start = 1.5
    for i in range(7):
        x = head_start + i * 2.2
        if i < 6:
            label = f'Ball {i+1}'
            shape = '(batch, 37)'
            params = 2405
            details = 'Softmax'
        else:
            label = 'Bonus'
            shape = '(batch, 7)'
            params = 455
            details = 'Softmax'
        create_layer_box(ax, x, 3.5, head_width, 1.3, 'output', label,
                        shape=shape, params=params)
        ax.text(x + head_width/2, 3.2, details, fontsize=7, ha='center', color='#666')
    
    # Stack outputs
    create_layer_box(ax, 5.5, 1.8, 7, 0.9, 'concat', 'Stack Outputs',
                    shape='(batch, 7, 37)', details='Bonus padded to 37')
    
    # Final output
    create_layer_box(ax, 6.5, 0.5, 5, 0.8, 'output', 'Final Output',
                    shape='(batch, 7, 37)')
    
    # Arrows
    draw_arrow(ax, (9, 13.5), (9, 12.8))
    draw_arrow(ax, (6, 11.8), (6, 11.2))
    draw_arrow(ax, (12, 11.8), (12, 11.2))
    draw_arrow(ax, (9, 10.3), (9, 9.9))
    draw_arrow(ax, (9, 8.8), (9, 8.3))
    draw_arrow(ax, (9, 7.2), (9, 6.6))
    draw_arrow(ax, (9, 5.8), (9, 5.5))
    
    # Fan out to heads
    for i in range(7):
        x = head_start + i * 2.2 + head_width/2
        draw_arrow(ax, (9, 5.5), (x, 4.8))
    
    # Fan in from heads
    for i in range(7):
        x = head_start + i * 2.2 + head_width/2
        draw_arrow(ax, (x, 3.5), (9, 2.7))
    
    draw_arrow(ax, (9, 1.8), (9, 1.3))
    
    # Parameter summary
    total_params = 1184 + 224 + 74240 + 33024 + 2405*6 + 455
    ax.text(1, 15, f'Total Parameters: ~{total_params:,}', fontsize=10, color='#333')
    
    # Key insight box
    insight_box = FancyBboxPatch((12.5, 6.5), 5, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(15, 8.5, 'Key Insight', ha='center', fontweight='bold', fontsize=10)
    ax.text(15, 7.8, 'Each Dense head learns', ha='center', fontsize=8)
    ax.text(15, 7.4, 'independently, preventing', ha='center', fontsize=8)
    ax.text(15, 7.0, 'mode collapse where all', ha='center', fontsize=8)
    ax.text(15, 6.6, 'positions predict same #', ha='center', fontsize=8)
    
    # Legend
    create_legend(ax, ['input', 'embedding', 'lstm', 'dropout', 'output', 'concat'])
    
    plt.tight_layout()
    plt.savefig("model/diagrams/multi_output_architecture_detailed.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/multi_output_architecture_detailed.png")


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

def visualize_transformer_model():
    """Create detailed architecture diagram for the Transformer model."""
    fig, ax = plt.subplots(figsize=(18, 22))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Title
    ax.text(9, 21.5, 'Transformer Model: Self-Attention Based Architecture', 
            fontsize=16, ha='center', fontweight='bold')
    ax.text(9, 20.9, 'Multi-Head Self-Attention captures relationships between all numbers',
            fontsize=11, ha='center', color='#666666')
    
    # ========== INPUT ==========
    create_layer_box(ax, 6.5, 19.5, 5, 1, 'input', 'Input Layer',
                    shape='(batch, 10, 7)', details='10 draws × 7 numbers')
    
    # Flatten
    create_layer_box(ax, 6.5, 18, 5, 0.9, 'dense', 'Flatten',
                    shape='(batch, 70)', details='10 × 7 = 70 positions')
    
    # ========== EMBEDDINGS ==========
    create_layer_box(ax, 3, 16.3, 5, 1.1, 'embedding', 'Number Embedding',
                    shape='37 → 64 dim', params=2368)
    create_layer_box(ax, 10, 16.3, 5, 1.1, 'embedding', 'Position Embedding',
                    shape='70 → 64 dim', params=4480)
    
    # Add
    create_layer_box(ax, 6, 14.8, 6, 0.8, 'concat', 'Add Embeddings',
                    shape='(batch, 70, 64)')
    
    # ========== TRANSFORMER BLOCK 1 ==========
    ax.text(9, 14, '━━━━━━━━━━━ TRANSFORMER BLOCK 1 ━━━━━━━━━━━', ha='center', fontsize=10, color='#7B1FA2')
    
    # Multi-head attention
    create_layer_box(ax, 5, 12, 8, 1.8, 'transformer', 'Multi-Head Self-Attention',
                    shape='(batch, 70, 64)', params=16640)
    ax.text(9, 11.6, '4 heads, key_dim=64', fontsize=8, ha='center', color='#666')
    ax.text(9, 11.3, 'Q, K, V from same input', fontsize=7, ha='center', color='#888')
    
    # Attention heads visualization
    head_colors = ['#E57373', '#64B5F6', '#81C784', '#FFD54F']
    for i, c in enumerate(head_colors):
        rect = Rectangle((5.5 + i*1.8, 12.8), 1.5, 0.4, facecolor=c, alpha=0.7)
        ax.add_patch(rect)
        ax.text(6.25 + i*1.8, 13, f'H{i+1}', ha='center', va='center', fontsize=7)
    
    # Add & Norm 1
    create_layer_box(ax, 5.5, 10.3, 3, 0.8, 'dropout', 'Dropout', details='rate=0.2')
    create_layer_box(ax, 9.5, 10.3, 3.5, 0.8, 'norm', 'Add & LayerNorm')
    
    # Feed Forward
    create_layer_box(ax, 5, 8.8, 8, 1.2, 'dense', 'Feed Forward Network',
                    shape='(batch, 70, 64)', params=16640)
    ax.text(9, 8.5, 'Dense(128, GELU) → Dropout → Dense(64)', fontsize=8, ha='center', color='#666')
    
    # Add & Norm 2
    create_layer_box(ax, 5.5, 7.5, 3, 0.8, 'dropout', 'Dropout', details='rate=0.2')
    create_layer_box(ax, 9.5, 7.5, 3.5, 0.8, 'norm', 'Add & LayerNorm')
    
    # ========== TRANSFORMER BLOCK 2 ==========
    ax.text(9, 6.8, '━━━━━━━━━━━ TRANSFORMER BLOCK 2 ━━━━━━━━━━━', ha='center', fontsize=10, color='#7B1FA2')
    
    create_layer_box(ax, 5, 5, 8, 1.4, 'transformer', 'Multi-Head Self-Attention',
                    shape='(batch, 70, 64)', params=16640)
    ax.text(9, 4.7, '4 heads, same structure as Block 1', fontsize=8, ha='center', color='#666')
    
    create_layer_box(ax, 5, 3.3, 8, 1.2, 'dense', 'Feed Forward + Add & Norm',
                    shape='(batch, 70, 64)', params=16640)
    
    # ========== OUTPUT ==========
    create_layer_box(ax, 5.5, 2, 7, 0.9, 'dense', 'Global Average Pooling',
                    shape='(batch, 64)', details='Pool over 70 positions')
    
    create_layer_box(ax, 5.5, 0.8, 7, 0.8, 'dense', 'Dense(256, GELU) + Dropout',
                    shape='(batch, 256)', params=16640)
    
    # Output heads
    head_width = 2.0
    head_y = -0.7
    for i in range(7):
        x = 1.5 + i * 2.2
        if i < 6:
            label = f'Ball {i+1}'
            params = 9509
        else:
            label = 'Bonus'
            params = 1799
        create_layer_box(ax, x, head_y, head_width, 0.9, 'output', label,
                        params=params)
    
    # Arrows
    draw_arrow(ax, (9, 19.5), (9, 18.9))
    draw_arrow(ax, (9, 18), (9, 17.4))
    draw_arrow(ax, (5.5, 16.3), (5.5, 15.6))
    draw_arrow(ax, (12.5, 16.3), (12.5, 15.6))
    draw_arrow(ax, (9, 14.8), (9, 13.8))
    draw_arrow(ax, (9, 12), (9, 11.1))
    draw_arrow(ax, (9, 10.3), (9, 10))
    draw_arrow(ax, (9, 8.8), (9, 8.3))
    draw_arrow(ax, (9, 7.5), (9, 6.4))
    draw_arrow(ax, (9, 5), (9, 4.5))
    draw_arrow(ax, (9, 3.3), (9, 2.9))
    draw_arrow(ax, (9, 2), (9, 1.6))
    
    # Fan out
    for i in range(7):
        x = 1.5 + i * 2.2 + head_width/2
        draw_arrow(ax, (9, 0.8), (x, 0.2))
    
    # Parameter summary
    total_params = 2368 + 4480 + 16640*4 + 9509*6 + 1799
    ax.text(1, 21, f'Total Parameters: ~{total_params:,}', fontsize=10, color='#333')
    
    # Self-attention explanation box
    insight_box = FancyBboxPatch((13, 12), 4.5, 3.5, boxstyle="round,pad=0.1",
                                  facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(15.25, 15, 'Self-Attention', ha='center', fontweight='bold', fontsize=10)
    ax.text(15.25, 14.4, 'Each position attends', ha='center', fontsize=8)
    ax.text(15.25, 14, 'to ALL other positions', ha='center', fontsize=8)
    ax.text(15.25, 13.5, 'Attn = softmax(QK^T/√d)V', ha='center', fontsize=8, family='monospace')
    ax.text(15.25, 12.9, 'Captures long-range', ha='center', fontsize=8)
    ax.text(15.25, 12.5, 'dependencies', ha='center', fontsize=8)
    
    # Legend
    create_legend(ax, ['input', 'embedding', 'transformer', 'dense', 'norm', 'dropout', 'output'])
    
    plt.tight_layout()
    plt.savefig("model/diagrams/transformer_architecture_detailed.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/transformer_architecture_detailed.png")


# ============================================================================
# SET PREDICTION MODEL
# ============================================================================

def visualize_set_prediction_model():
    """Create detailed architecture diagram for the Set Prediction model."""
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(8, 13.5, 'Set Prediction Model: Multi-Label Classification', 
            fontsize=16, ha='center', fontweight='bold')
    ax.text(8, 12.9, 'Predicts WHICH numbers will appear, not their positions',
            fontsize=11, ha='center', color='#666666')
    
    # ========== INPUT ==========
    create_layer_box(ax, 5.5, 11.5, 5, 1, 'input', 'Input Layer',
                    shape='(batch, 10, 7)', details='10 draws × 7 numbers')
    
    # Embedding
    create_layer_box(ax, 5.5, 9.8, 5, 1.1, 'embedding', 'Shared Embedding',
                    shape='37 → 32 dim', params=1184)
    ax.text(8, 9.5, 'Same embedding for all positions', fontsize=7, ha='center', color='#666')
    
    # Reshape
    create_layer_box(ax, 5.5, 8.3, 5, 0.9, 'dense', 'Reshape',
                    shape='(batch, 10, 224)', details='7 × 32 = 224')
    
    # LSTM
    create_layer_box(ax, 5.5, 6.8, 5, 1.2, 'lstm', 'LSTM Layer',
                    shape='(batch, 128)', params=181248)
    ax.text(8, 6.5, '128 units, dropout=0.3', fontsize=7, ha='center', color='#666')
    
    # Dense
    create_layer_box(ax, 5.5, 5.3, 5, 0.9, 'dense', 'Dense (ReLU)',
                    shape='(batch, 64)', params=8256)
    
    # Dropout
    create_layer_box(ax, 5.5, 4.1, 5, 0.7, 'dropout', 'Dropout',
                    details='rate=0.3')
    
    # ========== OUTPUTS ==========
    ax.text(8, 3.5, '━━━━━━━━━━━ OUTPUT HEADS ━━━━━━━━━━━', ha='center', fontsize=10, color='#2E7D32')
    
    # Main output (Sigmoid)
    create_layer_box(ax, 1, 1.5, 6.5, 1.5, 'output', 'Main Numbers Output',
                    shape='(batch, 37)', params=2405)
    ax.text(4.25, 1.2, 'SIGMOID activation', fontsize=8, ha='center', color='#666', fontweight='bold')
    ax.text(4.25, 0.8, 'Each output = P(number appears)', fontsize=7, ha='center', color='#888')
    
    # Bonus output (Softmax)
    create_layer_box(ax, 8.5, 1.5, 6.5, 1.5, 'output', 'Bonus Number Output',
                    shape='(batch, 7)', params=455)
    ax.text(11.75, 1.2, 'SOFTMAX activation', fontsize=8, ha='center', color='#666', fontweight='bold')
    ax.text(11.75, 0.8, 'Single number classification', fontsize=7, ha='center', color='#888')
    
    # Arrows
    draw_arrow(ax, (8, 11.5), (8, 10.9))
    draw_arrow(ax, (8, 9.8), (8, 9.2))
    draw_arrow(ax, (8, 8.3), (8, 8))
    draw_arrow(ax, (8, 6.8), (8, 6.2))
    draw_arrow(ax, (8, 5.3), (8, 4.8))
    draw_arrow(ax, (8, 4.1), (8, 3.8))
    draw_arrow(ax, (8, 3.5), (4.25, 3))
    draw_arrow(ax, (8, 3.5), (11.75, 3))
    
    # Key insight boxes
    # Sigmoid explanation
    insight1 = FancyBboxPatch((0.5, -0.3), 7, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(insight1)
    ax.text(4, 0.7, 'Multi-Label: Each number independent', ha='center', fontweight='bold', fontsize=9)
    ax.text(4, 0.3, '[0.9, 0.1, 0.8, 0.05, ...] ← pick top 6', ha='center', fontsize=8, family='monospace')
    ax.text(4, -0.1, 'Numbers can all be high (or low)', ha='center', fontsize=8)
    
    # Softmax explanation
    insight2 = FancyBboxPatch((8, -0.3), 7.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(insight2)
    ax.text(11.75, 0.7, 'Multi-Class: Exactly one winner', ha='center', fontweight='bold', fontsize=9)
    ax.text(11.75, 0.3, '[0.1, 0.05, 0.6, 0.1, 0.1, 0.04, 0.01]', ha='center', fontsize=8, family='monospace')
    ax.text(11.75, -0.1, 'Probabilities sum to 1', ha='center', fontsize=8)
    
    # Why this is conceptually correct
    why_box = FancyBboxPatch((11.5, 7), 4, 4.5, boxstyle="round,pad=0.1",
                              facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=2)
    ax.add_patch(why_box)
    ax.text(13.5, 11, 'Why Set Prediction?', ha='center', fontweight='bold', fontsize=10)
    ax.text(13.5, 10.4, 'Lottery is a SET:', ha='center', fontsize=9)
    ax.text(13.5, 9.9, '{3, 7, 22, 31, 35, 37}', ha='center', fontsize=9, family='monospace')
    ax.text(13.5, 9.3, 'Order does NOT matter!', ha='center', fontsize=9, color='#D32F2F')
    ax.text(13.5, 8.6, 'Other models predict:', ha='center', fontsize=8)
    ax.text(13.5, 8.2, 'position 1 = ?, position 2 = ?', ha='center', fontsize=8)
    ax.text(13.5, 7.7, 'This model predicts:', ha='center', fontsize=8)
    ax.text(13.5, 7.3, 'which 6 of 37 appear', ha='center', fontsize=8, fontweight='bold')
    
    # Parameter summary
    total_params = 1184 + 181248 + 8256 + 2405 + 455
    ax.text(0.5, 13, f'Total Parameters: ~{total_params:,}', fontsize=10, color='#333')
    
    # Legend
    create_legend(ax, ['input', 'embedding', 'lstm', 'dense', 'dropout', 'output'])
    
    plt.tight_layout()
    plt.savefig("model/diagrams/set_prediction_architecture_detailed.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/set_prediction_architecture_detailed.png")


# ============================================================================
# COMPARISON DIAGRAM
# ============================================================================

def visualize_model_comparison():
    """Create a side-by-side comparison of all models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Model Architecture Comparison', fontsize=18, fontweight='bold', y=1.02)
    
    models = [
        ('Original (Seq2Seq)', 'Bidirectional LSTM\n+ Bahdanau Attention', 
         '~500K params', 'High', 'Poor', '#E3F2FD'),
        ('Multi-Output', 'LSTM Encoder\n+ 7 Independent Heads',
         '~125K params', 'Low', 'Medium', '#E8F5E9'),
        ('Transformer', 'Multi-Head Self-Attention\n+ Position Encoding',
         '~150K params', 'Low', 'Medium', '#E1BEE7'),
        ('Set Prediction', 'LSTM + Sigmoid\nMulti-Label Output',
         '~200K params', 'Very Low', 'Best', '#FFF3E0'),
    ]
    
    for ax, (name, arch, params, collapse, fit, color) in zip(axes.flat, models):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Background
        bg = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='#666666', linewidth=2)
        ax.add_patch(bg)
        
        # Title
        ax.text(5, 9, name, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Architecture
        ax.text(5, 7.5, 'Architecture:', ha='center', fontsize=10, fontweight='bold')
        ax.text(5, 6.5, arch, ha='center', fontsize=10, va='center')
        
        # Parameters
        ax.text(5, 5, params, ha='center', fontsize=11)
        
        # Mode collapse
        collapse_color = '#D32F2F' if collapse == 'High' else '#FF9800' if collapse == 'Low' else '#4CAF50'
        ax.text(2.5, 3.5, 'Mode Collapse:', ha='center', fontsize=9)
        ax.text(2.5, 2.8, collapse, ha='center', fontsize=11, color=collapse_color, fontweight='bold')
        
        # Conceptual fit
        fit_color = '#D32F2F' if fit == 'Poor' else '#FF9800' if fit == 'Medium' else '#4CAF50'
        ax.text(7.5, 3.5, 'Conceptual Fit:', ha='center', fontsize=9)
        ax.text(7.5, 2.8, fit, ha='center', fontsize=11, color=fit_color, fontweight='bold')
        
        # Simple architecture sketch (text-based for compatibility)
        if name == 'Original (Seq2Seq)':
            ax.text(5, 1.5, '[In] → LSTM → LSTM → Attn → [Out]', ha='center', fontsize=10, family='monospace')
        elif name == 'Multi-Output':
            ax.text(5, 1.5, '[In] → LSTM → [7 Heads]', ha='center', fontsize=10, family='monospace')
        elif name == 'Transformer':
            ax.text(5, 1.5, '[In] → MHA → MHA → [Out]', ha='center', fontsize=10, family='monospace')
        else:
            ax.text(5, 1.5, '[In] → LSTM → Sigmoid(37)', ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig("model/diagrams/all_models_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/all_models_comparison.png")


# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================

def visualize_attention_pattern():
    """Create a visualization showing how attention works."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simulated attention weights
    np.random.seed(42)
    
    # Bahdanau attention (encoder-decoder)
    ax1 = axes[0]
    attention_weights = np.random.rand(7, 10)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    im1 = ax1.imshow(attention_weights, cmap='Blues', aspect='auto')
    ax1.set_xlabel('Encoder Position (Previous Draws)', fontsize=11)
    ax1.set_ylabel('Decoder Position (Output Ball)', fontsize=11)
    ax1.set_title('Bahdanau Attention (Original Model)\nDecoder attends to encoder sequence', fontsize=12)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([f'Draw\n{10-i}' for i in range(10)], fontsize=8)
    ax1.set_yticks(range(7))
    ax1.set_yticklabels([f'Ball {i+1}' if i < 6 else 'Bonus' for i in range(7)])
    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # Self-attention (transformer)
    ax2 = axes[1]
    self_attn = np.random.rand(70, 70)
    # Make it more diagonal-ish for visualization
    for i in range(70):
        self_attn[i, max(0, i-5):min(70, i+5)] *= 2
    self_attn = self_attn / self_attn.sum(axis=1, keepdims=True)
    
    im2 = ax2.imshow(self_attn[:20, :20], cmap='Purples', aspect='auto')
    ax2.set_xlabel('Key Position', fontsize=11)
    ax2.set_ylabel('Query Position', fontsize=11)
    ax2.set_title('Self-Attention (Transformer Model)\nEvery position attends to every other', fontsize=12)
    ax2.set_xticks([0, 7, 14])
    ax2.set_xticklabels(['Draw 10\nstart', 'Draw 10\nend', 'Draw 9\nstart'])
    ax2.set_yticks([0, 7, 14])
    ax2.set_yticklabels(['Draw 10\nstart', 'Draw 10\nend', 'Draw 9\nstart'])
    plt.colorbar(im2, ax=ax2, label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig("model/diagrams/attention_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/attention_comparison.png")


# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

def visualize_embedding():
    """Visualize how embeddings map numbers to vectors."""
    from sklearn.manifold import TSNE
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create and train a simple embedding
    np.random.seed(42)
    embeddings = np.random.randn(37, 32)
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Before training (random)
    ax1 = axes[0]
    ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                c=np.arange(37), cmap='viridis', s=150, alpha=0.8)
    for i in range(37):
        ax1.annotate(str(i+1), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, ha='center', va='center', color='white', fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax1.set_title('Before Training (Random)\nNumbers scattered randomly', fontsize=12)
    
    # After training (simulated clustering)
    ax2 = axes[1]
    # Simulate some structure - low numbers cluster, high numbers cluster
    trained_embeddings = embeddings.copy()
    for i in range(37):
        # Add bias based on number
        trained_embeddings[i, 0] += i * 0.3
        trained_embeddings[i, 1] += (i % 10) * 0.2
        # Even/odd
        if i % 2 == 0:
            trained_embeddings[i, 2] += 1
    
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=10)
    trained_2d = tsne2.fit_transform(trained_embeddings)
    
    scatter2 = ax2.scatter(trained_2d[:, 0], trained_2d[:, 1],
                           c=np.arange(37), cmap='viridis', s=150, alpha=0.8)
    for i in range(37):
        ax2.annotate(str(i+1), (trained_2d[i, 0], trained_2d[i, 1]),
                    fontsize=8, ha='center', va='center', color='white', fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax2.set_title('After Training (Simulated)\nSimilar numbers cluster together', fontsize=12)
    
    plt.colorbar(scatter2, ax=ax2, label='Number (1-37)')
    
    plt.tight_layout()
    plt.savefig("model/diagrams/embedding_visualization.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/embedding_visualization.png")


# ============================================================================
# DATA FLOW DIAGRAM
# ============================================================================

def visualize_data_flow():
    """Create a diagram showing data flow through the system."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(8, 9.5, 'Data Flow Through ILotto Models', fontsize=18, ha='center', fontweight='bold')
    
    # Input data box
    create_layer_box(ax, 0.5, 7, 3, 1.5, 'input', 'Input Data',
                    shape='(batch, 10, 7)', details='10 draws × 7 balls')
    
    # Arrow
    draw_arrow(ax, (3.5, 7.75), (5, 7.75))
    
    # Embedding box
    create_layer_box(ax, 5, 7, 3, 1.5, 'embedding', 'Embedding',
                    shape='37 → 32 dim', details='Numbers → Vectors')
    
    # Arrow
    draw_arrow(ax, (8, 7.75), (9.5, 7.75))
    
    # Encoder box
    create_layer_box(ax, 9.5, 7, 3, 1.5, 'lstm', 'Encoder',
                    shape='LSTM/Transformer', details='Learn patterns')
    
    # Arrow down
    draw_arrow(ax, (11, 7), (11, 5.5))
    
    # Attention/Processing box
    create_layer_box(ax, 9.5, 4, 3, 1.5, 'attention', 'Attention/Dense',
                    details='Focus on\nrelevant patterns')
    
    # Arrow left
    draw_arrow(ax, (9.5, 4.75), (8, 4.75))
    
    # Output heads box
    create_layer_box(ax, 5, 4, 3, 1.5, 'dense', 'Output Heads',
                    shape='7 Dense layers', details='1 per position')
    
    # Arrow left
    draw_arrow(ax, (5, 4.75), (3.5, 4.75))
    
    # Softmax box
    create_layer_box(ax, 0.5, 4, 3, 1.5, 'output', 'Softmax/Sigmoid',
                    shape='Probabilities', details='For each number')
    
    # Arrow down
    draw_arrow(ax, (2, 4), (2, 2.5))
    
    # Prediction box
    create_layer_box(ax, 0.5, 1, 3, 1.5, 'output', 'Prediction',
                    details='[4, 12, 22,\n33, 35, 37, 5]')
    
    # Example data at each stage
    ax.text(2, 6.3, 'Example:', fontsize=8, ha='center', color='#666')
    ax.text(2, 5.9, '[[7,12,23,31,35,36,3],', fontsize=7, ha='center', family='monospace')
    ax.text(2, 5.6, ' [2,15,22,28,33,37,5],', fontsize=7, ha='center', family='monospace')
    ax.text(2, 5.3, ' ...]', fontsize=7, ha='center', family='monospace')
    
    ax.text(6.5, 6.3, 'Each number\nbecomes a\n32-dim vector', fontsize=8, ha='center', color='#666')
    
    ax.text(11, 6.3, 'Sequence of\nlearned\nrepresentations', fontsize=8, ha='center', color='#666')
    
    # Shape annotations
    ax.text(14, 7.75, 'Shape: (batch, 10, 7)', fontsize=9, color='gray')
    ax.text(14, 4.75, 'Shape: (batch, 7, 37)', fontsize=9, color='gray')
    ax.text(14, 1.75, 'Shape: (batch, 7)', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig("model/diagrams/data_flow.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Generated: model/diagrams/data_flow.png")


# ============================================================================
# MAIN
# ============================================================================

def generate_all_diagrams():
    """Generate all model visualizations."""
    print("\n" + "="*70)
    print("  GENERATING DETAILED MODEL VISUALIZATIONS")
    print("="*70 + "\n")
    
    os.makedirs("model/diagrams", exist_ok=True)
    
    print("1. Generating detailed architecture diagrams...")
    visualize_original_model()
    visualize_multi_output_model()
    visualize_transformer_model()
    visualize_set_prediction_model()
    
    print("\n2. Generating comparison diagrams...")
    visualize_model_comparison()
    visualize_attention_pattern()
    
    print("\n3. Generating educational diagrams...")
    visualize_embedding()
    visualize_data_flow()
    
    print("\n" + "="*70)
    print("  ALL DIAGRAMS GENERATED")
    print("="*70)
    print("\nFiles saved to: model/diagrams/")
    print("\nGenerated files:")
    for f in sorted(os.listdir("model/diagrams")):
        print(f"  - {f}")


if __name__ == "__main__":
    generate_all_diagrams()
