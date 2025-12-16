"""
CTC (Connectionist Temporal Classification) components for visualization
Includes: Trellis, Forward-Backward algorithm, Decoding
"""

from manim import *
import numpy as np
from .colors import *


class CTCTrellis(VGroup):
    """
    Dynamic programming trellis for CTC visualization
    """

    def __init__(
        self,
        n_frames=10,
        labels="HELLO",
        cell_size=0.6,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_frames = n_frames
        self.labels = labels
        self.n_labels = len(labels) * 2 + 1  # With blanks
        self.cell_size = cell_size

        # Create expanded label sequence (with blanks)
        self.expanded_labels = self._expand_labels(labels)

        # Create trellis grid
        self.grid = self._create_grid()
        self.add(self.grid)

        # Create label annotations
        self.annotations = self._create_annotations()
        self.add(self.annotations)

        # Create nodes
        self.nodes = self._create_nodes()
        self.add(self.nodes)

    def _expand_labels(self, labels):
        """Expand labels with blank tokens"""
        expanded = ['ε']  # Start blank
        for char in labels:
            expanded.append(char)
            expanded.append('ε')
        return expanded

    def _create_grid(self):
        """Create background grid"""
        grid = VGroup()

        # Vertical lines (time steps)
        for i in range(self.n_frames + 1):
            x = i * self.cell_size - (self.n_frames * self.cell_size) / 2
            line = Line(
                start=[x, -self.n_labels * self.cell_size / 2, 0],
                end=[x, self.n_labels * self.cell_size / 2, 0],
                color=SLATE_700,
                stroke_width=0.5
            )
            grid.add(line)

        # Horizontal lines (labels)
        for j in range(self.n_labels + 1):
            y = j * self.cell_size - (self.n_labels * self.cell_size) / 2
            line = Line(
                start=[-self.n_frames * self.cell_size / 2, y, 0],
                end=[self.n_frames * self.cell_size / 2, y, 0],
                color=SLATE_700,
                stroke_width=0.5
            )
            grid.add(line)

        return grid

    def _create_annotations(self):
        """Create time and label annotations"""
        annotations = VGroup()

        # Time labels (top)
        for i in range(self.n_frames):
            x = (i + 0.5) * self.cell_size - (self.n_frames * self.cell_size) / 2
            y = self.n_labels * self.cell_size / 2 + 0.3
            label = Text(f"t={i}", font_size=10, color=SLATE_500)
            label.move_to([x, y, 0])
            annotations.add(label)

        # Label annotations (left)
        for j, char in enumerate(self.expanded_labels):
            x = -self.n_frames * self.cell_size / 2 - 0.4
            y = (self.n_labels - j - 0.5) * self.cell_size - (self.n_labels * self.cell_size) / 2
            color = SLATE_500 if char == 'ε' else CYAN_400
            label = Text(char, font_size=12, color=color)
            label.move_to([x, y, 0])
            annotations.add(label)

        return annotations

    def _create_nodes(self):
        """Create trellis nodes"""
        nodes = VGroup()

        for i in range(self.n_frames):
            for j in range(self.n_labels):
                x = (i + 0.5) * self.cell_size - (self.n_frames * self.cell_size) / 2
                y = (self.n_labels - j - 0.5) * self.cell_size - (self.n_labels * self.cell_size) / 2

                node = Circle(
                    radius=self.cell_size * 0.25,
                    fill_color=SLATE_800,
                    fill_opacity=0.8,
                    stroke_color=SLATE_600,
                    stroke_width=1
                )
                node.move_to([x, y, 0])
                nodes.add(node)

        return nodes

    def highlight_path(self, path_indices, color=EMERALD_400):
        """Highlight a path through the trellis"""
        path_group = VGroup()

        for i, j in enumerate(path_indices):
            if i >= self.n_frames or j >= self.n_labels:
                continue

            x = (i + 0.5) * self.cell_size - (self.n_frames * self.cell_size) / 2
            y = (self.n_labels - j - 0.5) * self.cell_size - (self.n_labels * self.cell_size) / 2

            highlight = Circle(
                radius=self.cell_size * 0.3,
                fill_color=color,
                fill_opacity=0.5,
                stroke_color=color,
                stroke_width=2
            )
            highlight.move_to([x, y, 0])
            path_group.add(highlight)

            # Add edge to next node
            if i < len(path_indices) - 1:
                next_j = path_indices[i + 1]
                next_x = (i + 1.5) * self.cell_size - (self.n_frames * self.cell_size) / 2
                next_y = (self.n_labels - next_j - 0.5) * self.cell_size - (self.n_labels * self.cell_size) / 2

                edge = Arrow(
                    start=[x, y, 0],
                    end=[next_x, next_y, 0],
                    color=color,
                    stroke_width=2,
                    buff=self.cell_size * 0.3
                )
                path_group.add(edge)

        return path_group


class CTCForwardVisualization(VGroup):
    """
    Visualization of CTC forward algorithm
    """

    def __init__(
        self,
        n_frames=8,
        labels="CAT",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Create trellis
        self.trellis = CTCTrellis(n_frames=n_frames, labels=labels)
        self.add(self.trellis)

        # Alpha values display
        self.alpha_display = self._create_alpha_display()
        self.add(self.alpha_display)

        # Formula
        self.formula = self._create_formula()
        self.add(self.formula)

    def _create_alpha_display(self):
        """Create display for alpha values"""
        display = VGroup()

        title = Text("Forward Pass (α)", font_size=18, color=CYAN_400)
        title.to_corner(UR)
        title.shift(DOWN * 0.5)

        display.add(title)

        return display

    def _create_formula(self):
        """Create CTC forward formula"""
        formula = MathTex(
            r"\alpha_t(s) = \sum_{s' \in \{s-1, s\}} \alpha_{t-1}(s') \cdot p(l_s | x_t)",
            font_size=20
        )
        formula.set_color(SLATE_300)
        formula.to_corner(DL)
        formula.shift(UP * 0.5)

        return formula


class CTCBackwardVisualization(VGroup):
    """
    Visualization of CTC backward algorithm
    """

    def __init__(
        self,
        n_frames=8,
        labels="CAT",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Create trellis
        self.trellis = CTCTrellis(n_frames=n_frames, labels=labels)
        self.add(self.trellis)

        # Beta values display
        self.beta_display = self._create_beta_display()
        self.add(self.beta_display)

        # Formula
        self.formula = self._create_formula()
        self.add(self.formula)

    def _create_beta_display(self):
        """Create display for beta values"""
        display = VGroup()

        title = Text("Backward Pass (β)", font_size=18, color=PURPLE_400)
        title.to_corner(UR)
        title.shift(DOWN * 0.5)

        display.add(title)

        return display

    def _create_formula(self):
        """Create CTC backward formula"""
        formula = MathTex(
            r"\beta_t(s) = \sum_{s' \in \{s, s+1\}} \beta_{t+1}(s') \cdot p(l_{s'} | x_{t+1})",
            font_size=20
        )
        formula.set_color(SLATE_300)
        formula.to_corner(DL)
        formula.shift(UP * 0.5)

        return formula


class CTCLossVisualization(VGroup):
    """
    Visualization of CTC loss computation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Loss formula
        self.loss_formula = self._create_loss_formula()
        self.add(self.loss_formula)

        # Gradient flow
        self.gradient = self._create_gradient_visualization()
        self.add(self.gradient)

    def _create_loss_formula(self):
        """Create CTC loss formula"""
        formula = VGroup()

        title = Text("CTC Loss", font_size=24, color=ROSE_400)
        title.move_to([0, 2, 0])

        loss_eq = MathTex(
            r"\mathcal{L}_{CTC} = -\log P(Y|X) = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi|X)",
            font_size=24
        )
        loss_eq.set_color(SLATE_50)
        loss_eq.move_to([0, 1, 0])

        explanation = Text(
            "Sum over all valid alignments π that map to target Y",
            font_size=14,
            color=SLATE_400
        )
        explanation.move_to([0, 0.2, 0])

        formula.add(title, loss_eq, explanation)

        return formula

    def _create_gradient_visualization(self):
        """Create gradient flow visualization"""
        gradient = VGroup()

        # Gradient formula
        grad_eq = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial y_t^k} = y_t^k - \frac{1}{P(Y|X)} \sum_{s: l_s = k} \alpha_t(s) \beta_t(s)",
            font_size=18
        )
        grad_eq.set_color(SLATE_300)
        grad_eq.move_to([0, -1, 0])

        label = Text("Gradient for backpropagation", font_size=12, color=SLATE_500)
        label.next_to(grad_eq, DOWN, buff=0.2)

        gradient.add(grad_eq, label)

        return gradient


class CTCDecoder(VGroup):
    """
    Visualization of CTC decoding process
    """

    def __init__(
        self,
        output_sequence=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if output_sequence is None:
            output_sequence = ['ε', 'H', 'H', 'ε', 'E', 'ε', 'L', 'L', 'L', 'O', 'ε']

        self.output_sequence = output_sequence

        # Create output sequence display
        self.sequence_display = self._create_sequence_display()
        self.add(self.sequence_display)

        # Create decoding steps
        self.decode_steps = self._create_decode_steps()
        self.add(self.decode_steps)

    def _create_sequence_display(self):
        """Create visual display of output sequence"""
        display = VGroup()

        title = Text("Model Output (per frame)", font_size=16, color=SLATE_400)
        title.move_to([0, 1.5, 0])
        display.add(title)

        cell_width = 0.6
        start_x = -len(self.output_sequence) * cell_width / 2

        for i, token in enumerate(self.output_sequence):
            # Cell
            is_blank = token == 'ε'
            cell = RoundedRectangle(
                width=cell_width * 0.9,
                height=0.6,
                corner_radius=0.05,
                fill_color=SLATE_700 if is_blank else CYAN_600,
                fill_opacity=0.5 if is_blank else 0.8,
                stroke_color=SLATE_600 if is_blank else CYAN_400,
                stroke_width=1
            )
            cell.move_to([start_x + i * cell_width + cell_width / 2, 0.5, 0])

            # Token label
            label = Text(token, font_size=14, color=SLATE_400 if is_blank else SLATE_50)
            label.move_to(cell.get_center())

            display.add(cell, label)

        return display

    def _create_decode_steps(self):
        """Create visualization of decoding steps"""
        steps = VGroup()

        # Step 1: Remove blanks
        step1 = VGroup()
        step1_title = Text("1. Remove blanks (ε)", font_size=14, color=SLATE_400)
        step1_title.move_to([-3, -0.5, 0])

        no_blanks = [t for t in self.output_sequence if t != 'ε']
        step1_result = Text(''.join(no_blanks), font_size=18, color=AMBER_400)
        step1_result.next_to(step1_title, RIGHT, buff=0.5)

        step1.add(step1_title, step1_result)

        # Step 2: Merge consecutive
        step2 = VGroup()
        step2_title = Text("2. Merge consecutive duplicates", font_size=14, color=SLATE_400)
        step2_title.move_to([-3, -1.2, 0])

        # Collapse consecutive
        collapsed = []
        for t in no_blanks:
            if not collapsed or collapsed[-1] != t:
                collapsed.append(t)

        step2_result = Text(''.join(collapsed), font_size=18, color=EMERALD_400)
        step2_result.next_to(step2_title, RIGHT, buff=0.5)

        step2.add(step2_title, step2_result)

        # Final result
        final = VGroup()
        final_title = Text("Final Output:", font_size=16, color=SLATE_50)
        final_title.move_to([-2, -2, 0])

        final_result = Text(''.join(collapsed), font_size=24, color=CYAN_400)
        final_result.next_to(final_title, RIGHT, buff=0.3)

        final.add(final_title, final_result)

        steps.add(step1, step2, final)

        return steps


class AlignmentVisualization(VGroup):
    """
    Visualization showing multiple valid alignments for CTC
    """

    def __init__(
        self,
        target="HI",
        n_frames=6,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.target = target
        self.n_frames = n_frames

        # Generate some valid alignments
        self.alignments = self._generate_alignments()

        # Create display
        self.display = self._create_display()
        self.add(self.display)

    def _generate_alignments(self):
        """Generate example valid alignments"""
        # For "HI" with 6 frames, some valid alignments:
        return [
            ['ε', 'H', 'ε', 'I', 'ε', 'ε'],
            ['H', 'H', 'ε', 'I', 'I', 'ε'],
            ['ε', 'H', 'H', 'I', 'ε', 'ε'],
            ['H', 'ε', 'ε', 'I', 'ε', 'ε'],
            ['ε', 'H', 'ε', 'ε', 'I', 'ε'],
        ]

    def _create_display(self):
        """Create visual display of alignments"""
        display = VGroup()

        title = Text(f"Valid alignments for '{self.target}':", font_size=18, color=SLATE_50)
        title.move_to([0, 2.5, 0])
        display.add(title)

        cell_width = 0.6
        row_height = 0.7

        for row_idx, alignment in enumerate(self.alignments):
            row = VGroup()
            start_x = -len(alignment) * cell_width / 2
            y_pos = 1.5 - row_idx * row_height

            for col_idx, token in enumerate(alignment):
                is_blank = token == 'ε'
                cell = RoundedRectangle(
                    width=cell_width * 0.9,
                    height=row_height * 0.8,
                    corner_radius=0.05,
                    fill_color=SLATE_700 if is_blank else CYAN_600,
                    fill_opacity=0.3 if is_blank else 0.6,
                    stroke_width=0
                )
                cell.move_to([start_x + col_idx * cell_width + cell_width / 2, y_pos, 0])

                label = Text(token, font_size=12, color=SLATE_500 if is_blank else SLATE_50)
                label.move_to(cell.get_center())

                row.add(cell, label)

            # Arrow pointing to collapsed result
            arrow = Arrow(
                start=[start_x + len(alignment) * cell_width + 0.3, y_pos, 0],
                end=[start_x + len(alignment) * cell_width + 1, y_pos, 0],
                color=SLATE_600,
                stroke_width=2,
                buff=0
            )
            result = Text(self.target, font_size=14, color=EMERALD_400)
            result.next_to(arrow, RIGHT, buff=0.1)

            row.add(arrow, result)
            display.add(row)

        # Explanation
        explanation = Text(
            "CTC sums probabilities over ALL valid alignments",
            font_size=14,
            color=SLATE_400
        )
        explanation.move_to([0, -2, 0])
        display.add(explanation)

        return display


class BeamSearchVisualization(VGroup):
    """
    Visualization of beam search decoding
    """

    def __init__(
        self,
        beam_width=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.beam_width = beam_width

        # Create tree structure
        self.tree = self._create_beam_tree()
        self.add(self.tree)

    def _create_beam_tree(self):
        """Create beam search tree visualization"""
        tree = VGroup()

        title = Text(f"Beam Search (width={self.beam_width})", font_size=18, color=SLATE_50)
        title.move_to([0, 3, 0])
        tree.add(title)

        # Example beam search expansion
        # Level 0: Start
        root = Circle(radius=0.3, fill_color=SLATE_700, fill_opacity=0.8, stroke_color=CYAN_400)
        root.move_to([0, 2, 0])
        root_label = Text("⟨s⟩", font_size=12, color=SLATE_50)
        root_label.move_to(root.get_center())
        tree.add(root, root_label)

        # Level 1: First predictions
        level1_tokens = ['H', 'A', 'T']
        level1_probs = [0.6, 0.2, 0.1]

        for i, (token, prob) in enumerate(zip(level1_tokens, level1_probs)):
            x = (i - 1) * 2
            node = Circle(
                radius=0.3,
                fill_color=CYAN_600 if i < self.beam_width else SLATE_700,
                fill_opacity=0.8 if i < self.beam_width else 0.3,
                stroke_color=CYAN_400 if i < self.beam_width else SLATE_600
            )
            node.move_to([x, 0.5, 0])

            node_label = Text(token, font_size=12, color=SLATE_50)
            node_label.move_to(node.get_center())

            prob_label = Text(f"{prob:.1f}", font_size=10, color=SLATE_400)
            prob_label.next_to(node, DOWN, buff=0.1)

            edge = Line(
                start=root.get_bottom(),
                end=node.get_top(),
                color=CYAN_400 if i < self.beam_width else SLATE_600,
                stroke_width=2 if i < self.beam_width else 1
            )

            tree.add(edge, node, node_label, prob_label)

        return tree
