from collections import deque
from math import cos, pi, sin
from typing import Hashable

import numpy as np


class MCTSVisualizer3D:
    def __init__(self, root_hash: Hashable, radius_step: float = 1.0, depth_step: float = 0.65) -> None:
        self.root_hash = root_hash
        self.radius_step = float(radius_step)
        self.depth_step = float(depth_step)
        self.adjacency: dict[Hashable, dict[Hashable, int]] = {}

    def add_transition(self, parent_hash: Hashable, child_hash: Hashable, visits: int = 1) -> None:
        children = self.adjacency.setdefault(parent_hash, {})
        children[child_hash] = children.get(child_hash, 0) + int(visits)

    def stats(self) -> dict[str, float]:
        depths = self._depths()
        edges = sum(len(children) for children in self.adjacency.values())
        return {
            "nodes": float(len(depths)),
            "edges": float(edges),
            "max_depth": float(max(depths.values(), default=0)),
            "avg_branching": float(edges / max(len(self.adjacency), 1)),
        }

    def _depths(self) -> dict[Hashable, int]:
        depths = {self.root_hash: 0}
        queue = deque([self.root_hash])
        while queue:
            node = queue.popleft()
            for child in self.adjacency.get(node, {}):
                if child not in depths:
                    depths[child] = depths[node] + 1
                    queue.append(child)
        return depths

    def _tree_children(self) -> dict[Hashable, list[Hashable]]:
        children_by_parent: dict[Hashable, list[Hashable]] = {}
        visited = {self.root_hash}
        queue = deque([self.root_hash])
        while queue:
            parent = queue.popleft()
            ordered_children = sorted(
                self.adjacency.get(parent, {}),
                key=lambda child: (-self.adjacency[parent][child], repr(child)),
            )
            for child in ordered_children:
                if child in visited:
                    continue
                visited.add(child)
                children_by_parent.setdefault(parent, []).append(child)
                queue.append(child)
        return children_by_parent

    def _compute_layout(self) -> dict[Hashable, tuple[float, float, float]]:
        children_by_parent = self._tree_children()
        positions: dict[Hashable, tuple[float, float, float]] = {self.root_hash: (0.0, 0.0, 0.0)}
        node_angles: dict[Hashable, float] = {self.root_hash: 0.0}
        queue = deque([(self.root_hash, 0, 2 * pi)])
        while queue:
            parent, depth, spread = queue.popleft()
            children = children_by_parent.get(parent, [])
            if not children:
                continue

            parent_x, parent_y, parent_z = positions[parent]
            parent_angle = node_angles[parent]
            child_spread = 2 * pi if depth == 0 else max(pi / 9, min(pi * 0.75, spread * 0.55))
            if len(children) == 1:
                child_angles = [parent_angle]
            else:
                start = parent_angle - child_spread / 2
                step = child_spread / (len(children) - 1)
                child_angles = [start + step * index for index in range(len(children))]

            local_radius = self.radius_step * (1.0 + 0.08 * min(len(children), 12)) / (1.0 + 0.08 * depth)
            for child, angle in zip(children, child_angles):
                positions[child] = (
                    parent_x + local_radius * cos(angle),
                    parent_y + local_radius * sin(angle),
                    parent_z + self.depth_step,
                )
                node_angles[child] = angle
                queue.append((child, depth + 1, child_spread))
        return positions

    def show(self, min_width: float = 0.2, max_width: float = 5.0, save_path: str | None = None) -> None:
        if save_path is not None:
            import matplotlib

            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        positions = self._compute_layout()
        depths = self._depths()
        segments = []
        linewidths = []
        edge_depths = []
        visits = [count for children in self.adjacency.values() for count in children.values()]
        max_visits = max(visits, default=1)

        for parent, children in self.adjacency.items():
            if parent not in positions:
                continue
            for child, count in children.items():
                if child not in positions:
                    continue
                segments.append([positions[parent], positions[child]])
                linewidths.append(min_width + (count / max_visits) * (max_width - min_width))
                edge_depths.append(depths.get(child, 0))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        if segments:
            ax.add_collection3d(
                Line3DCollection(
                    segments,
                    linewidths=linewidths,
                    colors=plt.cm.viridis(np.array(edge_depths, dtype=np.float64) / max(depths.values(), default=1)),
                    alpha=0.68,
                )
            )

        nodes = list(positions)
        points = np.array([positions[node] for node in nodes], dtype=np.float64)
        if len(points) > 0:
            node_depths = np.array([depths.get(node, 0) for node in nodes], dtype=np.float64)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=node_depths, cmap="viridis", s=9, depthshade=True)
            root = np.array(positions[self.root_hash])
            ax.scatter([root[0]], [root[1]], [root[2]], c="#D7263D", s=80, depthshade=True)
            limit = float(np.abs(points).max()) + 0.5
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(-0.5, max(1.0, points[:, 2].max() + 0.5))

        stats = self.stats()
        ax.set_title(
            f"MCTS tree - nodes={int(stats['nodes'])}, edges={int(stats['edges'])}, max_depth={int(stats['max_depth'])}"
        )
        ax.set_axis_off()
        fig.tight_layout()
        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path, dpi=140)
        plt.close(fig)


def demo_visualization() -> None:
    viz = MCTSVisualizer3D(0)
    for i in range(1, 80):
        viz.add_transition((i - 1) // 3, i, visits=i % 17 + 1)
    print(viz.stats())
    viz.show()


if __name__ == "__main__":
    demo_visualization()
