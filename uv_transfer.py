import igl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import trimesh
import argparse
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm


class UVIslandManager:
    def __init__(self, mesh, uv_coords, face_uv_indices, uv_threshold=0.01):
        if len(mesh.faces) == 0:
            raise ValueError("Original mesh must contain at least one triangle")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Original mesh vertices must be finite")
        self.mesh = mesh
        self.uv_coords = np.asarray(uv_coords, dtype=float)
        self.face_uv_indices = np.array(face_uv_indices)
        self.uv_threshold = uv_threshold

        # Uniform normalization preserves Euclidean correspondence across units.
        self.geometry_origin = mesh.vertices[0].copy()
        self.geometry_scale = np.max(np.abs(mesh.vertices - self.geometry_origin)) or 1.0
        self.geometry_vertices = (mesh.vertices - self.geometry_origin) / self.geometry_scale

        # Identical geometry/UV faces use their first occurrence for global queries.
        _, geometry_ids = np.unique(mesh.vertices, axis=0, return_inverse=True)
        _, uv_ids = np.unique(self.uv_coords, axis=0, return_inverse=True)
        triangle_keys = np.column_stack((geometry_ids[mesh.faces], uv_ids[self.face_uv_indices]))
        _, first_faces, inverse = np.unique(triangle_keys, axis=0, return_index=True, return_inverse=True)
        self.canonical_faces = first_faces[inverse]
        self.search_faces = np.sort(first_faces)

        # Detect UV islands
        self.islands = self._detect_islands()
        self.face_to_island = np.full(len(mesh.faces), -1)
        for idx, faces in enumerate(self.islands):
            self.face_to_island[faces] = idx

        # Original face indices for island-restricted surface queries.
        self.island_faces = {}
        for idx, faces in enumerate(self.islands):
            if faces:
                self.island_faces[idx] = np.array(faces)

    def _detect_islands(self):
        """Find edge-connected charts without welding distinct OBJ UV indices."""
        face_uvs = self.uv_coords[self.face_uv_indices]

        # Normalize each UV edge independently before checking for zero area.
        with np.errstate(invalid="ignore", over="ignore"):
            edges = face_uvs[:, 1:] - face_uvs[:, :1]
            scale = np.max(np.abs(edges), axis=-1, keepdims=True)
            edges = np.divide(edges, scale, out=np.zeros_like(edges), where=scale > 0)
            area = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]

        valid = np.isfinite(face_uvs).all(axis=(1, 2)) & (area != 0)
        valid_faces = np.flatnonzero(valid)
        if len(valid_faces) == 0:
            return []

        # A corner belongs to both a geometric vertex and an explicit UV vertex.
        corners = np.column_stack((self.mesh.faces[valid].ravel(), self.face_uv_indices[valid].ravel()))
        _, vertex_ids = np.unique(corners, axis=0, return_inverse=True)
        _, labels = igl.facet_components(vertex_ids.reshape(-1, 3))

        # Group by component while preserving face order within each chart.
        order = np.argsort(labels, kind="stable")
        splits = np.flatnonzero(np.diff(labels[order])) + 1
        return [valid_faces[group].tolist() for group in np.split(order, splits)]

    def interpolate_uv(self, vertex_pos, island_id=None):
        """Return (2,) or (N, 2) UVs; nonfinite query rows remain NaN."""
        # Preserve the requested output shape while normalizing geometry units.
        positions = np.asarray(vertex_pos, dtype=float)
        single_position = positions.shape == (3,)

        with np.errstate(over="ignore", invalid="ignore"):
            positions = (positions.reshape(-1, 3) - self.geometry_origin) / self.geometry_scale

        # Keep invalid rows as NaN and exclude them from native surface queries.
        uvs = np.full((len(positions), 2), np.nan)
        finite = np.isfinite(positions).all(axis=1)
        if not finite.any():
            return uvs[0] if single_position else uvs

        # Select the closest surface triangle within the requested source island.
        points = positions[finite]
        face_ids = self.island_faces.get(island_id, self.search_faces)
        _, indices, _ = igl.point_mesh_squared_distance(points, self.geometry_vertices, self.mesh.faces[face_ids])
        faces = face_ids[indices]
        triangles = self.geometry_vertices[self.mesh.faces[faces]]

        # Recenter and scale each triangle together with its query point.
        rows, axes = np.arange(len(points)), np.arange(3)
        origin = triangles[:, 0]
        local = triangles - origin[:, None]
        scale = np.max(np.abs(local), axis=(1, 2))
        scale = np.where(scale > 0, scale, 1.0)
        triangles = local / scale[:, None, None]
        points = (points - origin) / scale[:, None]

        # Pivot opposite the longest edge, then solve the scaled plane basis by SVD.
        edges = np.roll(triangles, -1, axis=1) - triangles
        length2 = np.vecdot(edges, edges)
        longest = length2.argmax(axis=1)
        order = (longest[:, None] + 2 + axes) % 3
        corners = triangles[rows[:, None], order]

        basis = (corners[:, 1:] - corners[:, :1]).swapaxes(1, 2)
        size = np.max(np.abs(basis), axis=1)
        unit = np.divide(basis, size[:, None], out=np.zeros_like(basis), where=size[:, None] > 0)
        inverse = np.linalg.pinv(unit)
        rhs = points - corners[:, 0]

        # Refine the plane solution once, then restore the original corner order.
        xy = np.matvec(inverse, rhs)
        xy += np.matvec(inverse, rhs - np.matvec(unit, xy))
        xy = np.divide(xy, size, out=np.zeros_like(xy), where=size > 0)

        plane = np.empty((len(points), 3))
        plane[rows[:, None], order] = np.column_stack((1 - xy.sum(axis=1), xy))
        full_rank = np.any(np.cross(unit[:, :, 0], unit[:, :, 1]) != 0, axis=1)

        # Outside the triangle, choose the nearest of its three actual edge segments.
        t = np.divide(
            np.vecdot(points[:, None] - triangles, edges), length2,
            out=np.zeros_like(length2), where=length2 > 0,
        ).clip(0, 1)
        edge_weights = np.zeros((len(points), 3, 3))
        edge_weights[:, axes, axes] = 1 - t
        edge_weights[:, axes, (axes + 1) % 3] = t

        delta = points[:, None] - (triangles + t[:, :, None] * edges)
        best = np.where(full_rank, np.vecdot(delta, delta).argmin(axis=1), longest)
        weights = edge_weights[rows, best]
        inside = full_rank & np.all(plane >= 0, axis=1)
        weights[inside] = plane[inside]

        # A collinear triangle uses its longest edge; a collapsed point has no unique UV.
        weights[length2.max(axis=1) == 0] = 1 / 3

        # Blend the selected face's UV corners and restore the scalar/batch shape.
        uvs[finite] = np.vecmat(weights, self.uv_coords[self.face_uv_indices[faces]])
        return uvs[0] if single_position else uvs

    def detect_simplified_islands(self, faces, face_uvs, face_to_orig_island):
        """Find edge-connected UV components while preserving source labels."""
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        face_uvs = np.asarray(face_uvs, dtype=float).reshape(-1, 3, 2)
        original_ids = np.asarray(face_to_orig_island)

        # Collect each shared edge once, including nonmanifold neighbors.
        neighbors, neighbor_edges = igl.triangle_triangle_adjacency_lists(faces)
        links = []
        for face, face_neighbors in enumerate(neighbors):
            for edge, adjacent_faces in enumerate(face_neighbors):
                adjacent_edges = neighbor_edges[face][edge]
                for neighbor, other_edge in zip(adjacent_faces, adjacent_edges):
                    if face < neighbor:
                        links.append((face, edge, neighbor, other_edge))

        links = np.asarray(links, dtype=np.int64).reshape(-1, 4)

        # In libigl, edge e is opposite corner e; its endpoints are e+1 and e+2.
        pairs = links[:, [0, 2]]
        corners = (links[:, [1, 3], None] + [1, 2]) % 3

        # Align both endpoints by geometric vertex ID, regardless of edge direction.
        order = np.argsort(faces[pairs[:, :, None], corners], axis=-1)
        corners = np.take_along_axis(corners, order, axis=-1)
        edge_uvs = face_uvs[pairs[:, :, None], corners]

        # Connect only finite, UV-continuous edges within the same source island.
        with np.errstate(invalid="ignore"):
            delta = edge_uvs[:, 0] - edge_uvs[:, 1]
            continuous = np.all(np.hypot(delta[..., 0], delta[..., 1]) <= self.uv_threshold, axis=-1)

        continuous &= np.isfinite(edge_uvs).all(axis=(1, 2, 3))
        continuous &= original_ids[pairs[:, 0]] == original_ids[pairs[:, 1]]

        # Keep actual connected components, including isolated faces and empty input.
        components = trimesh.graph.connected_components(pairs[continuous], nodes=np.arange(len(faces)), engine="scipy")
        return [list(map(int, group)) for group in components]


def visualize_uv_transfer(manager, simplified_mesh, face_uvs, islands, face_to_orig_island):
    """Show original and transferred mesh/UV islands together; return the figure."""
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))

    # Match each simplified component to its most frequent assigned source island.
    color_mapping = {}
    for island_id, faces in enumerate(islands):
        source_ids = [face_to_orig_island[face] for face in faces if face_to_orig_island[face] >= 0]
        color_mapping[island_id] = Counter(source_ids).most_common(1)[0][0] if source_ids else island_id

    views = (
        ("Original", manager.mesh, manager.uv_coords[manager.face_uv_indices], manager.islands, {}),
        ("Simplified", simplified_mesh, face_uvs, islands, color_mapping),
    )
    fig = plt.figure(figsize=(14, 10), layout="constrained")

    for row, (title, mesh, uvs, groups, mapping) in enumerate(views):
        # Use the same source-island colors for both the mesh and its UV edges.
        face_colors = np.full((len(mesh.faces), 4), 0.8)
        for island_id, faces in enumerate(groups):
            face_colors[faces] = colors[mapping.get(island_id, island_id) % len(colors)]

        # Duplicate source faces share the canonical colors used by global transfer.
        if row == 0:
            face_colors = face_colors[manager.canonical_faces]

        mesh_ax = fig.add_subplot(2, 2, 2 * row + 1, projection="3d")
        mesh_ax.add_collection3d(Poly3DCollection(
            mesh.triangles, facecolors=face_colors, edgecolors="k", linewidths=0.1, alpha=0.8
        ))
        mesh_ax.set(
            xlim=mesh.bounds[:, 0], ylim=mesh.bounds[:, 1], zlim=mesh.bounds[:, 2],
            title=f"{title} mesh ({len(groups)} islands)",
        )
        mesh_ax.set_box_aspect(mesh.extents.copy())

        # Batch all closed UV triangles into one collection instead of per-face plots.
        uv_ax = fig.add_subplot(2, 2, 2 * row + 2)
        uv_ax.add_collection(LineCollection(
            uvs[:, [0, 1, 2, 0]], colors=face_colors, linewidths=0.3, alpha=0.7
        ))
        uv_ax.set(
            aspect="equal", xlim=(-0.1, 1.1), ylim=(-0.1, 1.1),
            title=f"{title} UV map ({len(groups)} islands)",
        )
        uv_ax.grid(True, alpha=0.3)

    plt.show()
    return fig


def transfer_uvs(original_mesh, uv_coords, face_uv_indices,
                simplified_mesh, uv_threshold=0.01):
    """Transfer UV corners within source islands and find target components."""
    # Create UV island manager
    manager = UVIslandManager(original_mesh, uv_coords, face_uv_indices, uv_threshold)

    # Compute simplified face centers
    simp_centers = igl.barycenter(simplified_mesh.vertices, simplified_mesh.faces)
    if not np.isfinite(simp_centers).all():
        raise ValueError("Simplified mesh positions must be finite")

    # Find the closest triangle surface rather than the closest face centroid.
    _, closest, _ = igl.point_mesh_squared_distance(
        (simp_centers - manager.geometry_origin) / manager.geometry_scale,
        manager.geometry_vertices,
        original_mesh.faces[manager.search_faces],
    )
    closest = manager.search_faces[closest]

    # UV transfer
    face_uvs = np.zeros((len(simplified_mesh.faces), 3, 2))
    face_to_orig_island = manager.face_to_island[closest]

    # Group by island; unassigned faces (-1) fall back to the interpolator's global search.
    faces_by_island = defaultdict(list)
    for i, island_id in enumerate(face_to_orig_island):
        faces_by_island[island_id].append(i)

    # UV interpolation, one batch per island; shared vertices are looked up once
    for island_id, face_indices in tqdm(faces_by_island.items(), desc="Processing islands"):
        face_indices = np.asarray(face_indices)
        vertex_ids = simplified_mesh.faces[face_indices]
        unique_ids, inverse = np.unique(vertex_ids, return_inverse=True)
        uvs = manager.interpolate_uv(simplified_mesh.vertices[unique_ids], island_id)
        face_uvs[face_indices] = uvs[inverse]

    # Detect islands in simplified mesh
    islands = manager.detect_simplified_islands(simplified_mesh.faces, face_uvs, face_to_orig_island)

    return face_uvs, islands, manager, face_to_orig_island


def save_obj_with_uvs(path, vertices, faces, face_uvs):
    """Write independent geometry and UV indices without approximate welding."""
    uvs, inverse = np.unique(
        np.asarray(face_uvs, dtype=float).reshape(-1, 2), axis=0, return_inverse=True,
    )
    igl.writeOBJ(path, vertices, faces, TC=uvs, FTC=inverse.reshape(-1, 3))


def load_mesh_with_uvs(path):
    """Load unified geometry/UV vertices without re-exporting the OBJ."""
    mesh = trimesh.load_mesh(path, process=False)
    uv_coords = getattr(mesh.visual, "uv", None)
    if uv_coords is None:
        raise ValueError("No UV data found")

    return mesh, uv_coords, mesh.faces.copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UV Transfer")
    parser.add_argument("--original", required=True, help="Original mesh with UVs")
    parser.add_argument("--simplified", required=True, help="Simplified mesh")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--uv_threshold", type=float, default=0.01)

    args = parser.parse_args()

    # Load meshes
    print("Loading meshes...")
    original_mesh, uv_coords, face_uv_indices = load_mesh_with_uvs(args.original)
    simplified_mesh = trimesh.load_mesh(args.simplified, process=False)

    if isinstance(simplified_mesh, trimesh.Scene):
        simplified_mesh = list(simplified_mesh.geometry.values())[0]

    print(f"Original: {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")
    print(f"Simplified: {len(simplified_mesh.vertices)} vertices, {len(simplified_mesh.faces)} faces")

    # Transfer UV
    print("\nTransferring UVs...")
    face_uvs, islands, manager, face_to_orig_island = transfer_uvs(
        original_mesh, uv_coords, face_uv_indices,
        simplified_mesh, args.uv_threshold
    )

    print(f"Original UV islands: {len(manager.islands)}")
    print(f"Simplified UV islands: {len(islands)}")

    # Save
    save_obj_with_uvs(args.output, simplified_mesh.vertices, simplified_mesh.faces, face_uvs)
    print(f"\nSaved to: {args.output}")

    # Visualize
    if args.visualize:
        visualize_uv_transfer(manager, simplified_mesh, face_uvs, islands, face_to_orig_island)

