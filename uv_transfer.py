import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from collections import defaultdict, Counter
import trimesh
import argparse
from tqdm import tqdm


# Captures the three vt indices from trimesh's exported "f v/vt[/vn] ..." lines.
_FACE_UV_LINE = re.compile(
    r"(?m)^f \s*\d+/(\d+)\S*\s+\d+/(\d+)\S*\s+\d+/(\d+)\S*\s*$"
)


def _row_dot(a, b):
    # Matches np.dot's accumulation order on length-3 vectors, so results stay bit-identical.
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


class UVIslandManager:
    def __init__(self, mesh, uv_coords, face_uv_indices, uv_threshold=0.01):
        self.mesh = mesh
        self.uv_coords = np.array(uv_coords)
        self.face_uv_indices = np.array(face_uv_indices)
        self.uv_threshold = uv_threshold
        self.min_island_size = 5

        # Precompute
        self.face_centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.face_kdtree = cKDTree(self.face_centers)

        # Detect UV islands
        self.islands = self._detect_islands()
        self.face_to_island = np.full(len(mesh.faces), -1)
        for idx, faces in enumerate(self.islands):
            self.face_to_island[faces] = idx

        # KD-tree per island
        self.island_kdtrees = {}
        self.island_faces = {}
        for idx, faces in enumerate(self.islands):
            if faces:
                centers = self.face_centers[faces]
                self.island_kdtrees[idx] = cKDTree(centers)
                self.island_faces[idx] = np.array(faces)

    def _detect_islands(self):
        unique_uvs, inverse = np.unique(self.uv_coords, axis=0, return_inverse=True)

        # Welded UV vertex ids per face; a repeated vertex makes the face UV-degenerate (no island).
        face_uv_verts = inverse[self.face_uv_indices]
        valid = (
            (face_uv_verts[:, 0] != face_uv_verts[:, 1])
            & (face_uv_verts[:, 1] != face_uv_verts[:, 2])
            & (face_uv_verts[:, 0] != face_uv_verts[:, 2])
        )
        edges = face_uv_verts[valid]
        if len(edges) == 0:
            return []

        # Connected components over the UV edge graph
        rows = edges[:, [0, 1, 2]].ravel()
        cols = edges[:, [1, 2, 0]].ravel()
        n_verts = len(unique_uvs)
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_verts, n_verts),
        )
        _, labels = connected_components(graph, directed=False)

        # Rank components by smallest UV vertex id to reproduce BFS discovery order.
        comp_min_vert = np.full(labels.max() + 1, n_verts, dtype=np.int64)
        np.minimum.at(comp_min_vert, labels, np.arange(n_verts))

        # Group faces by component; a valid face lies entirely in its first UV vertex's component.
        face_comp = np.where(valid, labels[face_uv_verts[:, 0]], -1)
        valid_faces = np.flatnonzero(face_comp >= 0)
        order = np.argsort(comp_min_vert[face_comp[valid_faces]], kind="stable")
        sorted_faces = valid_faces[order]
        sorted_keys = comp_min_vert[face_comp[sorted_faces]]
        splits = np.flatnonzero(np.diff(sorted_keys)) + 1
        return [group.tolist() for group in np.split(sorted_faces, splits)]

    def _interpolate_uv_batch(self, positions, island_id=None):
        positions = np.asarray(positions, dtype=float)

        # Search within specified island if provided
        if island_id is not None and island_id in self.island_kdtrees:
            _, indices = self.island_kdtrees[island_id].query(positions, k=1)
            best_faces = self.island_faces[island_id][np.atleast_1d(indices)]
        else:
            # Global search
            _, indices = self.face_kdtree.query(positions, k=1)
            best_faces = np.atleast_1d(indices)

        tri = self.mesh.vertices[self.mesh.faces[best_faces]]
        v0 = tri[:, 0]
        v0v1 = tri[:, 1] - v0
        v0v2 = tri[:, 2] - v0
        normal = np.cross(v0v1, v0v2)
        normal_norm = np.linalg.norm(normal, axis=1)

        bary = np.full((len(positions), 3), 1.0 / 3.0)
        ok = normal_norm > 1e-10
        if ok.any():
            unit = normal[ok] / normal_norm[ok, np.newaxis]
            pos, base = positions[ok], v0[ok]
            e1, e2 = v0v1[ok], v0v2[ok]

            # Project to plane
            d = _row_dot(pos - base, unit)
            proj = pos - d[:, np.newaxis] * unit

            # Barycentric coordinates
            v0p = proj - base
            d00, d01, d11 = _row_dot(e1, e1), _row_dot(e1, e2), _row_dot(e2, e2)
            d20, d21 = _row_dot(v0p, e1), _row_dot(v0p, e2)

            denom = d00 * d11 - d01 * d01
            good = np.abs(denom) > 1e-10
            if good.any():
                v = (d11[good] * d20[good] - d01[good] * d21[good]) / denom[good]
                w = (d00[good] * d21[good] - d01[good] * d20[good]) / denom[good]
                u = 1.0 - v - w
                clipped = np.clip(np.stack([u, v, w], axis=1), 0, 1)
                clipped /= clipped.sum(axis=1, keepdims=True)
                bary[np.flatnonzero(ok)[good]] = clipped

        # UV interpolation
        face_uvs = self.uv_coords[self.face_uv_indices[best_faces]]
        return (bary[:, :, np.newaxis] * face_uvs).sum(axis=1)

    def interpolate_uv(self, vertex_pos, island_id=None):
        return self._interpolate_uv_batch(
            np.asarray(vertex_pos)[np.newaxis, :], island_id
        )[0]

    def detect_simplified_islands(self, faces, face_uvs, face_to_orig_island):
        faces = np.asarray(faces)
        n_faces = len(faces)

        # Corner index of each vertex within its face, for UV-threshold lookups
        face_vertex_indices = [
            {vertex: j for j, vertex in enumerate(face)} for face in faces
        ]

        # Adjacent face pairs sharing a manifold edge (exactly two incident faces)
        adjacency = trimesh.graph.face_adjacency(faces=faces)

        # Build connectivity matrix
        rows, cols = [], []

        for f1, f2 in adjacency:
            f1, f2 = int(f1), int(f2)

            # Always connect if from same original island
            if (face_to_orig_island[f1] >= 0 and
                face_to_orig_island[f1] == face_to_orig_island[f2]):
                rows.extend([f1, f2])
                cols.extend([f2, f1])
                continue

            # Check UV distance
            shared = set(face_vertex_indices[f1]) & set(face_vertex_indices[f2])
            if len(shared) >= 2:
                connected = True
                for v in shared:
                    idx1 = face_vertex_indices[f1][v]
                    idx2 = face_vertex_indices[f2][v]
                    if np.linalg.norm(face_uvs[f1, idx1] - face_uvs[f2, idx2]) > self.uv_threshold:
                        connected = False
                        break
                if connected:
                    rows.extend([f1, f2])
                    cols.extend([f2, f1])

        # Find connected components
        if rows:
            matrix = csr_matrix(([1]*len(rows), (rows, cols)), shape=(n_faces, n_faces))
            _, labels = connected_components(matrix, directed=False)

            islands = defaultdict(list)
            for i, label in enumerate(labels):
                islands[label].append(i)

            # Merge small islands
            return self._merge_small_islands(list(islands.values()), face_to_orig_island)

        return [[i] for i in range(n_faces)]

    def _merge_small_islands(self, islands, face_to_orig_island):
        large = [i for i, isl in enumerate(islands) if len(isl) >= self.min_island_size]
        small = [i for i, isl in enumerate(islands) if len(isl) < self.min_island_size]

        if not small:
            return islands

        # Merge mapping
        merge_map = {}
        for s_idx in small:
            s_orig = face_to_orig_island[islands[s_idx][0]]
            if s_orig >= 0:
                # Find large island with same original island
                for l_idx in large:
                    if face_to_orig_island[islands[l_idx][0]] == s_orig:
                        merge_map[s_idx] = l_idx
                        break

        # Execute merge
        result = []
        merged = set()

        for l_idx in large:
            merged_island = list(islands[l_idx])
            for s_idx, target in merge_map.items():
                if target == l_idx:
                    merged_island.extend(islands[s_idx])
                    merged.add(s_idx)
            result.append(merged_island)

        # Add unmerged small islands
        for s_idx in small:
            if s_idx not in merged:
                result.append(islands[s_idx])

        return result


def visualize_mesh_islands(mesh, islands, title="UV Islands", color_mapping=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print(f"Visualizing {title}...")
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    face_colors = np.ones((len(mesh.faces), 4)) * 0.8

    for idx, faces in enumerate(islands):
        color_idx = color_mapping.get(idx, idx) if color_mapping is not None else idx
        face_colors[faces] = colors[color_idx % len(colors)]

    poly = Poly3DCollection(mesh.vertices[mesh.faces],
                           facecolors=face_colors,
                           edgecolors='k',
                           linewidths=0.1,
                           alpha=0.8)
    ax.add_collection3d(poly)

    # Set axes
    bounds = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    ax.set_xlim(bounds[:, 0])
    ax.set_ylim(bounds[:, 1])
    ax.set_zlim(bounds[:, 2])
    ax.set_title(f"{title} ({len(islands)} islands)")
    plt.show()


def visualize_uv_map(uv_coords, face_uv_indices, islands, title="UV Map"):
    import matplotlib.pyplot as plt

    print(f"Visualizing {title}...")
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw UV boundaries for each face
    for face_uv_idx in face_uv_indices:
        face_uvs = uv_coords[face_uv_idx]
        tri = np.vstack([face_uvs, face_uvs[0]])
        ax.plot(tri[:, 0], tri[:, 1], 'k-', linewidth=0.3, alpha=0.7)

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"{title} ({len(islands)} islands)")
    ax.grid(True, alpha=0.3)
    plt.show()


def transfer_uvs(original_mesh, uv_coords, face_uv_indices,
                simplified_mesh, uv_threshold=0.01):
    # Create UV island manager
    manager = UVIslandManager(original_mesh, uv_coords, face_uv_indices, uv_threshold)

    # Compute simplified face centers
    simp_centers = simplified_mesh.vertices[simplified_mesh.faces].mean(axis=1)

    # Find closest original faces
    _, closest = manager.face_kdtree.query(simp_centers)

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
        uvs = manager._interpolate_uv_batch(
            simplified_mesh.vertices[unique_ids], island_id
        )
        face_uvs[face_indices] = uvs[inverse.reshape(-1)].reshape(len(face_indices), 3, 2)

    # Detect islands in simplified mesh
    islands = manager.detect_simplified_islands(simplified_mesh.faces, face_uvs, face_to_orig_island)

    return face_uvs, islands, manager, face_to_orig_island


def save_obj_with_uvs(path, vertices, faces, face_uvs):
    # Extract unique UVs
    vertex_uv_map = {}
    unique_uvs = []
    face_uv_indices = []

    for face, uvs in zip(faces, face_uvs):
        face_uv_idx = []
        for v, uv in zip(face, uvs):
            key = (v, round(uv[0], 6), round(uv[1], 6))
            if key not in vertex_uv_map:
                vertex_uv_map[key] = len(unique_uvs)
                unique_uvs.append(uv)
            face_uv_idx.append(vertex_uv_map[key])
        face_uv_indices.append(face_uv_idx)

    # Write OBJ
    with open(path, 'w') as f:
        f.write(f"# UV Transfer Result\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        f.write("\n")
        for uv in unique_uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        f.write("\n")
        for face, uv_idx in zip(faces, face_uv_indices):
            f.write(f"f {face[0]+1}/{uv_idx[0]+1} {face[1]+1}/{uv_idx[1]+1} {face[2]+1}/{uv_idx[2]+1}\n")


def load_mesh_with_uvs(path):
    mesh = trimesh.load_mesh(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    # Extract UV
    uv_coords = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv_coords is None:
        raise ValueError("No UV data found")

    # Parse face UV indices from OBJ
    obj_str = trimesh.exchange.obj.export_obj(mesh, include_texture=True)
    matches = _FACE_UV_LINE.findall(obj_str)
    face_uv_indices = np.array(matches, dtype=int) - 1

    if len(face_uv_indices) != len(mesh.faces):
        raise ValueError(
            "Face UV index count mismatch: "
            f"faces={len(mesh.faces)}, face_uv_indices={len(face_uv_indices)}"
        )

    return mesh, uv_coords, face_uv_indices


def create_island_color_mapping(islands, face_to_orig_island):
    color_mapping = {}

    for simp_idx, faces in enumerate(islands):
        if not faces:
            continue

        # Check which original island these simplified island faces mainly come from
        orig_islands = [
            face_to_orig_island[face_idx]
            for face_idx in faces
            if face_to_orig_island[face_idx] >= 0
        ]

        if orig_islands:
            # Find most common original island index
            color_mapping[simp_idx] = Counter(orig_islands).most_common(1)[0][0]
        else:
            # Use self index if no original island
            color_mapping[simp_idx] = simp_idx

    return color_mapping


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
        # Create color mapping for original and simplified islands
        color_mapping = create_island_color_mapping(islands, face_to_orig_island)

        # Original mesh displays colors in default index order
        visualize_mesh_islands(original_mesh, manager.islands, "Original UV Islands")

        # Simplified mesh displays with colors matching original
        visualize_mesh_islands(simplified_mesh, islands, "Simplified UV Islands", color_mapping)

        visualize_uv_map(uv_coords, face_uv_indices, manager.islands, "Original UV Map")
        visualize_uv_map(face_uvs.reshape(-1, 2),
                        np.arange(len(simplified_mesh.faces) * 3).reshape(-1, 3),
                        islands, "Simplified UV Map")

