import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from collections import defaultdict, deque
import trimesh
import argparse
from tqdm import tqdm


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
        
        # Build UV face graph
        uv_graph = defaultdict(set)
        face_to_uv_verts = {}
        
        for face_idx, face_uv_idx in enumerate(self.face_uv_indices):
            uv_verts = [inverse[idx] for idx in face_uv_idx]
            if len(set(uv_verts)) == 3:  # Valid faces only
                face_to_uv_verts[face_idx] = uv_verts
                for i in range(3):
                    v1, v2 = uv_verts[i], uv_verts[(i+1)%3]
                    uv_graph[v1].add(v2)
                    uv_graph[v2].add(v1)
        
        # Find connected components using BFS
        visited_verts = set()
        islands = []
        
        for start_vert in range(len(unique_uvs)):
            if start_vert not in visited_verts and start_vert in uv_graph:
                # BFS
                island_verts = set()
                queue = deque([start_vert])
                visited_verts.add(start_vert)
                
                while queue:
                    v = queue.popleft()
                    island_verts.add(v)
                    for neighbor in uv_graph[v]:
                        if neighbor not in visited_verts:
                            visited_verts.add(neighbor)
                            queue.append(neighbor)
                
                # Find faces containing these vertices
                island_faces = []
                for face_idx, uv_verts in face_to_uv_verts.items():
                    if all(v in island_verts for v in uv_verts):
                        island_faces.append(face_idx)
                
                if island_faces:
                    islands.append(island_faces)
        
        return islands
    
    def interpolate_uv(self, vertex_pos, island_id=None):
        # Search within specified island if provided
        if island_id is not None and island_id in self.island_kdtrees:
            kdtree = self.island_kdtrees[island_id]
            faces = self.island_faces[island_id]
            k = min(10, len(faces))
            _, indices = kdtree.query(vertex_pos, k=k)
            candidate_faces = faces[indices]
        else:
            # Global search
            _, candidate_faces = self.face_kdtree.query(vertex_pos, k=5)
        
        # Find closest face and compute barycentric coordinates
        best_face = candidate_faces[0]
        face_verts = self.mesh.vertices[self.mesh.faces[best_face]]
        
        # Compute barycentric coordinates (simplified)
        v0, v1, v2 = face_verts
        v0v1, v0v2 = v1 - v0, v2 - v0
        normal = np.cross(v0v1, v0v2)
        
        if np.linalg.norm(normal) > 1e-10:
            normal /= np.linalg.norm(normal)
            # Project to plane
            d = np.dot(vertex_pos - v0, normal)
            proj = vertex_pos - d * normal
            
            # Barycentric coordinates
            v0p = proj - v0
            d00, d01, d11 = np.dot(v0v1, v0v1), np.dot(v0v1, v0v2), np.dot(v0v2, v0v2)
            d20, d21 = np.dot(v0p, v0v1), np.dot(v0p, v0v2)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) > 1e-10:
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                bary = np.clip([u, v, w], 0, 1)
                bary /= bary.sum()
            else:
                bary = np.array([1/3, 1/3, 1/3])
        else:
            bary = np.array([1/3, 1/3, 1/3])
        
        # UV interpolation
        face_uv_idx = self.face_uv_indices[best_face]
        face_uvs = self.uv_coords[face_uv_idx]
        return np.sum(bary[:, np.newaxis] * face_uvs, axis=0)
    
    def detect_simplified_islands(self, faces, face_uvs, face_to_orig_island):
        n_faces = len(faces)
        
        # Find adjacent faces
        edge_to_faces = defaultdict(list)
        for i, face in enumerate(faces):
            for j in range(3):
                edge = tuple(sorted([face[j], face[(j+1)%3]]))
                edge_to_faces[edge].append(i)
        
        # Build connectivity matrix
        rows, cols = [], []
        
        for edge_faces in edge_to_faces.values():
            if len(edge_faces) == 2:
                f1, f2 = edge_faces
                
                # Always connect if from same original island
                if (face_to_orig_island[f1] >= 0 and 
                    face_to_orig_island[f1] == face_to_orig_island[f2]):
                    rows.extend([f1, f2])
                    cols.extend([f2, f1])
                    continue
                
                # Check UV distance
                shared = list(set(faces[f1]) & set(faces[f2]))
                if len(shared) >= 2:
                    connected = True
                    for v in shared:
                        idx1 = list(faces[f1]).index(v)
                        idx2 = list(faces[f2]).index(v)
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
            return self._merge_small_islands(list(islands.values()), face_uvs, face_to_orig_island)
        
        return [[i] for i in range(n_faces)]
    
    def _merge_small_islands(self, islands, face_uvs, face_to_orig_island):
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
    print(f"Visualizing {title}...")
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    face_colors = np.ones((len(mesh.faces), 4)) * 0.8
    
    if color_mapping is not None:
        # Use custom color mapping
        for idx, faces in enumerate(islands):
            if idx in color_mapping:
                color_idx = color_mapping[idx]
                face_colors[faces] = colors[color_idx % len(colors)]
            else:
                # Default color if no mapping
                face_colors[faces] = colors[idx % len(colors)]
    else:
        # Default color mapping
        for idx, faces in enumerate(islands):
            face_colors[faces] = colors[idx % len(colors)]
    
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
    face_to_orig_island = np.full(len(simplified_mesh.faces), -1)
    
    # Group by island
    faces_by_island = defaultdict(list)
    for i, orig_face in enumerate(closest):
        island_id = manager.face_to_island[orig_face]
        face_to_orig_island[i] = island_id
        if island_id >= 0:
            faces_by_island[island_id].append(i)
    
    # UV interpolation
    for island_id, face_indices in tqdm(faces_by_island.items(), desc="Processing islands"):
        for face_idx in face_indices:
            for v_idx, vertex in enumerate(simplified_mesh.vertices[simplified_mesh.faces[face_idx]]):
                face_uvs[face_idx, v_idx] = manager.interpolate_uv(vertex, island_id)
    
    # Handle unassigned faces
    unassigned = np.where(face_to_orig_island == -1)[0]
    for face_idx in unassigned:
        for v_idx, vertex in enumerate(simplified_mesh.vertices[simplified_mesh.faces[face_idx]]):
            face_uvs[face_idx, v_idx] = manager.interpolate_uv(vertex)
    
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
    if hasattr(mesh.visual, 'uv'):
        uv_coords = mesh.visual.uv
        
        # Parse face UV indices from OBJ
        obj_str = trimesh.exchange.obj.export_obj(mesh, include_texture=True)
        face_uv_indices = []
        
        for line in obj_str.splitlines():
            if line.startswith('f '):
                indices = []
                for part in line.split()[1:]:
                    if '/' in part:
                        components = part.split('/')
                        if len(components) >= 2 and components[1]:
                            indices.append(int(components[1]) - 1)
                if len(indices) == 3:
                    face_uv_indices.append(indices)
        
        return mesh, uv_coords, np.array(face_uv_indices)
    
    raise ValueError("No UV data found")


def create_island_color_mapping(islands, face_to_orig_island):
    color_mapping = {}
    
    for simp_idx, faces in enumerate(islands):
        if not faces:
            continue
            
        # Check which original island these simplified island faces mainly come from
        orig_islands = []
        for face_idx in faces:
            orig_island = face_to_orig_island[face_idx]
            if orig_island >= 0:
                orig_islands.append(orig_island)
        
        if orig_islands:
            # Find most common original island index
            from collections import Counter
            counter = Counter(orig_islands)
            most_common_orig = counter.most_common(1)[0][0]
            color_mapping[simp_idx] = most_common_orig
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
