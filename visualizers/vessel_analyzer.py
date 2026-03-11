"""
Retinal Vessel Analyzer - Backend Analysis Module
Handles segmentation model loading, vessel analysis, and dataset generation.
"""

import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, List, Tuple, Optional
from model import build_unet


# ================= CONFIG =================

checkpoint_path = "./models/checkpoint.pth"
image_size = (512, 512)

# BGR (OpenCV format)
BACKGROUND = [0, 0, 0]
ARTERY    = [255, 0, 0]
VEIN      = [0, 0, 255]
JUNCTION  = [0, 255, 0]

colormap = [
    BACKGROUND,
    ARTERY,
    VEIN,
    JUNCTION,
    [255, 255, 255]
]
num_classes = len(colormap)


# ================= RETINAL VESSEL ANALYZER =================

class RetinalVesselAnalyzer:
    """Analyzes retinal vessel maps and extracts parameters for simulation."""

    def __init__(self, binary_vessel_map: np.ndarray):
        self.vessel_map   = binary_vessel_map.astype(bool)
        self.skeleton     = None
        self.distance_map = None
        self.junctions    = []
        self.endpoints    = []
        self.segments     = []

    def compute_skeleton(self) -> np.ndarray:
        """Compute the skeleton (centerline) of vessels."""
        self.skeleton = skeletonize(self.vessel_map)
        return self.skeleton

    def compute_distance_map(self) -> np.ndarray:
        """Compute distance transform for diameter estimation."""
        self.distance_map = ndimage.distance_transform_edt(self.vessel_map)
        return self.distance_map

    def find_junctions_and_endpoints(self) -> Tuple[List, List]:
        """Find bifurcation points (junctions) and endpoints in the skeleton."""
        if self.skeleton is None:
            self.compute_skeleton()

        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        neighbor_count = ndimage.convolve(self.skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * self.skeleton

        junction_mask  = (neighbor_count > 2) & self.skeleton
        self.junctions = list(zip(*np.where(junction_mask)))

        endpoint_mask  = (neighbor_count == 1) & self.skeleton
        self.endpoints = list(zip(*np.where(endpoint_mask)))

        return self.junctions, self.endpoints

    def trace_segment(self, start: Tuple[int, int],
                      visited: set,
                      stop_at_junction: bool = True) -> List[Tuple[int, int]]:
        """Trace a vessel segment from a starting point."""
        if start in visited:
            return []

        path = [start]
        visited.add(start)
        current = start

        kernel_offsets = [(-1, -1), (-1, 0), (-1, 1),
                          (0,  -1),           (0,  1),
                          (1,  -1), (1,  0),  (1,  1)]

        while True:
            y, x      = current
            neighbors = []

            for dy, dx in kernel_offsets:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.skeleton.shape[0] and
                    0 <= nx < self.skeleton.shape[1] and
                    self.skeleton[ny, nx] and
                    (ny, nx) not in visited):
                    neighbors.append((ny, nx))

            if len(neighbors) == 0:
                break
            if len(neighbors) > 1 and stop_at_junction:
                break

            next_point = neighbors[0]
            path.append(next_point)
            visited.add(next_point)
            current = next_point

            neighbor_count = sum(
                1 for dy, dx in kernel_offsets
                if (0 <= current[0] + dy < self.skeleton.shape[0] and
                    0 <= current[1] + dx < self.skeleton.shape[1] and
                    self.skeleton[current[0] + dy, current[1] + dx])
            )

            if neighbor_count > 2 and stop_at_junction and len(path) > 1:
                break

        return path

    def extract_segments(self) -> List[Dict]:
        """Extract all vessel segments between junctions and endpoints."""
        if self.skeleton is None:
            self.compute_skeleton()
        if not self.junctions:
            self.find_junctions_and_endpoints()

        segments   = []
        visited    = set()
        segment_id = 0

        kernel_offsets = [(-1, -1), (-1, 0), (-1, 1),
                          (0,  -1),           (0,  1),
                          (1,  -1), (1,  0),  (1,  1)]

        for junction in self.junctions:
            y, x = junction
            for dy, dx in kernel_offsets:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.skeleton.shape[0] and
                    0 <= nx < self.skeleton.shape[1] and
                    self.skeleton[ny, nx] and
                    (ny, nx) not in visited):

                    path = self.trace_segment((ny, nx), visited, stop_at_junction=True)
                    if len(path) > 1:
                        segments.append({
                            'id':   segment_id,
                            'path': path,
                            'start': junction,
                            'end':   path[-1],
                            'type':  'junction_to_junction' if path[-1] in self.junctions
                                     else 'junction_to_endpoint'
                        })
                        segment_id += 1

        for endpoint in self.endpoints:
            if endpoint not in visited:
                path = self.trace_segment(endpoint, visited, stop_at_junction=True)
                if len(path) > 1:
                    segments.append({
                        'id':   segment_id,
                        'path': path,
                        'start': endpoint,
                        'end':   path[-1],
                        'type':  'endpoint_to_junction'
                    })
                    segment_id += 1

        self.segments = segments
        return segments

    def calculate_diameter(self, point: Tuple[int, int]) -> float:
        """Calculate vessel diameter at a given point using distance transform."""
        if self.distance_map is None:
            self.compute_distance_map()
        y, x = point
        return 2 * self.distance_map[y, x]

    def calculate_segment_parameters(self, segment: Dict) -> Dict:
        """Calculate all parameters for a vessel segment."""
        path      = segment['path']
        diameters = [self.calculate_diameter(p) for p in path]

        arc_length   = len(path) - 1
        chord_length = float(np.linalg.norm(
            np.array(path[-1]) - np.array(path[0])
        ))
        tortuosity = arc_length / chord_length if chord_length > 0 else 1.0

        curvatures = []
        if len(path) >= 3:
            for i in range(1, len(path) - 1):
                p1, p2, p3 = np.array(path[i-1]), np.array(path[i]), np.array(path[i+1])
                v1 = p2 - p1
                v2 = p3 - p2
                v1n = v1 / (np.linalg.norm(v1) + 1e-10)
                v2n = v2 / (np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
                curvatures.append(np.arccos(cos_angle))

        return {
            'segment_id':    int(segment['id']),
            'start_coord':   [int(segment['start'][0]), int(segment['start'][1])],
            'end_coord':     [int(segment['end'][0]),   int(segment['end'][1])],
            'segment_type':  segment['type'],
            'length_arc':    float(arc_length),
            'length_chord':  float(chord_length),
            'tortuosity':    float(tortuosity),
            'diameter_mean': float(np.mean(diameters)),
            'diameter_min':  float(np.min(diameters)),
            'diameter_max':  float(np.max(diameters)),
            'diameter_std':  float(np.std(diameters)),
            'curvature_mean': float(np.mean(curvatures)) if curvatures else 0.0,
            'curvature_max':  float(np.max(curvatures)) if curvatures else 0.0,
        }

    def calculate_bifurcation_parameters(self, junction: Tuple[int, int]) -> List[Dict]:
        """Calculate bifurcation parameters at a junction point."""
        connected = [s for s in self.segments
                     if s['start'] == junction or s['end'] == junction]

        if len(connected) < 2:
            return []

        bifurcations = []
        for i, seg1 in enumerate(connected):
            for seg2 in connected[i+1:]:

                def _vec(seg):
                    if seg['start'] == junction:
                        pts = seg['path'][:min(5, len(seg['path']))]
                        return np.array(pts[-1]) - np.array(pts[0])
                    else:
                        pts = seg['path'][-min(5, len(seg['path'])):]
                        return np.array(pts[0]) - np.array(pts[-1])

                v1n = _vec(seg1); v1n = v1n / (np.linalg.norm(v1n) + 1e-10)
                v2n = _vec(seg2); v2n = v2n / (np.linalg.norm(v2n) + 1e-10)

                angle_deg = float(np.degrees(np.arccos(
                    np.clip(np.dot(v1n, v2n), -1.0, 1.0)
                )))

                d1 = self.calculate_diameter(
                    junction if seg1['start'] == junction else seg1['end'])
                d2 = self.calculate_diameter(
                    junction if seg2['start'] == junction else seg2['end'])

                branching_ratio        = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
                murray_parent_diameter = float(2 * ((d1/2)**3 + (d2/2)**3) ** (1/3))

                bifurcations.append({
                    'junction_coord':                [int(junction[0]), int(junction[1])],
                    'segment1_id':                   int(seg1['id']),
                    'segment2_id':                   int(seg2['id']),
                    'bifurcation_angle_deg':          angle_deg,
                    'diameter1':                     float(d1),
                    'diameter2':                     float(d2),
                    'branching_ratio':               branching_ratio,
                    'murray_predicted_parent_diameter': murray_parent_diameter,
                })

        return bifurcations

    def generate_dataset(self) -> Dict:
        """Generate complete JSON dataset with all vessel parameters."""
        if self.skeleton     is None: self.compute_skeleton()
        if self.distance_map is None: self.compute_distance_map()
        if not self.junctions:        self.find_junctions_and_endpoints()
        if not self.segments:         self.extract_segments()

        segment_parameters = [self.calculate_segment_parameters(s) for s in self.segments]

        all_bifurcations = []
        for junction in self.junctions:
            all_bifurcations.extend(self.calculate_bifurcation_parameters(junction))

        all_diameters   = [s['diameter_mean'] for s in segment_parameters] or [0]
        all_tortuosities = [s['tortuosity']    for s in segment_parameters] or [0]
        all_lengths      = [s['length_arc']    for s in segment_parameters] or [0]

        def _stats(values):
            return {
                'mean': float(np.mean(values)),
                'std':  float(np.std(values)),
                'min':  float(np.min(values)),
                'max':  float(np.max(values)),
            }

        return {
            'metadata': {
                'image_shape':      list(self.vessel_map.shape),
                'total_segments':   len(segment_parameters),
                'total_junctions':  len(self.junctions),
                'total_endpoints':  len(self.endpoints),
                'total_bifurcations': len(all_bifurcations),
            },
            'global_statistics': {
                'diameter':       _stats(all_diameters),
                'tortuosity':     _stats(all_tortuosities),
                'segment_length': _stats(all_lengths),
            },
            'segments':     segment_parameters,
            'bifurcations': all_bifurcations,
        }


# ================= SEGMENTATION FUNCTIONS =================

def load_model():
    """Load the trained segmentation model."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = build_unet(num_classes=num_classes).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model_dict = model.state_dict()
        filtered   = {k: v for k, v in state_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        model.eval()

        return model, device
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        return None, None


def preprocess_image(image, device):
    """Preprocess image for model inference."""
    img     = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, image_size) / 255.0
    resized = np.transpose(resized, (2, 0, 1))
    resized = np.expand_dims(resized, axis=0).astype(np.float32)
    return torch.from_numpy(resized).to(device), img


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB image."""
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(colormap):
        rgb[mask == i] = c
    return rgb


def separate_artery_vein(mask_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Separate artery and vein masks from segmentation."""
    artery_mask = (
        (mask_rgb == ARTERY).all(axis=2) |
        (mask_rgb == JUNCTION).all(axis=2)
    )
    vein_mask = (
        (mask_rgb == VEIN).all(axis=2) |
        (mask_rgb == JUNCTION).all(axis=2)
    )

    artery_gray = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    artery_gray[artery_mask] = 255

    vein_gray = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    vein_gray[vein_mask] = 255

    return artery_gray, vein_gray


def segment_fundus(image, model, device):
    """Perform fundus image segmentation."""
    if model is None:
        h = image.shape[0] if isinstance(image, np.ndarray) else 512
        w = image.shape[1] if isinstance(image, np.ndarray) else 512
        dummy_mask = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_gray = np.zeros((h, w),    dtype=np.uint8)
        return dummy_mask, dummy_gray, dummy_gray, image

    tensor, original = preprocess_image(image, device)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    mask_rgb = mask_to_rgb(pred)
    mask_rgb = cv2.resize(mask_rgb, (original.shape[1], original.shape[0]))
    artery, vein = separate_artery_vein(mask_rgb)

    return mask_rgb, artery, vein, original


# ================= VISUALIZATION =================

def create_analysis_visualization(analyzer: RetinalVesselAnalyzer,
                                  vessel_type: str) -> Image.Image:
    """Create comprehensive 6-panel visualization for vessel analysis."""
    fig  = Figure(figsize=(15, 10))
    axes = fig.subplots(2, 3)

    # Panel 1 – vessel map
    axes[0, 0].imshow(analyzer.vessel_map, cmap='gray')
    axes[0, 0].set_title(f'{vessel_type} Vessel Map')
    axes[0, 0].axis('off')

    # Panel 2 – skeleton
    axes[0, 1].imshow(analyzer.skeleton, cmap='gray')
    axes[0, 1].set_title('Skeleton')
    axes[0, 1].axis('off')

    # Panel 3 – distance transform
    im = axes[0, 2].imshow(analyzer.distance_map, cmap='hot')
    axes[0, 2].set_title('Distance Transform')
    axes[0, 2].axis('off')
    fig.colorbar(im, ax=axes[0, 2])

    # Panel 4 – junctions & endpoints
    axes[1, 0].imshow(analyzer.skeleton, cmap='gray')
    if analyzer.junctions:
        jy, jx = zip(*analyzer.junctions)
        axes[1, 0].scatter(jx, jy, c='red',  s=30, marker='o', label='Junctions')
    if analyzer.endpoints:
        ey, ex = zip(*analyzer.endpoints)
        axes[1, 0].scatter(ex, ey, c='blue', s=30, marker='s', label='Endpoints')
    axes[1, 0].set_title('Junctions & Endpoints')
    axes[1, 0].legend()
    axes[1, 0].axis('off')

    # Panel 5 – segments
    axes[1, 1].imshow(analyzer.vessel_map, cmap='gray', alpha=0.3)
    if analyzer.segments:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(analyzer.segments)))
        for idx, segment in enumerate(analyzer.segments):
            if segment['path']:
                y_coords, x_coords = zip(*segment['path'])
                axes[1, 1].plot(x_coords, y_coords, color=colors[idx], linewidth=2)
    axes[1, 1].set_title(f'Segments (n={len(analyzer.segments)})')
    axes[1, 1].axis('off')

    # Panel 6 – statistics text
    dataset    = analyzer.generate_dataset()
    stats_text = f"""
{vessel_type} Statistics:

Diameter (pixels):
  Mean: {dataset['global_statistics']['diameter']['mean']:.2f}
  Std:  {dataset['global_statistics']['diameter']['std']:.2f}

Tortuosity:
  Mean: {dataset['global_statistics']['tortuosity']['mean']:.2f}
  Std:  {dataset['global_statistics']['tortuosity']['std']:.2f}

Segment Length:
  Mean: {dataset['global_statistics']['segment_length']['mean']:.2f}
  Std:  {dataset['global_statistics']['segment_length']['std']:.2f}

Counts:
  Segments:     {dataset['metadata']['total_segments']}
  Junctions:    {dataset['metadata']['total_junctions']}
  Endpoints:    {dataset['metadata']['total_endpoints']}
  Bifurcations: {dataset['metadata']['total_bifurcations']}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=9,
                    verticalalignment='center', family='monospace')
    axes[1, 2].axis('off')

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)


# ================= HIGH-LEVEL ANALYSIS ENTRY POINT =================

def run_full_analysis(image) -> Dict:
    """
    Run the complete segmentation + vessel analysis pipeline.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        Input fundus image (RGB).

    Returns
    -------
    dict with keys:
        mask_rgb_display  – np.ndarray (H,W,3) RGB segmentation overlay
        artery_display    – np.ndarray (H,W,3) greyscale artery mask as RGB
        vein_display      – np.ndarray (H,W,3) greyscale vein mask as RGB
        artery_viz        – PIL.Image  6-panel artery analysis figure
        vein_viz          – PIL.Image  6-panel vein analysis figure
        json_output       – str        pretty-printed JSON dataset
        summary           – str        human-readable text summary
    """
    model, device = load_model()

    mask_rgb, artery_mask, vein_mask, _ = segment_fundus(image, model, device)

    artery_binary = (artery_mask > 127).astype(np.uint8)
    vein_binary   = (vein_mask   > 127).astype(np.uint8)

    artery_analyzer = RetinalVesselAnalyzer(artery_binary)
    artery_dataset  = artery_analyzer.generate_dataset()
    artery_viz      = create_analysis_visualization(artery_analyzer, "Artery")

    vein_analyzer = RetinalVesselAnalyzer(vein_binary)
    vein_dataset  = vein_analyzer.generate_dataset()
    vein_viz      = create_analysis_visualization(vein_analyzer, "Vein")

    combined_dataset = {
        'arteries': artery_dataset,
        'veins':    vein_dataset,
        'combined_statistics': {
            'total_segments':   (artery_dataset['metadata']['total_segments'] +
                                 vein_dataset['metadata']['total_segments']),
            'total_junctions':  (artery_dataset['metadata']['total_junctions'] +
                                 vein_dataset['metadata']['total_junctions']),
            'total_bifurcations': (artery_dataset['metadata']['total_bifurcations'] +
                                   vein_dataset['metadata']['total_bifurcations']),
        }
    }

    a_stats = artery_dataset['global_statistics']
    v_stats = vein_dataset['global_statistics']
    a_meta  = artery_dataset['metadata']
    v_meta  = vein_dataset['metadata']
    c_stats = combined_dataset['combined_statistics']

    av_ratio = (a_stats['diameter']['mean'] / v_stats['diameter']['mean']
                if v_stats['diameter']['mean'] else 0.0)

    summary = f"""
═══════════════════════════════════════════════════
RETINAL VESSEL ANALYSIS SUMMARY
═══════════════════════════════════════════════════

ARTERIES:
─────────────────────────────────────────────────
• Segments:       {a_meta['total_segments']}
• Junctions:      {a_meta['total_junctions']}
• Bifurcations:   {a_meta['total_bifurcations']}
• Mean Diameter:  {a_stats['diameter']['mean']:.2f} pixels
• Mean Tortuosity:{a_stats['tortuosity']['mean']:.3f}
• Mean Length:    {a_stats['segment_length']['mean']:.2f} pixels

VEINS:
─────────────────────────────────────────────────
• Segments:       {v_meta['total_segments']}
• Junctions:      {v_meta['total_junctions']}
• Bifurcations:   {v_meta['total_bifurcations']}
• Mean Diameter:  {v_stats['diameter']['mean']:.2f} pixels
• Mean Tortuosity:{v_stats['tortuosity']['mean']:.3f}
• Mean Length:    {v_stats['segment_length']['mean']:.2f} pixels

COMBINED:
─────────────────────────────────────────────────
• Total Segments:     {c_stats['total_segments']}
• Total Junctions:    {c_stats['total_junctions']}
• Total Bifurcations: {c_stats['total_bifurcations']}

A/V Ratio: {av_ratio:.3f}
═══════════════════════════════════════════════════
"""

    mask_rgb_display = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    artery_display   = cv2.cvtColor(cv2.cvtColor(artery_mask, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    vein_display     = cv2.cvtColor(cv2.cvtColor(vein_mask,   cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

    return {
        'mask_rgb_display': mask_rgb_display,
        'artery_display':   artery_display,
        'vein_display':     vein_display,
        'artery_viz':       artery_viz,
        'vein_viz':         vein_viz,
        'json_output':      json.dumps(combined_dataset, indent=2),
        'summary':          summary,
    }
