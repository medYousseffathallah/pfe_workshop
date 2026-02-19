import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from matplotlib.collections import LineCollection
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

OUTPUT_DIR = r"C:\Users\youss\OneDrive\Desktop\pfe\pfe_preparation\images"

def draw_matrix(ax, x, y, matrix, title="", cell_width=0.6, cell_height=0.4):
    rows = len(matrix)
    cols = len(matrix[0])
    
    total_width = cols * cell_width
    total_height = rows * cell_height
    
    start_x = x - total_width / 2
    start_y = y + total_height / 2
    
    for i in range(rows):
        for j in range(cols):
            cell_x = start_x + j * cell_width + cell_width / 2
            cell_y = start_y - i * cell_height - cell_height / 2
            ax.text(cell_x, cell_y, matrix[i][j], fontsize=10, ha='center', va='center')
    
    ax.plot([start_x - 0.05, start_x - 0.05], [start_y, start_y - total_height], 'k-', linewidth=1.5)
    ax.plot([start_x - 0.15, start_x - 0.15], [start_y, start_y - total_height], 'k-', linewidth=1.5)
    ax.plot([start_x + total_width + 0.05, start_x + total_width + 0.05], [start_y, start_y - total_height], 'k-', linewidth=1.5)
    ax.plot([start_x + total_width + 0.15, start_x + total_width + 0.15], [start_y, start_y - total_height], 'k-', linewidth=1.5)
    
    if title:
        ax.text(x, start_y + 0.3, title, fontsize=12, ha='center', va='bottom', fontweight='bold')

def create_homography_transform():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    ax1.set_xlim(0, 800)
    ax1.set_ylim(600, 0)
    ax1.set_xlabel('u (pixels)')
    ax1.set_ylabel('v (pixels)')
    ax1.set_title('Image Coordinates', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    corners_img = np.array([[100, 100], [700, 120], [650, 500], [50, 480]])
    perspective_quad = Polygon(corners_img, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2, linestyle='--', alpha=0.5)
    ax1.add_patch(perspective_quad)
    
    colors = ['red', 'green', 'orange', 'purple']
    labels = ['$P_1$', '$P_2$', '$P_3$', '$P_4$']
    for i, (pt, color, label) in enumerate(zip(corners_img, colors, labels)):
        ax1.plot(pt[0], pt[1], 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        offset = [(15, 15), (15, -20), (15, 15), (-40, 15)][i]
        ax1.annotate(label, pt, textcoords="offset points", xytext=offset, fontsize=13, fontweight='bold', color=color)
    
    ax1.annotate('Perspective View\n(Road Region)', (350, 300), fontsize=11, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='orange'))
    
    ax2 = axes[1]
    ax2.set_xlim(-0.1, 1.3)
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('World Coordinates', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    corners_world = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    world_quad = Polygon(corners_world, fill=True, facecolor='lightgreen', edgecolor='blue', linewidth=2, alpha=0.5)
    ax2.add_patch(world_quad)
    
    world_labels = ["$P_1'$", "$P_2'$", "$P_3'$", "$P_4'$"]
    offsets = [(-0.08, -0.08), (1.08, -0.08), (1.08, 1.05), (-0.12, 1.05)]
    for i, (pt, color, label) in enumerate(zip(corners_world, colors, world_labels)):
        ax2.plot(pt[0], pt[1], 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        ax2.text(offsets[i][0], offsets[i][1], label, fontsize=13, fontweight='bold', color=color)
    
    ax2.annotate('Top-Down View\n(Rectified)', (0.5, 0.5), fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue'))
    
    ax3 = axes[2]
    ax3.set_xlim(-0.5, 5)
    ax3.set_ylim(-0.5, 4)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Homography Matrix', fontsize=13, fontweight='bold')
    
    h_matrix = [['$h_{11}$', '$h_{12}$', '$h_{13}$'],
                ['$h_{21}$', '$h_{22}$', '$h_{23}$'],
                ['$h_{31}$', '$h_{32}$', '$h_{33}$']]
    draw_matrix(ax3, 2.5, 3.2, h_matrix, title='$H$', cell_width=0.7, cell_height=0.45)
    
    ax3.text(2.5, 2.0, r'$[xw,\ yw,\ w]^T = H \cdot [u,\ v,\ 1]^T$', 
            fontsize=13, ha='center', va='center')
    
    ax3.text(2.5, 1.2, r'World: $(X, Y) = (\frac{xw}{w}, \frac{yw}{w})$', 
            fontsize=12, ha='center', va='center')
    
    ax3.annotate('', xy=(4.3, 3.2), xytext=(4.8, 3.2),
                arrowprops=dict(arrowstyle='<-', color='green', lw=2))
    ax3.text(4.9, 3.2, '3×3', fontsize=11, ha='left', va='center', fontweight='bold')
    
    ax3.text(2.5, 0.4, '4-point correspondence:\n$H$ computed from 4 point pairs', 
            fontsize=10, ha='center', va='center', style='italic', color='gray')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'homography_transform.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def create_speed_estimation_workflow():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    boxes = [
        ('Video\nFrame', (1, 3), '#AED6F1'),
        ('Object\nDetection', (3.5, 3), '#ABEBC6'),
        ('Tracking\n(Kalman)', (6, 3), '#F9E79F'),
        ('World\nMapping', (8.5, 3), '#F5B7B1'),
        ('Speed\nCalculation', (11, 3), '#D7BDE2'),
        ('Output', (13, 3), '#D5D8DC')
    ]
    
    for label, (x, y), color in boxes:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow_positions = [(1.9, 3), (4.4, 3), (6.9, 3), (9.4, 3), (11.9, 3)]
    arrow_ends = [(2.6, 3), (5.1, 3), (7.6, 3), (10.1, 3), (12.2, 3)]
    for start, end in zip(arrow_positions, arrow_ends):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#1A5276', lw=3, mutation_scale=20))
    
    annotations = [
        (2.5, 4.5, 'YOLOv8', '#145A32'),
        (5, 4.5, 'DeepSORT', '#145A32'),
        (7.5, 4.5, 'Homography H', '#145A32'),
        (10, 4.5, r'$v = \frac{d}{\Delta t}$', '#145A32'),
        (3.5, 1.5, 'Bounding Boxes\n+ Confidences', '#5D6D7E'),
        (6, 1.5, 'Track IDs\n+ State Vectors', '#5D6D7E'),
        (8.5, 1.5, 'Pixel to Meter\nConversion', '#5D6D7E'),
        (11, 1.5, 'Speed (km/h)\n+ Trajectory', '#5D6D7E')
    ]
    
    for x, y, text, color in annotations:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5))
    
    ax.set_title('Speed Estimation Pipeline Workflow', fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'speed_estimation_workflow.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def create_trajectory_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    t = np.linspace(0, 4*np.pi, 100)
    x = 5 * t / (4*np.pi) + 1.5 * np.sin(t)
    y = 2 * np.cos(t/2) + 2
    
    ax1.fill(x, y, alpha=0.2, color='blue')
    ax1.plot(x, y, 'b-', linewidth=2, label='Vehicle Trajectory')
    
    sample_indices = np.linspace(0, len(t)-1, 20, dtype=int)
    ax1.scatter(x[sample_indices], y[sample_indices], c='red', s=60, zorder=5, label='Position Samples', edgecolor='black')
    
    idx = 60
    ax1.annotate('', xy=(x[idx+5], y[idx+5]), xytext=(x[idx], y[idx]),
                arrowprops=dict(arrowstyle='->', color='green', lw=3, mutation_scale=20))
    ax1.text((x[idx]+x[idx+5])/2 + 0.4, (y[idx]+y[idx+5])/2 + 0.4, 
            r'$\vec{v}$ (velocity)', fontsize=12, color='green', fontweight='bold')
    
    ax1.scatter([x[idx]], [y[idx]], c='yellow', s=200, edgecolor='black', linewidth=2, zorder=10)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Vehicle Trajectory with Sampled Positions', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    ax2 = axes[1]
    
    t_short = np.linspace(0, np.pi/2, 50)
    arc_radius = 2
    arc_x = arc_radius * np.cos(t_short + np.pi)
    arc_y = arc_radius * np.sin(t_short + np.pi)
    
    ax2.fill(arc_x, arc_y, alpha=0.15, color='blue')
    ax2.plot(arc_x, arc_y, 'b-', linewidth=2.5)
    
    p1_idx = 10
    p2_idx = 25
    p3_idx = 40
    
    ax2.scatter([arc_x[p1_idx], arc_x[p2_idx], arc_x[p3_idx]], 
               [arc_y[p1_idx], arc_y[p2_idx], arc_y[p3_idx]], 
               c=['red', 'green', 'orange'], s=120, zorder=5, edgecolor='black', linewidth=1.5)
    ax2.text(arc_x[p1_idx]-0.4, arc_y[p1_idx]-0.3, '$P_{i-1}$', fontsize=13, fontweight='bold')
    ax2.text(arc_x[p2_idx]+0.2, arc_y[p2_idx]-0.4, '$P_i$', fontsize=13, fontweight='bold')
    ax2.text(arc_x[p3_idx]+0.2, arc_y[p3_idx]+0.2, '$P_{i+1}$', fontsize=13, fontweight='bold')
    
    vec1_start = np.array([arc_x[p2_idx], arc_y[p2_idx]])
    vec1_end = np.array([arc_x[p1_idx], arc_y[p1_idx]])
    vec2_end = np.array([arc_x[p3_idx], arc_y[p3_idx]])
    
    vec1 = vec1_end - vec1_start
    vec2 = vec2_end - vec1_start
    
    vec1_norm = vec1 / np.linalg.norm(vec1) * 1.5
    vec2_norm = vec2 / np.linalg.norm(vec2) * 1.5
    
    ax2.annotate('', xy=vec1_start - vec1_norm, xytext=vec1_start,
                arrowprops=dict(arrowstyle='->', color='red', lw=3, mutation_scale=20))
    ax2.annotate('', xy=vec1_start + vec2_norm, xytext=vec1_start,
                arrowprops=dict(arrowstyle='->', color='orange', lw=3, mutation_scale=20))
    
    ax2.text(-3.8, -0.3, r'$\vec{d}_1$', fontsize=13, color='red', fontweight='bold')
    ax2.text(0.6, -1.3, r'$\vec{d}_2$', fontsize=13, color='orange', fontweight='bold')
    
    angle_arc = np.linspace(np.arctan2(vec1[1], vec1[0]) + np.pi, 
                           np.arctan2(vec2[1], vec2[0]) + np.pi, 20)
    arc_draw_x = arc_x[p2_idx] + 0.5 * np.cos(angle_arc)
    arc_draw_y = arc_y[p2_idx] + 0.5 * np.sin(angle_arc)
    ax2.plot(arc_draw_x, arc_draw_y, 'purple', linewidth=2.5)
    ax2.text(arc_x[p2_idx] + 0.7, arc_y[p2_idx] + 0.6, r'$\theta_{turn}$', fontsize=13, color='purple', fontweight='bold')
    
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('Heading Angle & Turn Angle Calculation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-4.5, 2)
    ax2.set_ylim(-3, 2)
    
    ax2.text(-4.2, 1.7, r'$\theta_{heading} = \arctan2(\Delta y, \Delta x)$', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#FCF3CF', alpha=0.9, edgecolor='orange', linewidth=1.5))
    ax2.text(-4.2, 0.9, r'$\theta_{turn} = |\theta_2 - \theta_1|$', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.9, edgecolor='green', linewidth=1.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'trajectory_analysis.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def create_curvature_calculation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    
    t = np.linspace(0, np.pi, 100)
    radius = 3
    center = (0, 0)
    curve_x = center[0] + radius * np.cos(t)
    curve_y = center[1] + radius * np.sin(t)
    
    ax1.fill(curve_x, curve_y, alpha=0.15, color='blue')
    ax1.plot(curve_x, curve_y, 'b-', linewidth=2.5, label='Trajectory')
    
    idx = 50
    point = np.array([curve_x[idx], curve_y[idx]])
    
    dx = np.gradient(curve_x)
    dy = np.gradient(curve_y)
    vx = dx[idx]
    vy = dy[idx]
    v_mag = np.sqrt(vx**2 + vy**2)
    vx_norm, vy_norm = vx/v_mag, vy/v_mag
    
    v_scale = 2
    ax1.annotate('', xy=(point[0] + vx_norm*v_scale, point[1] + vy_norm*v_scale),
                xytext=point,
                arrowprops=dict(arrowstyle='->', color='green', lw=3.5, mutation_scale=25))
    ax1.text(point[0] + vx_norm*v_scale + 0.4, point[1] + vy_norm*v_scale + 0.4,
            r'$\vec{v}$ (velocity)', fontsize=12, color='green', fontweight='bold')
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ax_val = ddx[idx]
    ay_val = ddy[idx]
    a_mag = np.sqrt(ax_val**2 + ay_val**2)
    ax_norm, ay_norm = ax_val/a_mag, ay_val/a_mag
    
    a_scale = 1.5
    ax1.annotate('', xy=(point[0] + ax_norm*a_scale, point[1] + ay_norm*a_scale),
                xytext=point,
                arrowprops=dict(arrowstyle='->', color='red', lw=3.5, mutation_scale=25))
    ax1.text(point[0] + ax_norm*a_scale + 0.4, point[1] + ay_norm*a_scale - 0.6,
            r'$\vec{a}$ (acceleration)', fontsize=12, color='red', fontweight='bold')
    
    ax1.scatter([point[0]], [point[1]], c='yellow', s=250, edgecolor='black', linewidth=2, zorder=10)
    
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius * np.cos(circle_theta)
    circle_y = center[1] + radius * np.sin(circle_theta)
    ax1.plot(circle_x, circle_y, 'gray', linewidth=1, linestyle='--', alpha=0.5)
    
    ax1.plot([center[0], point[0]], [center[1], point[1]], 'purple', linewidth=2, linestyle=':')
    ax1.scatter([center[0]], [center[1]], c='purple', s=100, marker='x', linewidth=3)
    ax1.text(center[0]-0.6, center[1]-0.6, 'O', fontsize=13, color='purple', fontweight='bold')
    ax1.text((center[0]+point[0])/2 - 1.0, (center[1]+point[1])/2 + 0.2, r'$R = \frac{1}{\kappa}$', 
            fontsize=12, color='purple', fontweight='bold')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Velocity and Acceleration Vectors', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    ax2.text(5, 7.5, 'Curvature Formulas', fontsize=15, fontweight='bold', ha='center')
    
    formulas = [
        (r'$\kappa = \frac{|\vec{v} \times \vec{a}|}{|\vec{v}|^3}$', 6),
        (r'$\kappa = \frac{|v_x a_y - v_y a_x|}{(v_x^2 + v_y^2)^{3/2}}$', 4.5),
        (r'$\omega = v \cdot \kappa$', 3),
        (r'$R = \frac{1}{\kappa} = \frac{|\vec{v}|^3}{|\vec{v} \times \vec{a}|}$', 1.5),
    ]
    
    colors = ['#FADBD8', '#D5F5E3', '#D6EAF8', '#FCF3CF']
    edges = ['#E74C3C', '#27AE60', '#3498DB', '#F39C12']
    
    for (formula, y_pos), color, edge in zip(formulas, colors, edges):
        ax2.text(5, y_pos, formula, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9, edgecolor=edge, linewidth=2))
    
    ax2.text(5, 0.3, r'$\kappa$: curvature (1/m)    $\omega$: angular rate (rad/s)    R: radius of curvature (m)',
            fontsize=10, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'curvature_calculation.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def create_speed_smoothing():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    true_speed = 60 + 10 * np.sin(t * 0.5) + 5 * np.sin(t * 1.5)
    noise = np.random.normal(0, 5, len(t))
    raw_speed = true_speed + noise
    
    ax1 = axes[0, 0]
    ax1.plot(t, raw_speed, 'b-', linewidth=1, alpha=0.7, label='Raw (Noisy) Speed')
    ax1.plot(t, true_speed, 'g-', linewidth=2.5, label='True Speed')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Speed (km/h)', fontsize=12)
    ax1.set_title('Raw vs True Speed', fontsize=13, fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(t, raw_speed, true_speed, alpha=0.1, color='blue')
    
    def ema(data, alpha):
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    ax2 = axes[0, 1]
    alphas = [0.1, 0.3, 0.5]
    colors = ['#E74C3C', '#F39C12', '#8E44AD']
    
    ax2.plot(t, raw_speed, 'b-', linewidth=1, alpha=0.3, label='Raw Speed')
    for alpha, color in zip(alphas, colors):
        smoothed = ema(raw_speed, alpha)
        ax2.plot(t, smoothed, color=color, linewidth=2.5, label=fr'EMA ($\alpha$={alpha})')
    
    ax2.plot(t, true_speed, 'g--', linewidth=2, alpha=0.7, label='True Speed')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Speed (km/h)', fontsize=12)
    ax2.set_title('EMA Smoothing with Different ' + r'$\alpha$' + ' Values', fontsize=13, fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    
    def double_ema(data, alpha):
        single = ema(data, alpha)
        double = ema(single, alpha)
        return 2 * single - double
    
    alpha = 0.3
    single_ema = ema(raw_speed, alpha)
    double_ema_result = double_ema(raw_speed, alpha)
    
    ax3.plot(t, raw_speed, 'b-', linewidth=1, alpha=0.3, label='Raw Speed')
    ax3.plot(t, single_ema, '#E74C3C', linewidth=2.5, label='Single EMA')
    ax3.plot(t, double_ema_result, '#8E44AD', linewidth=2.5, label='Double EMA (Corrected)')
    ax3.plot(t, true_speed, 'g--', linewidth=2.5, label='True Speed')
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Speed (km/h)', fontsize=12)
    ax3.set_title('Single vs Double EMA Smoothing', fontsize=13, fontweight='bold')
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    
    ax4.text(5, 7.3, 'Exponential Moving Average (EMA)', fontsize=15, fontweight='bold', ha='center')
    
    formula_box = FancyBboxPatch((0.5, 2.5), 9, 4.3,
                                  boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='#FCF3CF', alpha=0.9, edgecolor='#F39C12', linewidth=2)
    ax4.add_patch(formula_box)
    
    ax4.text(5, 6.2, r'EMA Formula:', fontsize=12, ha='center', fontweight='bold')
    ax4.text(5, 5.5, r'$S_t = \alpha \cdot X_t + (1-\alpha) \cdot S_{t-1}$', fontsize=13, ha='center')
    
    ax4.text(5, 4.5, r'Where:', fontsize=11, ha='center', style='italic')
    ax4.text(5, 3.9, r'$X_t$: Current observation (raw speed)', fontsize=10, ha='center')
    ax4.text(5, 3.4, r'$S_t$: Smoothed value at time t', fontsize=10, ha='center')
    ax4.text(5, 2.9, r'$\alpha$: Smoothing factor $(0 < \alpha < 1)$', fontsize=10, ha='center')
    
    ax4.text(5, 1.8, 'Parameter Selection:', fontsize=11, ha='center', fontweight='bold')
    ax4.text(5, 1.3, r'Higher $\alpha$: More responsive, less smoothing', fontsize=10, ha='center', color='#E74C3C')
    ax4.text(5, 0.9, r'Lower $\alpha$: More smoothing, higher lag', fontsize=10, ha='center', color='#27AE60')
    ax4.text(5, 0.4, 'Typical values: ' + r'$\alpha \in [0.1, 0.5]$', fontsize=10, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'speed_smoothing.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating mathematical visualization diagrams...")
    
    paths = []
    
    print("\n1. Creating homography transform diagram...")
    paths.append(create_homography_transform())
    
    print("2. Creating speed estimation workflow diagram...")
    paths.append(create_speed_estimation_workflow())
    
    print("3. Creating trajectory analysis diagram...")
    paths.append(create_trajectory_analysis())
    
    print("4. Creating curvature calculation diagram...")
    paths.append(create_curvature_calculation())
    
    print("5. Creating speed smoothing diagram...")
    paths.append(create_speed_smoothing())
    
    print("\n" + "="*60)
    print("Generated files:")
    print("="*60)
    for path in paths:
        print(f"  - {path}")
        print(f"  - {path.replace('.png', '.svg')}")
    
    return paths

if __name__ == "__main__":
    main()
