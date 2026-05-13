import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def save_routing_animation(weights, task_name, output_filename="routing.gif", speed_ms=50):
    """
    ルーティングの重み（weights）を受け取り、アニメーションをGIFまたはMP4として保存します。
    
    Args:
        weights (numpy.ndarray or torch.Tensor): (seq_len, num_synapses) の形状を持つルーティング重み
        task_name (str): タスク名（タイトル用）
        output_filename (str): 保存するファイル名（例: "copy_routing.gif", "reverse.mp4"）
        speed_ms (int): アニメーションの更新間隔（ミリ秒）
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
        
    seq_len, num_synapses = weights.shape
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(-2, 12)
    ax.set_ylim(-1, max(seq_len, num_synapses))
    ax.axis('off')

    token_y = np.linspace(seq_len-1, 0, seq_len)
    synapse_y = np.linspace(num_synapses-1, 0, num_synapses)

    ax.scatter([0]*seq_len, token_y, s=300, c='#1f77b4', zorder=4)
    ax.scatter([10]*num_synapses, synapse_y, s=300, c='#ff7f0e', zorder=4)

    for i in range(seq_len):
        ax.text(-0.8, token_y[i], f'Token {i}', ha='right', va='center', fontsize=12)
    for j in range(num_synapses):
        ax.text(10.8, synapse_y[j], f'Synapse {j}', ha='left', va='center', fontsize=12)

    frames_per_token = 20
    total_frames = seq_len * frames_per_token

    particles, = ax.plot([], [], 'ro', ms=10, zorder=5)
    trails = []
    for _ in range(seq_len * 2):
        line, = ax.plot([], [], 'r-', alpha=0.6, lw=2, zorder=3)
        trails.append(line)

    connections = []
    for i in range(seq_len):
        targets = np.where(weights[i] > 0)[0]
        connections.append(targets)

    def init():
        particles.set_data([], [])
        for line in trails:
            line.set_data([], [])
        return [particles] + trails

    def update(frame):
        token_idx = frame // frames_per_token
        progress = (frame % frames_per_token) / (frames_per_token - 1)
        
        trail_idx = 0
        current_px, current_py = [], []
        
        for i in range(seq_len):
            targets = connections[i]
            for j in targets:
                start_x, start_y = 0, token_y[i]
                end_x, end_y = 10, synapse_y[j]
                
                if i < token_idx:
                    trails[trail_idx].set_data([start_x, end_x], [start_y, end_y])
                    trails[trail_idx].set_color('lightgray')
                elif i == token_idx:
                    cur_x = start_x + (end_x - start_x) * progress
                    cur_y = start_y + (end_y - start_y) * progress
                    trails[trail_idx].set_data([start_x, cur_x], [start_y, cur_y])
                    trails[trail_idx].set_color('red')
                    current_px.append(cur_x)
                    current_py.append(cur_y)
                else:
                    trails[trail_idx].set_data([], [])
                trail_idx += 1
                
        particles.set_data(current_px, current_py)
        ax.set_title(f'Routing for [{task_name}] Task - Processing Token {token_idx}', fontsize=14)
        return [particles] + trails

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=speed_ms)
    
    print(f"Saving animation to {output_filename}...")
    if output_filename.endswith(".gif"):
        # Pillow Writer を使用してGIFとして保存
        ani.save(output_filename, writer='pillow', fps=1000//speed_ms)
    elif output_filename.endswith(".mp4"):
        # ffmpeg を使用してMP4として保存
        ani.save(output_filename, writer='ffmpeg', fps=1000//speed_ms)
    else:
        print("拡張子は .gif または .mp4 を推奨します。")
        ani.save(output_filename)
        
    plt.close()
    print("Done!")

if __name__ == "__main__":
    # テスト用のダミーデータ
    dummy_weights = np.zeros((8, 16))
    dummy_weights[0, 2] = 1; dummy_weights[0, 5] = 1
    dummy_weights[1, 1] = 1; dummy_weights[1, 7] = 1
    dummy_weights[2, 3] = 1; dummy_weights[2, 9] = 1
    save_routing_animation(dummy_weights, "TEST", "test_routing.gif", speed_ms=40)
